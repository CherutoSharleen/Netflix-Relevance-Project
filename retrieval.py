"""Make data and representation ablations reproducible before relevance evaluation."""

import argparse
import csv
import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np
from nltk.stem import PorterStemmer
from numpy.typing import NDArray
from rank_bm25 import BM25Okapi

from film_data import DATA_PATH, FIELDS, FilmRecord, load_records, normalize_text


DEFAULT_FIELDS = ("title", "description", "listed_in", "cast", "director", "country")
WORD_PATTERN = re.compile(r"[^\W_]+", re.UNICODE)
STEMMER = PorterStemmer()


@dataclass(frozen=True)
class IndexConfig:
    fields: tuple[str, ...] = DEFAULT_FIELDS
    field_weights: dict[str, int] = field(default_factory=dict)
    tokenizer: str = "whitespace"
    lowercase: bool = True
    stemming: bool = False
    k1: float = 1.5
    b: float = 0.75
    epsilon: float = 0.25

    def __post_init__(self) -> None:
        if not self.fields or len(set(self.fields)) != len(self.fields):
            raise ValueError("Select at least one field, without duplicates")
        if any(name not in FIELDS for name in self.fields):
            raise ValueError(f"Unknown field; choose from {', '.join(FIELDS)}")
        if any(name not in self.fields for name in self.field_weights):
            raise ValueError("Weights may only refer to selected fields")
        if any(type(weight) is not int or weight < 1 for weight in self.field_weights.values()):
            raise ValueError("Field weights must be positive integers; omit a field to exclude it")
        if self.tokenizer not in {"whitespace", "words"}:
            raise ValueError("Tokenizer must be 'whitespace' or 'words'")
        if not math.isfinite(self.k1) or self.k1 <= 0:
            raise ValueError("k1 must be finite and positive")
        if not math.isfinite(self.b) or not 0 <= self.b <= 1:
            raise ValueError("b must be between 0 and 1")
        if not math.isfinite(self.epsilon) or self.epsilon < 0:
            raise ValueError("epsilon must be finite and nonnegative")


def tokenize(text: str, config: IndexConfig) -> list[str]:
    text = normalize_text(text)
    if config.lowercase:
        text = text.lower()
    tokens = text.split() if config.tokenizer == "whitespace" else WORD_PATTERN.findall(text)
    if config.stemming:
        return [STEMMER.stem(token, to_lowercase=config.lowercase) for token in tokens]
    return tokens


def document_tokens(record: FilmRecord, config: IndexConfig) -> list[str]:
    return [
        token
        for name in config.fields
        for token in tokenize(record[name], config) * config.field_weights.get(name, 1)
    ]


@dataclass(frozen=True)
class SearchResult:
    record: FilmRecord
    score: float


@dataclass
class FilmIndex:
    records: list[FilmRecord]
    config: IndexConfig
    model: BM25Okapi

    def scores(self, query: str) -> NDArray[np.float64]:
        return self.model.get_scores(tokenize(query, self.config))

    def search(self, query: str, k: int = 10) -> list[SearchResult]:
        if k < 1:
            raise ValueError("k must be positive")
        scores = self.scores(query)
        # Original CSV order breaks ties, including zero and negative scores.
        order = sorted(range(len(self.records)), key=lambda position: -scores[position])
        return [
            SearchResult(self.records[position], float(scores[position]))
            for position in order[:k]
        ]


def build_index(
    records: list[FilmRecord], config: IndexConfig | None = None,
) -> FilmIndex:
    config = config or IndexConfig()
    corpus = [document_tokens(record, config) for record in records]
    if not any(corpus):
        raise ValueError("Selected fields produce an empty vocabulary")
    model = BM25Okapi(corpus, k1=config.k1, b=config.b, epsilon=config.epsilon)
    return FilmIndex(records, config, model)


def parse_weights(values: Sequence[str]) -> dict[str, int]:
    weights: dict[str, int] = {}
    for value in values:
        name, separator, raw_weight = value.partition("=")
        if not separator or name in weights:
            raise ValueError("Use --weight FIELD=INTEGER once per field")
        try:
            weights[name] = int(raw_weight)
        except ValueError as error:
            raise ValueError(f"Invalid integer field weight: {value}") from error
    return weights


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=DATA_PATH)
    parser.add_argument("--fields", nargs="+", default=list(DEFAULT_FIELDS), choices=FIELDS)
    parser.add_argument("--weight", action="append", default=[], metavar="FIELD=INTEGER")
    parser.add_argument("--tokenizer", choices=("whitespace", "words"), default="whitespace")
    parser.add_argument("--stem", action="store_true", help="Apply English Porter stemming")
    parser.add_argument("--preserve-case", action="store_true")
    parser.add_argument("--k1", type=float, default=1.5)
    parser.add_argument("--b", type=float, default=0.75)
    parser.add_argument("--epsilon", type=float, default=0.25)
    parser.add_argument("--query", help="Run one query and exit; otherwise prompt interactively")
    parser.add_argument("-k", type=int, default=10)
    parser.add_argument("--audit-only", action="store_true", help="Print cleaning audit without indexing")
    args = parser.parse_args(argv)
    try:
        if args.k < 1:
            raise ValueError("k must be positive")
        config = IndexConfig(
            fields=tuple(args.fields), field_weights=parse_weights(args.weight),
            tokenizer=args.tokenizer, lowercase=not args.preserve_case,
            stemming=args.stem, k1=args.k1, b=args.b, epsilon=args.epsilon,
        )
        records, report = load_records(args.csv)
        print(json.dumps({"data": str(args.csv.resolve()),
                          "sha256": hashlib.sha256(args.csv.read_bytes()).hexdigest(),
                          "cleaning": asdict(report),
                          "config": asdict(config)}, indent=2))
        if args.audit_only:
            return
        index = build_index(records, config)
    except (OSError, UnicodeError, ValueError, csv.Error) as error:
        parser.error(str(error))
    print(f"Indexed {len(records)} records.")
    if args.query is not None:
        print_results(index, args.query, args.k)
        return
    while True:
        try:
            query = input("query> ").strip()
        except EOFError:
            break
        if query.lower() in {"", "q", "quit", "exit"}:
            break
        print_results(index, query, args.k)


def print_results(index: FilmIndex, query: str, k: int) -> None:
    tokens = tokenize(query, index.config)
    if not tokens or not any(token in index.model.idf for token in tokens):
        print("No query terms occur in the index; all scores are zero (not relevance labels).")
    for rank, result in enumerate(index.search(query, k), 1):
        record = result.record
        print(f"{rank:2}. [{result.score:8.4f}] {record['show_id']} "
              f"{record['title']} ({record['release_year']}, {record['type']})")
        print(f"    {record['listed_in']}")


if __name__ == "__main__":
    main()
