"""Freeze user-authored query partitions before retrieval settings can be tuned."""

import csv
import hashlib
import json
import math
import random
from pathlib import Path
from typing import Sequence

from film_data import normalize_text


SPLITS = ("dev", "heldout")


def fingerprint(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_table(path: Path, columns: Sequence[str]) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source, strict=True)
        if reader.fieldnames != list(columns):
            raise ValueError(f"{path}: expected CSV columns {', '.join(columns)}")
        rows: list[dict[str, str]] = []
        for number, row in enumerate(reader, 2):
            if None in row or any(value is None for value in row.values()):
                raise ValueError(f"{path}: malformed CSV record {number}")
            rows.append({column: row[column] for column in columns})
    return rows


def write_json(path: Path, value: object) -> None:
    # Exclusive creation protects frozen splits and sampled queues from replacement.
    with path.open("x", encoding="utf-8") as target:
        json.dump(value, target, indent=2, ensure_ascii=True)
        target.write("\n")


def read_json(path: Path) -> dict[str, object]:
    value: object = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def freeze_queries(
    source: Path, output: Path, corpus: Path, heldout_fraction: float = 0.2, seed: int = 42,
) -> None:
    if not math.isfinite(heldout_fraction) or not 0 < heldout_fraction < 1:
        raise ValueError("Held-out fraction must be strictly between zero and one")
    rows = read_table(source, ("query_id", "query"))
    if len(rows) < 2:
        raise ValueError("Supply at least two human-authored queries to split")
    ids: set[str] = set()
    texts: set[str] = set()
    for row in rows:
        row["query_id"] = row["query_id"].strip()
        row["query"] = normalize_text(row["query"])
        if not row["query_id"] or not row["query"]:
            raise ValueError("Query IDs and query text must not be blank")
        if row["query_id"] in ids or row["query"].casefold() in texts:
            raise ValueError("Duplicate query ID or text could leak across the query split")
        ids.add(row["query_id"])
        texts.add(row["query"].casefold())
    shuffled = sorted(ids)
    random.Random(seed).shuffle(shuffled)
    count = min(len(rows) - 1, max(1, math.ceil(len(rows) * heldout_fraction)))
    heldout = set(shuffled[:count])
    write_json(output, {
        "version": 1, "source_sha256": fingerprint(source),
        "corpus_sha256": fingerprint(corpus), "seed": seed,
        "heldout_fraction": heldout_fraction,
        "queries": [
            row | {"split": "heldout" if row["query_id"] in heldout else "dev"}
            for row in sorted(rows, key=lambda row: row["query_id"])
        ],
    })


def load_queries(path: Path, corpus: Path) -> dict[str, dict[str, str]]:
    frozen = read_json(path)
    if frozen.get("version") != 1 or frozen.get("corpus_sha256") != fingerprint(corpus):
        raise ValueError("Query split version or corpus fingerprint does not match")
    rows = frozen.get("queries")
    if not isinstance(rows, list) or not rows:
        raise ValueError("Frozen split contains no queries")
    queries: dict[str, dict[str, str]] = {}
    texts: set[str] = set()
    for row in rows:
        if not isinstance(row, dict) or set(row) != {"query_id", "query", "split"}:
            raise ValueError("Invalid frozen query schema")
        query_id, text, partition = row["query_id"], row["query"], row["split"]
        if not isinstance(query_id, str) or not isinstance(text, str) or not isinstance(partition, str):
            raise ValueError("Frozen query values must be strings")
        if not query_id.strip() or not text.strip() or not partition.strip():
            raise ValueError("Frozen query values must be nonempty strings")
        if query_id in queries or text.casefold() in texts:
            raise ValueError("Duplicate query ID or text in frozen split")
        if partition not in SPLITS:
            raise ValueError("Invalid query partition")
        queries[query_id] = {"query_id": query_id, "query": text, "split": partition}
        texts.add(text.casefold())
    if {row["split"] for row in queries.values()} != set(SPLITS):
        raise ValueError("Frozen split must contain development and held-out queries")
    return queries
