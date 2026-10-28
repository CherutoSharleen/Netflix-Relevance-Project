"""Persist human grades and blinded queues without inferring any relevance labels."""

import csv
import os
import random
import uuid
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from film_data import FIELDS, load_records
from query_sets import fingerprint, load_queries, read_json, read_table, write_json
from retrieval import build_index


LABEL_COLUMNS = (
    "query_id", "show_id", "grade", "timestamp", "pass_id",
    "split_sha256", "corpus_sha256", "pool_sha256",
)
Pair = tuple[str, str]
LabelKey = tuple[str, str, str]


@dataclass(frozen=True)
class Pool:
    kind: str
    pass_id: str
    partition: str
    split_sha256: str
    corpus_sha256: str
    source_pool_sha256: str


def create_pool(
    split_file: Path, corpus: Path, output: Path, partition: str, k: int, seed: int,
    additional_pairs: Path | None = None,
) -> None:
    if partition not in {"dev", "heldout"}:
        raise ValueError("Choose dev or heldout")
    if k < 1:
        raise ValueError("Candidate count must be positive")
    queries = load_queries(split_file, corpus)
    records, _ = load_records(corpus)
    if k > len(records):
        raise ValueError("Candidate count exceeds corpus size")
    index = build_index(records)
    pairs = [
        {"query_id": query_id, "show_id": result.record["show_id"]}
        for query_id, query in queries.items() if query["split"] == partition
        for result in index.search(query["query"], k)
    ]
    if additional_pairs is not None:
        record_ids = {row["show_id"] for row in records}
        seen = {(pair["query_id"], pair["show_id"]) for pair in pairs}
        for pair in read_table(additional_pairs, ("query_id", "show_id")):
            query_id, show_id = pair["query_id"], pair["show_id"]
            if query_id not in queries or queries[query_id]["split"] != partition:
                raise ValueError("Additional candidates must belong to the selected query partition")
            if show_id not in record_ids:
                raise ValueError(f"Unknown additional candidate record: {show_id}")
            if (query_id, show_id) not in seen:
                pairs.append(pair)
                seen.add((query_id, show_id))
    random.Random(seed).shuffle(pairs)
    write_json(output, {
        "version": 1, "kind": "initial", "pass_id": "initial",
        "split_sha256": fingerprint(split_file), "corpus_sha256": fingerprint(corpus),
        "partition": partition, "seed": seed,
        "candidate_method": "default BM25 top-k plus optional human-supplied candidates",
        "additional_pairs_sha256": fingerprint(additional_pairs) if additional_pairs else None,
        "k": k, "pairs": pairs,
    })


def load_pool(
    path: Path, split_file: Path, corpus: Path,
) -> tuple[Pool, list[Pair], dict[str, dict[str, str]]]:
    queries = load_queries(split_file, corpus)
    pool = read_json(path)
    if (pool.get("version") != 1
            or pool.get("split_sha256") != fingerprint(split_file)
            or pool.get("corpus_sha256") != fingerprint(corpus)):
        raise ValueError("Pool does not match the frozen query split and corpus")
    kind = pool.get("kind")
    if not isinstance(kind, str) or kind not in {"initial", "relabel"}:
        raise ValueError("Unknown pool kind")
    pass_id = pool.get("pass_id")
    if not isinstance(pass_id, str) or not pass_id.strip():
        raise ValueError("Pool is missing its pass ID")
    if (kind == "initial") != (pass_id == "initial"):
        raise ValueError("Pool kind and pass ID disagree")
    source_hash = pool.get("source_pool_sha256", "")
    if not isinstance(source_hash, str):
        raise ValueError("Original pool fingerprint must be a string")
    if kind == "relabel" and (len(source_hash) != 64 or source_hash == fingerprint(path)):
        raise ValueError("Re-label pool is missing its original pool fingerprint")
    partition = pool.get("partition")
    if not isinstance(partition, str) or partition not in {"dev", "heldout"}:
        raise ValueError("Pool is missing its query partition")
    rows = pool.get("pairs")
    if not isinstance(rows, list) or not rows:
        raise ValueError("Pool contains no query-record pairs")
    records, _ = load_records(corpus)
    record_ids = {row["show_id"] for row in records}
    pairs: list[Pair] = []
    seen: set[Pair] = set()
    for row in rows:
        if not isinstance(row, dict) or set(row) != {"query_id", "show_id"}:
            raise ValueError("Pool pairs must contain only query_id and show_id, never grades")
        query_id, show_id = row["query_id"], row["show_id"]
        if not isinstance(query_id, str) or not isinstance(show_id, str):
            raise ValueError("Pool identifiers must be strings")
        if query_id not in queries or show_id not in record_ids:
            raise ValueError("Pool refers to an unknown query or record")
        if queries[query_id]["split"] != partition:
            raise ValueError("Pool mixes development and held-out queries")
        pair = (query_id, show_id)
        if pair in seen:
            raise ValueError("Pool contains duplicate query-record pairs")
        seen.add(pair)
        pairs.append(pair)
    return Pool(
        kind=kind, pass_id=pass_id, partition=partition,
        split_sha256=fingerprint(split_file), corpus_sha256=fingerprint(corpus),
        source_pool_sha256=source_hash,
    ), pairs, queries


def load_labels(
    path: Path, split_file: Path, corpus: Path,
) -> dict[LabelKey, dict[str, str]]:
    if not path.exists():
        return {}
    labels: dict[LabelKey, dict[str, str]] = {}
    queries = load_queries(split_file, corpus)
    records, _ = load_records(corpus)
    record_ids = {row["show_id"] for row in records}
    split_hash, corpus_hash = fingerprint(split_file), fingerprint(corpus)
    for row in read_table(path, LABEL_COLUMNS):
        if row["grade"] not in {"0", "1", "2", "3"}:
            raise ValueError("Judgment CSV contains a grade outside 0-3")
        if row["split_sha256"] != split_hash or row["corpus_sha256"] != corpus_hash:
            raise ValueError("Judgment CSV belongs to a different split or corpus")
        if row["query_id"] not in queries or row["show_id"] not in record_ids:
            raise ValueError("Judgment CSV contains an unknown query or record")
        if any(not row[name].strip() for name in LABEL_COLUMNS):
            raise ValueError("Judgment CSV contains an empty required value")
        timestamp = datetime.fromisoformat(row["timestamp"])
        if timestamp.utcoffset() is None:
            raise ValueError("Judgment timestamps must include a timezone")
        if len(row["pool_sha256"]) != 64:
            raise ValueError("Invalid pool fingerprint in judgment CSV")
        key = (row["pass_id"], row["query_id"], row["show_id"])
        if key in labels:
            raise ValueError("Duplicate grade for the same pair and pass")
        labels[key] = row
    return labels


def require_labels(
    labels: dict[LabelKey, dict[str, str]], pairs: list[Pair], pass_id: str, pool_hash: str,
) -> list[int]:
    missing = [pair for pair in pairs if (pass_id, *pair) not in labels]
    if missing:
        raise ValueError(
            f"Missing human labels: {len(missing)} pair(s) in pass {pass_id}. "
            "Complete the labelling pass; no grades will be inferred."
        )
    rows = [labels[(pass_id, *pair)] for pair in pairs]
    if any(row["pool_sha256"] != pool_hash for row in rows):
        raise ValueError("Existing grades refer to a different candidate pool")
    return [int(row["grade"]) for row in rows]


def sample_relabel(
    initial_pool: Path, split_file: Path, corpus: Path, labels_file: Path,
    output: Path, size: int, seed: int,
) -> None:
    pool, pairs, _ = load_pool(initial_pool, split_file, corpus)
    if pool.kind != "initial":
        raise ValueError("Sample from an initial pool, not another re-label pass")
    if not 1 <= size <= len(pairs):
        raise ValueError(f"Sample size must be between 1 and {len(pairs)}")
    labels = load_labels(labels_file, split_file, corpus)
    require_labels(labels, pairs, "initial", fingerprint(initial_pool))
    sample = random.Random(seed).sample(sorted(pairs), size)
    write_json(output, {
        "version": 1, "kind": "relabel", "pass_id": f"relabel-{uuid.uuid4().hex}",
        "split_sha256": fingerprint(split_file), "corpus_sha256": fingerprint(corpus),
        "source_pool_sha256": fingerprint(initial_pool),
        "partition": pool.partition, "seed": seed,
        "pairs": [{"query_id": query_id, "show_id": show_id} for query_id, show_id in sample],
    })


def append_label(path: Path, row: dict[str, str]) -> None:
    new_file = not path.exists()
    with path.open("x" if new_file else "a", encoding="utf-8", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=LABEL_COLUMNS)
        if new_file:
            writer.writeheader()
        writer.writerow(row)
        target.flush()
        os.fsync(target.fileno())


def label_pool(
    pool_file: Path, split_file: Path, corpus: Path, labels_file: Path,
    read: Callable[[str], str] = input, display: Callable[[str], None] = print,
) -> None:
    pool, pairs, queries = load_pool(pool_file, split_file, corpus)
    labels = load_labels(labels_file, split_file, corpus)
    pool_hash = fingerprint(pool_file)
    done = [pair for pair in pairs if (pool.pass_id, *pair) in labels]
    if done:
        require_labels(labels, done, pool.pass_id, pool_hash)
    if pool.kind == "relabel":
        require_labels(labels, pairs, "initial", pool.source_pool_sha256)
    records, _ = load_records(corpus)
    by_id = {row["show_id"]: row for row in records}
    pending = [pair for pair in pairs if (pool.pass_id, *pair) not in labels]
    display(f"{pool.partition} pass: {len(pending)} unlabelled pair(s). Earlier grades and scores are hidden.")
    for query_id, show_id in pending:
        display(f"\nQuery [{query_id}]: {queries[query_id]['query']}")
        for name in FIELDS:
            display(f"{name}: {by_id[show_id][name]}")
        while True:
            try:
                answer = read("Your relevance grade (0-3), or q to save and quit: ").strip()
            except (EOFError, KeyboardInterrupt):
                display("\nStopped. Previously submitted grades are saved; no grade entered for this pair.")
                return
            if answer.lower() == "q":
                display("Stopped. Previously submitted grades are saved.")
                return
            if answer not in {"0", "1", "2", "3"}:
                display("Enter an explicit integer 0, 1, 2, or 3. There is no default grade.")
                continue
            append_label(labels_file, {
                "query_id": query_id, "show_id": show_id, "grade": answer,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "pass_id": pool.pass_id, "split_sha256": pool.split_sha256,
                "corpus_sha256": pool.corpus_sha256, "pool_sha256": pool_hash,
            })
            break
    display("Pass complete.")


def agreement(first: list[int], second: list[int]) -> dict[str, int | float | None | str]:
    if not first or len(first) != len(second):
        raise ValueError("Consistency requires nonempty, equally sized paired human grades")
    if any(type(grade) is not int or grade not in range(4) for grade in first + second):
        raise ValueError("Consistency grades must be integers from 0 to 3")
    count = len(first)
    exact = sum(a == b for a, b in zip(first, second)) / count
    # Quadratic disagreement weights: (a-b)^2 / (3-0)^2.
    observed = sum((a - b) ** 2 / 9 for a, b in zip(first, second)) / count
    a_counts, b_counts = Counter(first), Counter(second)
    expected = sum(
        (a - b) ** 2 / 9 * a_counts[a] * b_counts[b] / count ** 2
        for a in range(4) for b in range(4)
    )
    kappa = None if expected == 0 else 1 - observed / expected
    return {
        "paired_count": count, "exact_agreement": exact,
        "quadratic_weighted_kappa": kappa,
        "kappa_note": "undefined: both passes use the same single grade" if kappa is None else "",
    }


def consistency(
    pool_file: Path, split_file: Path, corpus: Path, labels_file: Path,
) -> dict[str, int | float | None | str]:
    pool, pairs, _ = load_pool(pool_file, split_file, corpus)
    if pool.kind != "relabel":
        raise ValueError("Consistency requires a re-label pool")
    labels = load_labels(labels_file, split_file, corpus)
    first = require_labels(labels, pairs, "initial", pool.source_pool_sha256)
    second = require_labels(labels, pairs, pool.pass_id, fingerprint(pool_file))
    return agreement(first, second) | {"partition": pool.partition}
