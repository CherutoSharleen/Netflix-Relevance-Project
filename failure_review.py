"""Export literal matching evidence and count only failure causes assigned by a human."""

import csv
import io
import json
from collections import Counter
from dataclasses import replace
from html import escape
from pathlib import Path

from analysis import Evaluation
from film_data import FIELDS, FilmRecord
from query_sets import read_table
from retrieval import IndexConfig, tokenize


CAUSES = {
    "tokenisation": "Word splitting, punctuation, case, or stemming prevents useful matches.",
    "missing_or_poor_field_data": "Record fields are absent, incorrect, or too sparse.",
    "field_coverage": "Useful matching text exists in a field excluded from the index.",
    "vocabulary_mismatch": "Query and record express the same intent with different words.",
    "field_weighting": "Relative emphasis on fields is inappropriate for this query.",
    "ambiguous_judgment": "Query intent or the relevance decision is uncertain.",
    "true_ranking_failure": "Adequate data and representation exist, but scoring orders results poorly.",
    "other": "A human-identified cause not covered above; explain it in notes.",
}
ASSIGNMENT_COLUMNS = ("experiment", "query_id", "cause", "notes")
REVIEW_COLUMNS = (
    "experiment", "query_id", "query", "ndcg_at_k", "k", "role", "rank", "show_id",
    "title", "score", "human_grade", "field_matches", "excluded_matching_fields",
    "word_token_matches", "stemmed_matches", "missing_fields", "record_fields",
)


def csv_text(columns: tuple[str, ...], rows: list[dict]) -> str:
    target = io.StringIO(newline="")
    writer = csv.DictWriter(target, fieldnames=columns)
    writer.writeheader()
    writer.writerows(rows)
    return target.getvalue()


def matching_fields(query: str, record: FilmRecord, config: IndexConfig) -> dict[str, list[str]]:
    query_tokens = set(tokenize(query, config))
    matches: dict[str, list[str]] = {}
    for name in FIELDS:
        overlap = query_tokens & set(tokenize(record[name], config))
        if overlap:
            matches[name] = sorted(overlap)
    return matches


def failures(report: Evaluation) -> list[tuple[str, str]]:
    return [
        (experiment.name, query.query_id)
        for experiment in report.experiments for query in experiment.queries
        if query.metrics["ndcg_at_k"] < report.failure_threshold
    ]


def review_files(report: Evaluation, records: list[FilmRecord]) -> dict[str, str]:
    by_id = {record["show_id"]: record for record in records}
    rows: list[dict] = []
    for experiment in report.experiments:
        for query in experiment.queries:
            if query.metrics["ndcg_at_k"] >= report.failure_threshold:
                continue
            hits = [
                (str(hit["show_id"]), "top_result", rank, hit["score"], hit["grade"])
                for rank, hit in enumerate(query.top, 1)
            ] + [(str(hit["show_id"]), "relevant_missed", "", "", hit["grade"]) for hit in query.missed]
            for show_id, role, rank, score, grade in hits:
                if show_id not in by_id:
                    raise ValueError(f"Unknown record in review report: {show_id}")
                record = by_id[show_id]
                matches = matching_fields(query.query, record, experiment.config)
                word_config = replace(experiment.config, tokenizer="words", stemming=False)
                stem_config = replace(word_config, stemming=True)
                rows.append({
                    "experiment": experiment.name, "query_id": query.query_id, "query": query.query,
                    "ndcg_at_k": query.metrics["ndcg_at_k"], "k": report.k, "role": role,
                    "rank": rank, "show_id": show_id, "title": record["title"], "score": score,
                    "human_grade": grade, "field_matches": json.dumps(matches),
                    "excluded_matching_fields": json.dumps(sorted(set(matches) - set(experiment.config.fields))),
                    "word_token_matches": json.dumps(matching_fields(query.query, record, word_config)),
                    "stemmed_matches": json.dumps(matching_fields(query.query, record, stem_config)),
                    "missing_fields": json.dumps([name for name in FIELDS if not record[name]]),
                    "record_fields": json.dumps(record),
                })
    assignments = [
        {"experiment": name, "query_id": query_id, "cause": "", "notes": ""}
        for name, query_id in failures(report)
    ]
    return {
        "review.csv": csv_text(REVIEW_COLUMNS, rows),
        "assignments.csv": csv_text(ASSIGNMENT_COLUMNS, assignments),
        "taxonomy.json": json.dumps(CAUSES, indent=2) + "\n",
    }


def ablation_deltas(report: Evaluation) -> list[dict[str, str | float]]:
    baseline = report.experiments[0]
    return [
        {"baseline": baseline.name, "experiment": experiment.name, "metric": metric,
         "baseline_value": baseline.aggregate[metric], "value": value,
         "delta": value - baseline.aggregate[metric]}
        for experiment in report.experiments[1:] for metric, value in experiment.aggregate.items()
    ]


def cause_counts(report: Evaluation, assignments_file: Path) -> list[dict[str, str | int]]:
    expected = set(failures(report))
    assigned: set[tuple[str, str]] = set()
    counts: Counter[tuple[str, str]] = Counter()
    for row in read_table(assignments_file, ASSIGNMENT_COLUMNS):
        pair = (row["experiment"], row["query_id"])
        if pair not in expected or pair in assigned:
            raise ValueError("Assignments must cover each failing query/experiment exactly once")
        if not row["cause"]:
            raise ValueError("Missing human failure assignments. Fill every cause yourself before summarizing.")
        if row["cause"] not in CAUSES:
            raise ValueError(f"Unknown failure cause: {row['cause']}")
        if row["cause"] == "other" and not row["notes"].strip():
            raise ValueError("Explain an 'other' cause in notes")
        assigned.add(pair)
        counts[(pair[0], row["cause"])] += 1
    if assigned != expected:
        raise ValueError("Missing human failure assignments for some failing query/experiment pairs")
    return [
        {"experiment": name, "cause": cause, "count": count}
        for (name, cause), count in sorted(counts.items())
    ]


def cause_chart(counts: list[dict[str, str | int]], baseline: str) -> str:
    totals: Counter[str] = Counter()
    for row in counts:
        if row["experiment"] == baseline:
            totals[str(row["cause"])] += int(row["count"])
    if not totals:
        return ('<svg xmlns="http://www.w3.org/2000/svg" width="650" height="80">'
                '<text x="20" y="40">No baseline queries fell below the chosen threshold.</text></svg>')
    maximum = max(totals.values())
    height = 80 + 40 * len(totals)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="850" height="{height}" viewBox="0 0 850 {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="20" y="28" fill="#222" font-family="sans-serif">Human-assigned baseline causes: {escape(baseline)}</text>',
    ]
    for index, (cause, count) in enumerate(sorted(totals.items())):
        y = 55 + index * 40
        parts.extend([
            f'<text x="20" y="{y + 18}" fill="#222" font-family="sans-serif">{escape(cause)}</text>',
            f'<rect x="310" y="{y}" width="{440 * count / maximum:.2f}" height="25" fill="#c33"/>',
            f'<text x="{320 + 440 * count / maximum:.2f}" y="{y + 18}" fill="#222">{count}</text>',
        ])
    return "".join(parts) + "</svg>"


def summary_files(report: Evaluation, assignments_file: Path) -> dict[str, str]:
    counts = cause_counts(report, assignments_file)
    return {
        "failure_counts.csv": csv_text(("experiment", "cause", "count"), counts),
        "ablation_deltas.csv": csv_text(
            ("baseline", "experiment", "metric", "baseline_value", "value", "delta"),
            ablation_deltas(report),
        ),
        "failure_counts.svg": cause_chart(counts, report.experiments[0].name),
    }


def write_bundle(output: Path, files: dict[str, str]) -> None:
    output.mkdir(parents=False, exist_ok=False)
    for name, content in files.items():
        with (output / name).open("x", encoding="utf-8", newline="") as target:
            target.write(content)
