"""Connect human judgments to reproducible runs without guessing grades or failure causes."""

import math
import random
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

from film_data import FilmRecord, load_records
from judgments import load_labels, load_pool, require_labels
from metrics import mean_reciprocal_rank, ndcg_at_k, precision_at_k, recall_at_k, reciprocal_rank
from query_sets import fingerprint, load_queries, read_json, read_table, write_json
from retrieval import IndexConfig, build_index


@dataclass(frozen=True)
class Experiment:
    name: str
    config: IndexConfig


@dataclass
class QueryResult:
    query_id: str
    query: str
    metrics: dict[str, float]
    top: list[dict[str, str | int | float]]
    missed: list[dict[str, str | int]]


@dataclass
class ExperimentResult:
    name: str
    config: IndexConfig
    aggregate: dict[str, float]
    queries: list[QueryResult]


@dataclass
class Evaluation:
    version: int
    partition: str
    split_sha256: str
    corpus_sha256: str
    pool_sha256: str
    labels_sha256: str
    k: int
    relevant_grade: int
    failure_threshold: float
    experiments: list[ExperimentResult]


@dataclass(frozen=True)
class FrozenSettings:
    experiment: Experiment
    k: int
    relevant_grade: int
    failure_threshold: float


def parse_config(value: object) -> IndexConfig:
    if not isinstance(value, dict):
        raise ValueError("An index configuration must be a JSON object")
    allowed = set(IndexConfig.__dataclass_fields__)
    if set(value) - allowed:
        raise ValueError("Unknown index configuration option")
    fields = value.get("fields", list(IndexConfig().fields))
    weights = value.get("field_weights", {})
    if not isinstance(fields, list) or not all(isinstance(name, str) for name in fields):
        raise ValueError("Configuration fields must be a JSON list of strings")
    if not isinstance(weights, dict) or not all(
        isinstance(name, str) and type(weight) is int for name, weight in weights.items()
    ):
        raise ValueError("Field weights must map field names to integers")
    lowercase, stemming = value.get("lowercase", True), value.get("stemming", False)
    if not isinstance(lowercase, bool) or not isinstance(stemming, bool):
        raise ValueError("lowercase and stemming must be booleans")
    numbers: dict[str, float] = {}
    for key, default in (("k1", 1.5), ("b", 0.75), ("epsilon", 0.25)):
        number = value.get(key, default)
        if not isinstance(number, (int, float)) or isinstance(number, bool):
            raise ValueError(f"{key} must be numeric")
        numbers[key] = float(number)
    tokenizer = value.get("tokenizer", "whitespace")
    if not isinstance(tokenizer, str):
        raise ValueError("tokenizer must be a string")
    return IndexConfig(
        fields=tuple(fields), field_weights=weights, tokenizer=tokenizer,
        lowercase=lowercase, stemming=stemming,
        k1=numbers["k1"], b=numbers["b"], epsilon=numbers["epsilon"],
    )


def load_experiments(path: Path) -> list[Experiment]:
    value = read_json(path)
    rows = value.get("experiments")
    if not isinstance(rows, list) or not rows:
        raise ValueError("Supply a nonempty experiments list; its first entry is the baseline")
    experiments: list[Experiment] = []
    for row in rows:
        if not isinstance(row, dict) or set(row) != {"name", "config"}:
            raise ValueError("Each experiment requires exactly name and config")
        name = row["name"]
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Experiment names must be nonempty strings")
        if name in {experiment.name for experiment in experiments}:
            raise ValueError("Experiment names must be unique")
        experiments.append(Experiment(name, parse_config(row["config"])))
    return experiments


def load_frozen(path: Path, split_file: Path, corpus: Path) -> FrozenSettings:
    value = read_json(path)
    if (value.get("version") != 1 or value.get("partition_selected_on") != "dev"
            or value.get("split_sha256") != fingerprint(split_file)
            or value.get("corpus_sha256") != fingerprint(corpus)):
        raise ValueError("Frozen configuration must have been selected on this development split/corpus")
    name = value.get("name")
    if not isinstance(name, str) or not name.strip():
        raise ValueError("Frozen configuration is missing its experiment name")
    k, grade, threshold = value.get("k"), value.get("relevant_grade"), value.get("failure_threshold")
    if not isinstance(k, int) or isinstance(k, bool) or k < 1:
        raise ValueError("Invalid frozen cutoff")
    if not isinstance(grade, int) or isinstance(grade, bool) or grade not in (1, 2, 3):
        raise ValueError("Invalid frozen relevance threshold")
    if (not isinstance(threshold, (int, float)) or isinstance(threshold, bool)
            or not math.isfinite(threshold) or not 0 <= threshold <= 1):
        raise ValueError("Invalid frozen failure threshold")
    report_hash = value.get("dev_report_sha256")
    if not isinstance(report_hash, str) or len(report_hash) != 64:
        raise ValueError("Frozen settings require the source development report fingerprint")
    return FrozenSettings(Experiment(name, parse_config(value.get("config"))), k, grade, float(threshold))


def require_frozen(
    frozen_file: Path | None, split_file: Path, corpus: Path, experiments: list[Experiment],
    k: int, relevant_grade: int | None = None, failure_threshold: float | None = None,
) -> None:
    if frozen_file is None:
        raise ValueError("Held-out work requires a configuration frozen from a development report")
    settings = load_frozen(frozen_file, split_file, corpus)
    if experiments != [settings.experiment] or k != settings.k:
        raise ValueError("Held-out settings and cutoff must match the frozen configuration")
    if ((relevant_grade is not None and relevant_grade != settings.relevant_grade)
            or (failure_threshold is not None and failure_threshold != settings.failure_threshold)):
        raise ValueError("Held-out metric thresholds must match the frozen development protocol")


def prepare_pool(
    split_file: Path, corpus: Path, experiments: list[Experiment], partition: str,
    output: Path, k: int, additional_pairs: Path | None = None, frozen_file: Path | None = None,
) -> None:
    queries = load_queries(split_file, corpus)
    if partition not in {"dev", "heldout"} or not experiments:
        raise ValueError("Choose a partition and at least one experiment")
    if partition == "heldout":
        require_frozen(frozen_file, split_file, corpus, experiments, k)
    records, _ = load_records(corpus)
    if type(k) is not int or not 1 <= k <= len(records):
        raise ValueError("Pool depth must be a positive integer no larger than the corpus")
    pairs: set[tuple[str, str]] = set()
    for experiment in experiments:
        index = build_index(records, experiment.config)
        pairs.update(
            (query_id, result.record["show_id"])
            for query_id, query in queries.items() if query["split"] == partition
            for result in index.search(query["query"], k)
        )
    if additional_pairs is not None:
        record_ids = {row["show_id"] for row in records}
        for pair in read_table(additional_pairs, ("query_id", "show_id")):
            query_id, show_id = pair["query_id"], pair["show_id"]
            if query_id not in queries or queries[query_id]["split"] != partition or show_id not in record_ids:
                raise ValueError("Additional pairs must use known queries/records in the selected partition")
            pairs.add((query_id, show_id))
    shuffled = sorted(pairs)
    random.Random(42).shuffle(shuffled)
    write_json(output, {
        "version": 1, "kind": "initial", "pass_id": "initial", "partition": partition,
        "split_sha256": fingerprint(split_file), "corpus_sha256": fingerprint(corpus),
        "candidate_method": "union of predeclared runs and optional human candidates",
        "experiments": [asdict(experiment) for experiment in experiments], "k": k, "seed": 42,
        "pairs": [{"query_id": query_id, "show_id": show_id} for query_id, show_id in shuffled],
    })


def initial_judgments(
    pool_file: Path, split_file: Path, corpus: Path, labels_file: Path, partition: str,
) -> tuple[dict[str, dict[str, str]], dict[str, dict[str, int]]]:
    pool, pairs, queries = load_pool(pool_file, split_file, corpus)
    if pool.kind != "initial" or pool.partition != partition:
        raise ValueError("Evaluation requires an initial pool from the selected partition")
    labels = load_labels(labels_file, split_file, corpus)
    grades = require_labels(labels, pairs, "initial", fingerprint(pool_file))
    judgments: dict[str, dict[str, int]] = defaultdict(dict)
    for (query_id, show_id), grade in zip(pairs, grades):
        judgments[query_id][show_id] = grade
    expected = {key for key, query in queries.items() if query["split"] == partition}
    if set(judgments) != expected:
        raise ValueError("The judged pool must cover every query in the selected partition")
    return queries, dict(judgments)


def evaluate_run(
    experiment: Experiment, records: list[FilmRecord], queries: dict[str, dict[str, str]],
    judgments: dict[str, dict[str, int]], k: int, relevant_grade: int,
) -> ExperimentResult:
    index = build_index(records, experiment.config)
    results: list[QueryResult] = []
    rankings: dict[str, list[str]] = {}
    for query_id, grades in judgments.items():
        retrieved = index.search(queries[query_id]["query"], k)
        ranking = [result.record["show_id"] for result in retrieved]
        rankings[query_id] = ranking
        try:
            values = {
                "precision_at_k": precision_at_k(ranking, grades, k, relevant_grade=relevant_grade),
                "recall_at_k": recall_at_k(ranking, grades, k, relevant_grade=relevant_grade),
                "reciprocal_rank_at_k": reciprocal_rank(ranking, grades, relevant_grade=relevant_grade),
                "ndcg_at_k": ndcg_at_k(ranking, grades, k),
            }
        except ValueError as error:
            raise ValueError(f"{experiment.name}, query {query_id}: {error}") from error
        results.append(QueryResult(
            query_id, queries[query_id]["query"], values,
            [{"show_id": result.record["show_id"], "score": result.score,
              "grade": grades[result.record["show_id"]]} for result in retrieved],
            [{"show_id": show_id, "grade": grades[show_id]}
             for show_id in sorted(grades) if grades[show_id] >= relevant_grade and show_id not in ranking],
        ))
    aggregate = {
        name: sum(result.metrics[name] for result in results) / len(results)
        for name in ("precision_at_k", "recall_at_k", "ndcg_at_k")
    }
    aggregate["mrr_at_k"] = mean_reciprocal_rank(rankings, judgments, relevant_grade=relevant_grade)
    return ExperimentResult(experiment.name, experiment.config, aggregate, results)


def evaluate(
    pool_file: Path, split_file: Path, corpus: Path, labels_file: Path,
    experiments: list[Experiment], partition: str, k: int, relevant_grade: int,
    failure_threshold: float, frozen_file: Path | None = None,
) -> Evaluation:
    if (type(failure_threshold) not in (int, float)
            or not math.isfinite(failure_threshold) or not 0 <= failure_threshold <= 1):
        raise ValueError("The nDCG failure threshold must be between zero and one")
    if not experiments or len({experiment.name for experiment in experiments}) != len(experiments):
        raise ValueError("Supply uniquely named experiments")
    if partition == "heldout":
        require_frozen(frozen_file, split_file, corpus, experiments, k, relevant_grade, failure_threshold)
    queries, judgments = initial_judgments(pool_file, split_file, corpus, labels_file, partition)
    records, _ = load_records(corpus)
    if type(k) is not int or not 1 <= k <= len(records):
        raise ValueError("Evaluation k must be a positive integer no larger than the corpus")
    results = [
        evaluate_run(experiment, records, queries, judgments, k, relevant_grade)
        for experiment in experiments
    ]
    return Evaluation(
        1, partition, fingerprint(split_file), fingerprint(corpus), fingerprint(pool_file),
        fingerprint(labels_file), k, relevant_grade, failure_threshold, results,
    )


def save_evaluation(path: Path, result: Evaluation) -> None:
    write_json(path, asdict(result))


def load_evaluation(path: Path) -> Evaluation:
    """Validate saved numerical outputs before generating summaries or freezing settings."""
    data = read_json(path)
    required = set(Evaluation.__dataclass_fields__)
    if set(data) != required or data.get("version") != 1:
        raise ValueError("Invalid evaluation report schema/version")
    if data["partition"] not in ("dev", "heldout"):
        raise ValueError("Invalid evaluation partition")
    for name in ("split_sha256", "corpus_sha256", "pool_sha256", "labels_sha256"):
        if not isinstance(data[name], str) or len(data[name]) != 64:
            raise ValueError("Invalid evaluation fingerprint")
    if type(data["k"]) is not int or data["k"] < 1:
        raise ValueError("Invalid evaluation cutoff")
    if type(data["relevant_grade"]) is not int or data["relevant_grade"] not in (1, 2, 3):
        raise ValueError("Invalid evaluation relevance threshold")
    threshold = data["failure_threshold"]
    if type(threshold) not in (int, float) or not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("Invalid evaluation failure threshold")
    rows = data["experiments"]
    if not isinstance(rows, list) or not rows:
        raise ValueError("Evaluation contains no experiments")
    results: list[ExperimentResult] = []
    expected_queries: set[str] | None = None
    for row in rows:
        if not isinstance(row, dict) or set(row) != {"name", "config", "aggregate", "queries"}:
            raise ValueError("Invalid experiment result schema")
        name = row["name"]
        if not isinstance(name, str) or not name.strip() or name in {item.name for item in results}:
            raise ValueError("Invalid or duplicate experiment name")
        aggregate = validate_metrics(row["aggregate"], {"precision_at_k", "recall_at_k", "ndcg_at_k", "mrr_at_k"})
        query_rows = row["queries"]
        if not isinstance(query_rows, list) or not query_rows:
            raise ValueError("Experiment has no query results")
        queries: list[QueryResult] = []
        for query in query_rows:
            if not isinstance(query, dict) or set(query) != {"query_id", "query", "metrics", "top", "missed"}:
                raise ValueError("Invalid query result schema")
            if not all(isinstance(query[key], str) and query[key].strip() for key in ("query_id", "query")):
                raise ValueError("Invalid query result identifiers/text")
            metrics = validate_metrics(query["metrics"], {
                "precision_at_k", "recall_at_k", "ndcg_at_k", "reciprocal_rank_at_k",
            })
            if not isinstance(query["top"], list) or not isinstance(query["missed"], list):
                raise ValueError("Invalid query result records")
            for item in query["missed"]:
                if (not isinstance(item, dict) or set(item) != {"show_id", "grade"}
                        or not isinstance(item["show_id"], str)
                        or type(item["grade"]) is not int or not data["relevant_grade"] <= item["grade"] <= 3):
                    raise ValueError("Invalid missed record or human grade")
            for hit in query["top"]:
                if (not isinstance(hit, dict) or set(hit) != {"show_id", "score", "grade"}
                        or not isinstance(hit["show_id"], str)
                        or type(hit["score"]) not in (int, float) or not math.isfinite(hit["score"])
                        or type(hit["grade"]) is not int or hit["grade"] not in range(4)):
                    raise ValueError("Invalid retrieved record")
            top_ids = [hit["show_id"] for hit in query["top"]]
            missed_ids = [hit["show_id"] for hit in query["missed"]]
            if (len(top_ids) > data["k"] or len(set(top_ids)) != len(top_ids)
                    or len(set(missed_ids)) != len(missed_ids) or set(top_ids) & set(missed_ids)):
                raise ValueError("Duplicate or inconsistent ranked/missed records")
            queries.append(QueryResult(query["query_id"], query["query"], metrics, query["top"], query["missed"]))
        ids = {query.query_id for query in queries}
        if len(ids) != len(queries) or (expected_queries is not None and ids != expected_queries):
            raise ValueError("Experiments must cover the same unique queries")
        expected_queries = ids
        for metric, reported in aggregate.items():
            query_metric = "reciprocal_rank_at_k" if metric == "mrr_at_k" else metric
            actual = sum(query.metrics[query_metric] for query in queries) / len(queries)
            if not math.isclose(reported, actual, rel_tol=1e-12, abs_tol=1e-12):
                raise ValueError("Report aggregate does not equal its query-level mean")
        results.append(ExperimentResult(name, parse_config(row["config"]), aggregate, queries))
    if data["partition"] == "heldout" and len(results) != 1:
        raise ValueError("Held-out reports cannot contain ablation comparisons")
    return Evaluation(
        version=1, partition=data["partition"], split_sha256=data["split_sha256"],
        corpus_sha256=data["corpus_sha256"], pool_sha256=data["pool_sha256"],
        labels_sha256=data["labels_sha256"], k=data["k"], relevant_grade=data["relevant_grade"],
        failure_threshold=float(threshold), experiments=results,
    )


def validate_metrics(value: object, keys: set[str]) -> dict[str, float]:
    if not isinstance(value, dict) or set(value) != keys:
        raise ValueError("Invalid metric names in report")
    if any(type(number) not in (int, float) or not math.isfinite(number) or not 0 <= number <= 1
           for number in value.values()):
        raise ValueError("Report metrics must be finite values between zero and one")
    return {key: float(number) for key, number in value.items()}


def freeze_configuration(report_file: Path, name: str, output: Path) -> None:
    report = load_evaluation(report_file)
    if report.partition != "dev":
        raise ValueError("Settings must be selected from a development report, never heldout")
    selected = next((item for item in report.experiments if item.name == name), None)
    if selected is None:
        raise ValueError("Selected experiment is not present in the development report")
    write_json(output, {
        "version": 1, "partition_selected_on": "dev", "name": selected.name,
        "config": asdict(selected.config), "dev_report_sha256": fingerprint(report_file),
        "split_sha256": report.split_sha256, "corpus_sha256": report.corpus_sha256,
        "k": report.k, "relevant_grade": report.relevant_grade,
        "failure_threshold": report.failure_threshold,
    })
