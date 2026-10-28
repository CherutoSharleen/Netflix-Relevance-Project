"""Share public search resources while keeping uploaded research files session-local."""

import io
import json
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path

import streamlit as st

from analysis import (
    Evaluation, Experiment, evaluate, freeze_configuration, load_evaluation, load_experiments,
    load_frozen, parse_config, prepare_pool, save_evaluation,
)
from failure_review import review_files, summary_files
from film_data import DATA_PATH, CleaningReport, FilmRecord, load_records
from query_sets import fingerprint, load_queries
from retrieval import FilmIndex, build_index


ROOT = Path(__file__).resolve().parent


@st.cache_data(ttl=3600, max_entries=1)
def public_records(corpus_sha256: str) -> tuple[list[FilmRecord], CleaningReport]:
    return load_records(DATA_PATH)


@st.cache_resource(ttl=3600, max_entries=2)
def public_index(config_json: str, corpus_sha256: str) -> FilmIndex:
    config = parse_config(json.loads(config_json))
    hosted_limits([Experiment("search", config)])
    records, _ = public_records(corpus_sha256)
    return build_index(records, config)


def hosted_limits(experiments: list[Experiment]) -> None:
    if len(experiments) > 8:
        raise ValueError("The hosted demo accepts at most 8 configurations per operation; use the local CLI for larger runs.")
    if any(weight > 5 for experiment in experiments for weight in experiment.config.field_weights.values()):
        raise ValueError("Hosted field weights are capped at 5 to limit memory use; larger weights remain available in the CLI.")


@dataclass
class WorkflowOutput:
    files: dict[str, bytes]
    report: Evaluation | None = None


def archive(files: dict[str, bytes]) -> bytes:
    target = io.BytesIO()
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        for name, content in files.items():
            bundle.writestr(name, content)
    return target.getvalue()


def uploaded_workflow(
    action: str, uploads: dict[str, bytes], partition: str = "dev",
    k: int = 10, relevant_grade: int | None = None,
    failure_threshold: float | None = None, selected: str = "",
) -> WorkflowOutput:
    """Use fixed temporary filenames, never upload-provided paths or a global cache."""
    allowed = {"split", "pool", "labels", "experiments", "frozen", "extra", "report", "assignments"}
    if set(uploads) - allowed:
        raise ValueError("Unknown upload type")
    required = {
        "Prepare candidate pool": {"split"},
        "Evaluate judgments": {"split", "pool", "labels"},
        "Summarize assigned failures": {"report", "assignments"},
        "Freeze final settings": {"report"},
    }
    if action not in required or not required[action] <= set(uploads):
        raise ValueError("Missing required uploaded research files")
    with tempfile.TemporaryDirectory(prefix="netflix-research-") as directory:
        root = Path(directory)
        paths = {name: root / name for name in uploads}
        for name, content in uploads.items():
            paths[name].write_bytes(content)
        if action in {"Summarize assigned failures", "Freeze final settings"}:
            report = load_evaluation(paths["report"])
            if report.corpus_sha256 != fingerprint(DATA_PATH):
                raise ValueError("Report does not match the deployed corpus")
            if action == "Freeze final settings":
                output = root / "frozen.json"
                freeze_configuration(paths["report"], selected, output)
                return WorkflowOutput({"frozen.json": output.read_bytes()})
            files = summary_files(report, paths["assignments"])
            return WorkflowOutput({name: text.encode("utf-8") for name, text in files.items()}, report)
        if partition == "heldout":
            if "frozen" not in paths or "experiments" in paths:
                raise ValueError("Held-out work requires frozen settings, never an ablation list")
            settings = load_frozen(paths["frozen"], paths["split"], DATA_PATH)
            experiments = [settings.experiment]
            k, relevant_grade, failure_threshold = settings.k, settings.relevant_grade, settings.failure_threshold
        else:
            if partition != "dev" or "experiments" not in paths:
                raise ValueError("Development work requires an experiments JSON upload")
            experiments = load_experiments(paths["experiments"])
        hosted_limits(experiments)
        if not 1 <= k <= 100:
            raise ValueError("The hosted demo supports depth 1-100; use the CLI for deeper runs.")
        if len(load_queries(paths["split"], DATA_PATH)) > 100:
            raise ValueError("The hosted demo accepts at most 100 queries; use the CLI for larger studies.")
        if action == "Prepare candidate pool":
            output = root / "candidate_pool.json"
            prepare_pool(paths["split"], DATA_PATH, experiments, partition, output,
                         k, paths.get("extra"), paths.get("frozen"))
            return WorkflowOutput({"candidate_pool.json": output.read_bytes()})
        if relevant_grade is None or failure_threshold is None:
            raise ValueError("Choose the binary relevance threshold and failure threshold explicitly")
        report = evaluate(
            paths["pool"], paths["split"], DATA_PATH, paths["labels"], experiments,
            partition, k, relevant_grade, failure_threshold, paths.get("frozen"),
        )
        output = root / "evaluation.json"
        save_evaluation(output, report)
        records, _ = load_records(DATA_PATH)
        files = {name: text.encode("utf-8") for name, text in review_files(report, records).items()}
        files["evaluation.json"] = output.read_bytes()
        return WorkflowOutput(files, report)
