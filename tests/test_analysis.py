"""Verify analysis protocols on artificial records without supplying research judgments."""

import csv
import io
import json
from dataclasses import asdict, replace
from pathlib import Path

import pytest

from analysis import (
    Experiment, evaluate, freeze_configuration, load_evaluation, load_experiments,
    load_frozen, parse_config, prepare_pool, save_evaluation,
)
from analysis_cli import main
from failure_review import (
    ASSIGNMENT_COLUMNS, ablation_deltas, cause_chart, cause_counts, failures,
    matching_fields, review_files, summary_files, write_bundle,
)
from film_data import FIELDS, load_records
from judgments import append_label, load_pool
from query_sets import fingerprint, freeze_queries
from retrieval import IndexConfig


@pytest.fixture
def study(tmp_path: Path) -> dict[str, Path]:
    paths = {name: tmp_path / name for name in (
        "corpus.csv", "queries.csv", "split.json", "pool.json", "labels.csv",
        "experiments.json", "extra.csv", "report.json", "frozen.json",
    )}
    with paths["corpus.csv"].open("w", encoding="utf-8", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=FIELDS)
        writer.writeheader()
        for index, title in enumerate(("Orbit,", "Cooking", "Animals", "Ocean", "Archive"), 1):
            writer.writerow(dict.fromkeys(FIELDS, "") | {
                "show_id": f"s{index}", "title": title,
                "description": "A story.", "country": "orbit cooking animals ocean",
            })
    paths["queries.csv"].write_text(
        "query_id,query\nq1,orbit\nq2,cooking\nq3,animals\nq4,ocean\n", encoding="utf-8",
    )
    freeze_queries(paths["queries.csv"], paths["split.json"], paths["corpus.csv"], 0.25)
    paths["experiments.json"].write_text(json.dumps({"experiments": [
        {"name": "baseline", "config": {"fields": ["title", "description"]}},
        {"name": "words", "config": {"fields": ["title", "description"], "tokenizer": "words"}},
    ]}), encoding="utf-8")
    paths["extra.csv"].write_text(
        "query_id,show_id\n" + "".join(
            f"{query_id},s{number}\n" for query_id in ("q1", "q2", "q4") for number in range(1, 6)
        ), encoding="utf-8",
    )
    prepare_pool(paths["split.json"], paths["corpus.csv"], load_experiments(paths["experiments.json"]),
                 "dev", paths["pool.json"], 2, paths["extra.csv"])
    add_fixture_grades(paths, paths["pool.json"])
    return paths


def add_fixture_grades(paths: dict[str, Path], pool_file: Path) -> None:
    # These are synthetic fixture grades, not relevance judgments for Netflix.
    pool, pairs, _ = load_pool(pool_file, paths["split.json"], paths["corpus.csv"])
    for query_id, show_id in pairs:
        grade = "3" if show_id[1:] == query_id[1:] else ("2" if show_id == "s5" else "0")
        append_label(paths["labels.csv"], {
            "query_id": query_id, "show_id": show_id, "grade": grade,
            "timestamp": "2026-01-01T00:00:00+00:00", "pass_id": "initial",
            "split_sha256": pool.split_sha256, "corpus_sha256": pool.corpus_sha256,
            "pool_sha256": fingerprint(pool_file),
        })


def run(study: dict[str, Path]):
    return evaluate(
        study["pool.json"], study["split.json"], study["corpus.csv"], study["labels.csv"],
        load_experiments(study["experiments.json"]), "dev", 2, 1, 1.0,
    )


def assignments(path: Path, pairs: list[tuple[str, str]], cause: str = "field_coverage") -> None:
    with path.open("w", encoding="utf-8", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=ASSIGNMENT_COLUMNS)
        writer.writeheader()
        for name, query_id in pairs:
            writer.writerow({"experiment": name, "query_id": query_id, "cause": cause, "notes": ""})


def test_evaluate_round_trip_and_review_evidence(study: dict[str, Path]) -> None:
    report = run(study)
    save_evaluation(study["report.json"], report)
    assert asdict(load_evaluation(study["report.json"])) == asdict(report)
    assert report.partition == "dev" and report.k == 2
    assert all(len(experiment.queries) == 3 for experiment in report.experiments)
    records, _ = load_records(study["corpus.csv"])
    files = review_files(report, records)
    reviews = list(csv.DictReader(io.StringIO(files["review.csv"])))
    assert any(row["role"] == "relevant_missed" and row["human_grade"] == "2" for row in reviews)
    assert any("country" in row["excluded_matching_fields"] for row in reviews)
    assert all(not row["cause"] for row in csv.DictReader(io.StringIO(files["assignments.csv"])))
    assert "true_ranking_failure" in json.loads(files["taxonomy.json"])
    assert len(ablation_deltas(report)) == 4


def test_literal_matching_is_not_a_cause_assignment(study: dict[str, Path]) -> None:
    records, _ = load_records(study["corpus.csv"])
    assert "title" not in matching_fields("orbit", records[0], IndexConfig())
    assert matching_fields("orbit", records[0], IndexConfig(tokenizer="words"))["title"] == ["orbit"]


def test_missing_labels_stop_without_a_report(study: dict[str, Path]) -> None:
    study["labels.csv"].unlink()
    with pytest.raises(ValueError, match="Missing human labels"):
        run(study)
    assert not study["report.json"].exists()


def test_unjudged_new_results_are_never_assumed_irrelevant(study: dict[str, Path]) -> None:
    # Prepare a narrower pool and grade it, then request deeper evaluation.
    study["pool.json"].unlink()
    study["labels.csv"].unlink()
    experiments = load_experiments(study["experiments.json"])
    prepare_pool(study["split.json"], study["corpus.csv"], experiments, "dev", study["pool.json"], 1)
    add_fixture_grades(study, study["pool.json"])
    with pytest.raises(ValueError, match="Missing human labels"):
        run(study)


def test_changed_input_fingerprints_and_relabel_pools_are_rejected(study: dict[str, Path]) -> None:
    pool = json.loads(study["pool.json"].read_text())
    pool["kind"], pool["pass_id"] = "relabel", "relabel-fixture"
    pool["source_pool_sha256"] = "a" * 64
    study["pool.json"].write_text(json.dumps(pool), encoding="utf-8")
    with pytest.raises(ValueError, match="initial pool"):
        run(study)


def test_blank_causes_block_summary_and_user_causes_are_counted(
    study: dict[str, Path], tmp_path: Path,
) -> None:
    report = run(study)
    path = tmp_path / "assignments.csv"
    assignments(path, failures(report), "")
    with pytest.raises(ValueError, match="Missing human failure assignments"):
        summary_files(report, path)
    assignments(path, failures(report))
    counts = cause_counts(report, path)
    assert sum(row["count"] for row in counts) == len(failures(report))
    bundle = summary_files(report, path)
    assert set(bundle) == {"failure_counts.csv", "ablation_deltas.csv", "failure_counts.svg"}
    assert "<svg" in bundle["failure_counts.svg"]
    write_bundle(tmp_path / "summary", bundle)
    with pytest.raises(FileExistsError):
        write_bundle(tmp_path / "summary", bundle)


def test_assignments_must_match_exact_failure_pairs(study: dict[str, Path], tmp_path: Path) -> None:
    report = run(study)
    path = tmp_path / "assignments.csv"
    pairs = failures(report)
    for invalid in (pairs[:-1], pairs + [pairs[0]], pairs + [("unknown", "unknown")]):
        assignments(path, invalid)
        with pytest.raises(ValueError):
            cause_counts(report, path)
    assignments(path, pairs, "invented_category")
    with pytest.raises(ValueError, match="Unknown failure cause"):
        cause_counts(report, path)


def test_chart_does_not_pool_correlated_ablation_failures() -> None:
    chart = cause_chart([
        {"experiment": "base", "cause": "field_coverage", "count": 2},
        {"experiment": "other", "cause": "true_ranking_failure", "count": 99},
    ], "base")
    assert "field_coverage" in chart and "true_ranking_failure" not in chart


def test_failure_threshold_is_strict(study: dict[str, Path]) -> None:
    report = replace(run(study), failure_threshold=0)
    assert failures(report) == []


def test_heldout_requires_frozen_settings_including_metric_protocol(
    study: dict[str, Path], tmp_path: Path,
) -> None:
    report = run(study)
    save_evaluation(study["report.json"], report)
    freeze_configuration(study["report.json"], "words", study["frozen.json"])
    settings = load_frozen(study["frozen.json"], study["split.json"], study["corpus.csv"])
    heldout_pool = tmp_path / "heldout.json"
    with pytest.raises(ValueError, match="frozen"):
        prepare_pool(study["split.json"], study["corpus.csv"], [settings.experiment],
                     "heldout", heldout_pool, 2)
    prepare_pool(study["split.json"], study["corpus.csv"], [settings.experiment],
                 "heldout", heldout_pool, 2, frozen_file=study["frozen.json"])
    add_fixture_grades(study, heldout_pool)
    arguments = (
        heldout_pool, study["split.json"], study["corpus.csv"], study["labels.csv"],
        [settings.experiment], "heldout", 2, 1, 1.0,
    )
    heldout_report = evaluate(*arguments, frozen_file=study["frozen.json"])
    assert heldout_report.partition == "heldout"
    with pytest.raises(ValueError, match="thresholds"):
        evaluate(*arguments[:-2], 2, 1.0, frozen_file=study["frozen.json"])
    heldout_file = tmp_path / "heldout-report.json"
    save_evaluation(heldout_file, heldout_report)
    with pytest.raises(ValueError, match="development report"):
        freeze_configuration(heldout_file, "words", tmp_path / "bad-frozen.json")
    with pytest.raises(FileExistsError):
        freeze_configuration(study["report.json"], "words", study["frozen.json"])


@pytest.mark.parametrize("value", [
    {"fields": "title"}, {"field_weights": {"title": 1.5}},
    {"lowercase": "false"}, {"k1": True}, {"tokenizer": []}, {"extra": 1},
])
def test_invalid_config_json_fails_explicitly(value: dict) -> None:
    with pytest.raises(ValueError):
        parse_config(value)


def test_tampered_aggregate_is_not_used_for_selection(study: dict[str, Path]) -> None:
    data = asdict(run(study))
    data["experiments"][0]["aggregate"]["ndcg_at_k"] = 0.123456
    study["report.json"].write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="aggregate"):
        load_evaluation(study["report.json"])


def test_cli_rejects_heldout_ablations(study: dict[str, Path], tmp_path: Path) -> None:
    with pytest.raises(SystemExit) as error:
        main(["prepare", "--split-file", str(study["split.json"]),
              "--csv", str(study["corpus.csv"]), "--partition", "heldout",
              "--experiments", str(study["experiments.json"]), "--output", str(tmp_path / "no.json")])
    assert error.value.code == 2
    assert not (tmp_path / "no.json").exists()


def test_hosted_upload_round_trip_and_session_exports(
    study: dict[str, Path], monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    import streamlit_data

    monkeypatch.setattr(streamlit_data, "DATA_PATH", study["corpus.csv"])
    uploads = {
        "split": study["split.json"].read_bytes(), "pool": study["pool.json"].read_bytes(),
        "labels": study["labels.csv"].read_bytes(), "experiments": study["experiments.json"].read_bytes(),
    }
    output = streamlit_data.uploaded_workflow("Evaluate judgments", uploads, k=2, relevant_grade=1,
                                            failure_threshold=1.0)
    assert output.report is not None
    assert "evaluation.json" in output.files and "assignments.csv" in output.files
    assert streamlit_data.archive(output.files).startswith(b"PK")
    path = tmp_path / "human_causes.csv"
    assignments(path, failures(output.report))
    summary = streamlit_data.uploaded_workflow("Summarize assigned failures", {
        "report": output.files["evaluation.json"], "assignments": path.read_bytes(),
    })
    assert "failure_counts.svg" in summary.files
    frozen = streamlit_data.uploaded_workflow("Freeze final settings", {
        "report": output.files["evaluation.json"],
    }, selected="baseline")
    assert "frozen.json" in frozen.files
    assert json.loads(frozen.files["frozen.json"])["partition_selected_on"] == "dev"
    with pytest.raises(ValueError, match="Missing required"):
        streamlit_data.uploaded_workflow("Evaluate judgments", {})


def test_hosted_limits_are_explicit_not_silent() -> None:
    from streamlit_data import hosted_limits

    with pytest.raises(ValueError, match="capped at 5"):
        hosted_limits([Experiment("large", IndexConfig(field_weights={"title": 6}))])
    with pytest.raises(ValueError, match="at most 8"):
        hosted_limits([Experiment(str(number), IndexConfig()) for number in range(9)])


def test_summary_tables_and_chart_render_in_the_combined_app(
    study: dict[str, Path], tmp_path: Path,
) -> None:
    from streamlit.testing.v1 import AppTest

    from streamlit_data import WorkflowOutput

    report = run(study)
    path = tmp_path / "causes.csv"
    assignments(path, failures(report))
    files = {name: content.encode("utf-8") for name, content in summary_files(report, path).items()}
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "streamlit_app.py"),
                           default_timeout=120).run()
    app.session_state["research_output"] = WorkflowOutput(files, report)
    app.switch_page(str(Path("app_pages") / "research.py")).run()
    assert not app.exception
    assert len(app.dataframe) == 3
    charts = app.get("vega_lite_chart") or app.get("arrow_vega_lite_chart")
    assert len(charts) == 1
    spec = json.loads(charts[0].proto.spec)
    assert spec["mark"]["type"] == "bar"
    assert spec["encoding"]["x"]["field"] == "count"
    assert spec["encoding"]["y"]["field"] == "cause"


def test_cli_full_dev_workflow(study: dict[str, Path], tmp_path: Path) -> None:
    output = tmp_path / "cli-report.json"
    main(["evaluate", "--csv", str(study["corpus.csv"]), "--split-file", str(study["split.json"]),
          "--pool", str(study["pool.json"]), "--labels", str(study["labels.csv"]),
          "--experiments", str(study["experiments.json"]), "-k", "2", "--relevant-grade", "1",
          "--failure-threshold", "1", "--output", str(output)])
    review = tmp_path / "cli-review"
    main(["review", "--report", str(output), "--csv", str(study["corpus.csv"]), "--output", str(review)])
    assignments(review / "assignments.csv", failures(load_evaluation(output)))
    summary = tmp_path / "cli-summary"
    main(["summarize", "--report", str(output), "--assignments", str(review / "assignments.csv"),
          "--output", str(summary)])
    assert (summary / "failure_counts.svg").exists()
    main(["freeze", "--report", str(output), "--experiment", "words",
          "--output", str(study["frozen.json"])])
    assert load_frozen(study["frozen.json"], study["split.json"], study["corpus.csv"]).experiment.name == "words"
