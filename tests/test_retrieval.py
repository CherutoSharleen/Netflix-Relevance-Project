"""Check representation ablations without creating or implying relevance judgments."""

import json

import numpy as np
import pandas as pd
import pytest
from rank_bm25 import BM25Okapi

from film_data import FIELDS, FilmRecord
from retrieval import (
    IndexConfig, build_index, document_tokens, main, parse_weights, tokenize,
)
from search import build_index as build_legacy_index, search as legacy_search


def records() -> list[FilmRecord]:
    return [
        dict.fromkeys(FIELDS, "") | {"show_id": f"s{number}", "title": title,
                                     "description": description}
        for number, (title, description) in enumerate([
            ("Orbit", "Astronauts exploring space"),
            ("Kitchen", "Chefs cooking dinner"),
            ("Forest", "Animals living outdoors"),
            ("Ocean", "Fish swimming underwater"),
        ], 1)
    ]


def test_defaults_preserve_legacy_concatenated_representation() -> None:
    rows = records()
    config = IndexConfig()
    corpus = [
        " ".join(row[name] for name in config.fields).lower().split()
        for row in rows
    ]
    index = build_index(rows)
    expected = BM25Okapi(corpus)
    np.testing.assert_array_equal(index.scores("SPACE"), expected.get_scores(["space"]))
    assert [document_tokens(row, config) for row in rows] == corpus


def test_legacy_search_still_builds_and_returns_results() -> None:
    rows = pd.DataFrame(records())
    rows["search_row"] = rows["title"] + " " + rows["description"]
    model = build_legacy_index(rows)
    result = legacy_search("cooking", model, rows, k=1)
    assert result.iloc[0]["show_id"] == "s2"


def test_query_and_document_tokenization_are_identical() -> None:
    config = IndexConfig(fields=("title",), tokenizer="words", stemming=True)
    assert document_tokens({"title": "RUNNING, caf\u00e9!"}, config) == tokenize("RUNNING, caf\u00e9!", config)
    assert tokenize("Running, forests!", config) == ["run", "forest"]


def test_tokenizer_and_case_ablations() -> None:
    assert tokenize("Space, O'Brien sci-fi", IndexConfig()) == ["space,", "o'brien", "sci-fi"]
    assert tokenize("Space, O'Brien sci-fi", IndexConfig(tokenizer="words")) == [
        "space", "o", "brien", "sci", "fi",
    ]
    assert tokenize("NASA Space", IndexConfig(lowercase=False)) == ["NASA", "Space"]
    assert tokenize("NASA", IndexConfig(lowercase=False, stemming=True)) == ["NASA"]


def test_weighting_repeats_tokens_before_bm25_length_normalization() -> None:
    config = IndexConfig(fields=("title", "description"), field_weights={"title": 3})
    row = records()[0]
    tokens = document_tokens(row, config)
    assert tokens == ["orbit", "orbit", "orbit", "astronauts", "exploring", "space"]
    index = build_index(records(), config)
    assert index.model.doc_len[0] == 6
    expected = BM25Okapi([document_tokens(row, config) for row in records()])
    np.testing.assert_array_equal(index.scores("orbit"), expected.get_scores(["orbit"]))


def test_excluded_fields_cannot_match() -> None:
    index = build_index(records(), IndexConfig(fields=("title",)))
    assert np.all(index.scores("space") == 0)
    assert index.search("orbit", 1)[0].record["show_id"] == "s1"


def test_bm25_parameters_reach_the_model() -> None:
    config = IndexConfig(k1=2.0, b=0.25, epsilon=0.1)
    model = build_index(records(), config).model
    assert (model.k1, model.b, model.epsilon) == (2.0, 0.25, 0.1)


def test_top_k_and_deterministic_zero_score_ties() -> None:
    index = build_index(records())
    assert index.search("cooking", 1)[0].record["show_id"] == "s2"
    assert [result.record["show_id"] for result in index.search("unknown", 2)] == ["s1", "s2"]
    assert all(result.score == 0 for result in index.search(""))
    assert len(index.search("space", 100)) == 4
    with pytest.raises(ValueError, match="positive"):
        index.search("space", 0)


def test_empty_documents_are_allowed_but_empty_vocabulary_is_not() -> None:
    rows = records()
    rows[0]["title"] = ""
    index = build_index(rows, IndexConfig(fields=("title",)))
    assert index.model.doc_len[0] == 0
    with pytest.raises(ValueError, match="empty vocabulary"):
        build_index(rows, IndexConfig(fields=("cast",)))
    with pytest.raises(ValueError, match="empty vocabulary"):
        build_index([])


@pytest.mark.parametrize("kwargs", [
    {"fields": ()}, {"fields": ("title", "title")}, {"fields": ("typo",)},
    {"field_weights": {"rating": 2}}, {"field_weights": {"title": 0}},
    {"field_weights": {"title": -1}}, {"field_weights": {"title": 1.5}},
    {"field_weights": {"title": True}}, {"tokenizer": "typo"}, {"k1": 0},
    {"k1": float("inf")}, {"b": -0.1}, {"b": 1.1}, {"b": float("nan")},
    {"epsilon": -1}, {"epsilon": float("nan")},
])
def test_invalid_configuration_is_rejected(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        IndexConfig(**kwargs)


def test_weight_arguments() -> None:
    assert parse_weights(["title=3", "description=2"]) == {"title": 3, "description": 2}
    for values in (["title"], ["title=1.5"], ["title=2", "title=3"]):
        with pytest.raises(ValueError):
            parse_weights(values)


def test_cli_audit_outputs_data_identity_and_configuration(capsys: pytest.CaptureFixture[str]) -> None:
    main(["--audit-only", "--tokenizer", "words", "--stem", "--weight", "title=3"])
    report = json.loads(capsys.readouterr().out)
    assert report["cleaning"]["output_rows"] == 8807
    assert len(report["sha256"]) == 64
    assert report["config"]["stemming"] is True
    assert report["config"]["field_weights"] == {"title": 3}


def test_cli_rejects_bad_configuration(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as error:
        main(["--weight", "title=0"])
    assert error.value.code == 2
    assert "positive integers" in capsys.readouterr().err
