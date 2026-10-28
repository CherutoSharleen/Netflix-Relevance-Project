"""Exercise the combined navigation while keeping the legacy classifier independently runnable."""

from pathlib import Path

from streamlit.testing.v1 import AppTest


ROOT = Path(__file__).resolve().parents[1]


def combined() -> AppTest:
    return AppTest.from_file(str(ROOT / "streamlit_app.py"), default_timeout=120).run()


def test_search_loads_without_invented_relevance_results() -> None:
    app = combined()
    assert not app.exception
    assert app.metric[0].value == "8,807"
    assert not app.dataframe
    assert app.text_input(key="retrieval_query").value == ""


def test_search_and_invalid_configuration_are_handled() -> None:
    app = combined()
    app.text_input(key="retrieval_query").set_value("zzzznomatchingtermszzzz")
    app.button[0].click().run()
    assert not app.exception
    assert app.warning
    assert len(app.dataframe[0].value) == 10
    assert app.dataframe[0].value.iloc[0]["show_id"] == "s1"
    app.multiselect(key="retrieval_fields").set_value([])
    app.button[0].click().run()
    assert not app.exception
    assert app.error and not app.dataframe


def test_research_requires_human_inputs_before_any_result() -> None:
    app = combined()
    app.switch_page(str(Path("app_pages") / "research.py")).run()
    assert not app.exception
    assert not app.dataframe
    app.button(key="research_run").click().run()
    assert not app.exception
    assert "Missing required" in app.error[0].value
    assert not app.dataframe


def test_combined_navigation_preserves_original_genre_page() -> None:
    app = combined()
    app.switch_page("app.py").run()
    assert not app.exception
    assert any("Guess the Genre" in title.value for title in app.title)
    assert len(app.metric) == 1
    assert app.slider[0].value == 1.0
    standalone = AppTest.from_file(str(ROOT / "app.py"), default_timeout=120).run()
    assert not standalone.exception
    assert standalone.metric[0].value == app.metric[0].value
