"""Expose configurable BM25 search without presenting scores as human relevance."""

import csv
import io
import json
from dataclasses import asdict

import streamlit as st

from film_data import DATA_PATH, FIELDS
from query_sets import fingerprint
from retrieval import DEFAULT_FIELDS, IndexConfig, parse_weights, tokenize
from streamlit_data import ROOT, public_index, public_records


st.title("Netflix retrieval lab", anchor=False)
st.markdown("Search the film catalogue, then investigate **why retrieval fails**—without assuming the answer.")
st.caption("Research hypothesis: failures may originate upstream of BM25 scoring. This has not been established.")

with st.sidebar:
    st.header("Search configuration")
    with st.form("search_settings"):
        fields = st.multiselect("Fields", FIELDS, default=list(DEFAULT_FIELDS), key="retrieval_fields")
        weights = st.text_input("Integer field weights", placeholder="title=3 description=1", key="retrieval_weights")
        tokenizer = st.selectbox("Tokenisation", ("whitespace", "words"), key="retrieval_tokenizer")
        stemming = st.checkbox("English Porter stemming")
        preserve_case = st.checkbox("Preserve case")
        k1 = st.number_input("BM25 k1", min_value=0.1, max_value=10.0, value=1.5, step=0.1)
        b = st.number_input("Length normalization b", min_value=0.0, max_value=1.0, value=0.75)
        epsilon = st.number_input("IDF floor epsilon", min_value=0.0, max_value=2.0, value=0.25)
        k = st.number_input("Results to show", min_value=1, max_value=100, value=10)
        query = st.text_input("Search query", placeholder="Enter your own search phrase", key="retrieval_query")
        submitted = st.form_submit_button("Search", type="primary")
    st.caption("Weights repeat tokens; they also change document length. This is not BM25F.")
    st.caption("Use only development queries while choosing research settings.")

signature = fingerprint(DATA_PATH)
records, report = public_records(signature)
with st.container(horizontal=True):
    st.metric("Catalogue records", f"{len(records):,}")
    st.metric("Movies", f"{sum(row['type'] == 'Movie' for row in records):,}")
    st.metric("TV shows", f"{sum(row['type'] == 'TV Show' for row in records):,}")

if submitted:
    st.session_state.pop("retrieval_results", None)
    try:
        if not query.strip():
            raise ValueError("Enter a query; no example research query is pre-filled.")
        config = IndexConfig(
            fields=tuple(fields), field_weights=parse_weights(weights.split()), tokenizer=tokenizer,
            lowercase=not preserve_case, stemming=stemming, k1=k1, b=b, epsilon=epsilon,
        )
        with st.spinner("Building the selected representation and searching..."):
            index = public_index(json.dumps(asdict(config), sort_keys=True), signature)
            rows = [
                {"show_id": result.record["show_id"], "title": result.record["title"],
                 "type": result.record["type"], "release_year": result.record["release_year"],
                 "listed_in": result.record["listed_in"], "description": result.record["description"],
                 "score": result.score}
                for result in index.search(query, int(k))
            ]
        st.session_state["retrieval_results"] = {
            "query": query, "rows": rows, "config": asdict(config),
            "has_terms": any(token in index.model.idf for token in tokenize(query, config)),
        }
    except ValueError as error:
        st.error(str(error))

result = st.session_state.get("retrieval_results")
if result is not None:
    st.subheader(f"Results for: {result['query']}")
    st.caption("BM25 scores are not relevance grades. Ties follow source CSV order.")
    if not result["has_terms"]:
        st.warning("No query terms occur in this index. All scores are zero; this order is not evidence of relevance.")
    st.dataframe(result["rows"], hide_index=True)
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(result["rows"][0]))
    writer.writeheader()
    writer.writerows(result["rows"])
    st.download_button("Download search results", buffer.getvalue(), "search_results.csv", "text/csv")
    with st.expander("Exact search configuration"):
        st.json(result["config"])
else:
    st.info("Enter a query in the sidebar and select Search. No relevance judgments or research results have been supplied.")

with st.expander("Data audit and research boundaries"):
    st.json(asdict(report))
    st.caption("The source CSV is unchanged. See DATA.md for every cleaning decision.")
    st.download_button("Download data documentation", (ROOT / "DATA.md").read_text(encoding="utf-8"), "DATA.md")
    st.markdown("Human-only labelling and failure assignments are separate from this search demonstration. "
                "The research page stops when required judgments are missing.")
