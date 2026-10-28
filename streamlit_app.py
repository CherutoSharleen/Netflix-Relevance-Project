"""Keep retrieval research and the original classifier together without replacing either."""

from pathlib import Path

import streamlit as st


ROOT = Path(__file__).resolve().parent
st.set_page_config(page_title="Netflix retrieval lab", page_icon=":material/search:", layout="wide")
page = st.navigation([
    st.Page(ROOT / "app_pages" / "retrieval_search.py", title="Film search",
            icon=":material/search:", default=True, url_path="search"),
    st.Page(ROOT / "app_pages" / "research.py", title="Retrieval research",
            icon=":material/analytics:", url_path="research"),
    st.Page(ROOT / "app.py", title="Guess the genre",
            icon=":material/movie:", url_path="genre"),
], position="top")
page.run()
