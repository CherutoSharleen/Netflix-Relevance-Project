"""
Guess the Genre: a Naive Bayes movie genre classifier (Netflix dataset)

Run locally:
    pip install streamlit pandas scikit-learn
    streamlit run app.py

Put netflix_titles.csv in the same folder as this file.
"""

import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

# Labels that aren't really genres, so we skip them when picking a movie's main genre
SKIP = {"International Movies", "Independent Movies", "Movies"}
N_GENRES = 6  # keep the most common genres so the problem stays manageable


# ---------- 1. Load and prepare the data ----------
@st.cache_data
def load_data(path="netflix_titles.csv"):
    df = pd.read_csv(path)
    df = df[df["type"] == "Movie"]                       # movies only (TV genres are named differently)
    df = df.dropna(subset=["description", "listed_in"])  # need both text and genre

    def main_genre(genres):
        # "listed_in" looks like "Dramas, International Movies, Romantic Movies"
        # -> take the first one that's a real genre
        for g in genres.split(", "):
            if g not in SKIP:
                return g
        return None

    df["genre"] = df["listed_in"].apply(main_genre)
    df = df.dropna(subset=["genre"])
    top = df["genre"].value_counts().head(N_GENRES).index
    return df[df["genre"].isin(top)][["title", "description", "genre"]]


# ---------- 2. Train the model ----------
@st.cache_resource
def train(df, alpha):
    X_train, X_test, y_train, y_test = train_test_split(
        df["description"], df["genre"],
        test_size=0.2, random_state=42, stratify=df["genre"],
    )
    vec = CountVectorizer(stop_words="english", min_df=2)  # turn text into word counts
    model = MultinomialNB(alpha=alpha)                     # alpha = the Bayesian prior (smoothing)
    model.fit(vec.fit_transform(X_train), y_train)

    acc = accuracy_score(y_test, model.predict(vec.transform(X_test)))
    baseline = y_test.value_counts(normalize=True).iloc[0]  # always guessing the most common genre
    return vec, model, acc, baseline


# ---------- 3. The app ----------
st.set_page_config(page_title="Guess the Genre", page_icon="🎬")
st.title("🎬 Guess the Genre")
st.write("Type a movie plot and a **Naive Bayes** model trained on Netflix descriptions guesses the genre.")

df = load_data()

with st.sidebar:
    st.header("Model settings")
    alpha = st.slider(
        "Prior strength (alpha)", 0.01, 5.0, 1.0, 0.01,
        help="How much to trust 'every word could appear in any genre' before seeing data. "
             "Low = trust the data fully, high = smooth things out.",
    )
    vec, model, acc, baseline = train(df, alpha)
    st.metric("Test accuracy", f"{acc:.0%}")
    st.caption(f"Guessing the most common genre every time would get {baseline:.0%}.")
    st.caption(f"Trained on {len(df):,} movies across {N_GENRES} genres.")

text = st.text_area(
    "Describe a movie plot:",
    "A group of friends go camping in the woods and are hunted by a mysterious creature.",
)

if text.strip():
    probs = model.predict_proba(vec.transform([text]))[0]
    result = pd.Series(probs, index=model.classes_).sort_values(ascending=False)

    st.subheader(f"Prediction: {result.index[0]} ({result.iloc[0]:.0%})")
    st.bar_chart(result)

    # Which words pushed the model toward its answer?
    vocab = vec.vocabulary_
    words = {w for w in vec.build_analyzer()(text) if w in vocab}
    if words:
        i = list(model.classes_).index(result.index[0])
        logp = model.feature_log_prob_
        influence = pd.Series(
            {w: logp[i, vocab[w]] - logp[:, vocab[w]].mean() for w in words}
        ).sort_values(ascending=False)
        st.write(f"**Words that pointed toward {result.index[0]}:**")
        st.write(", ".join(influence[influence > 0].head(8).index) or "none stood out")
    else:
        st.info("None of these words appeared in the training data, so the model is just guessing.")

# Explore what the model learned
st.divider()
st.subheader("What each genre 'sounds like'")
genre = st.selectbox("Pick a genre", model.classes_)
i = list(model.classes_).index(genre)
words = vec.get_feature_names_out()
distinct = pd.Series(model.feature_log_prob_[i] - model.feature_log_prob_.mean(axis=0), index=words)
st.write(", ".join(distinct.sort_values(ascending=False).head(15).index))

with st.expander("See example movies from the data"):
    st.dataframe(df[df["genre"] == genre].sample(5, random_state=1), hide_index=True)
