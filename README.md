# 🎬 Guess the Genre

**Type a movie plot, and a Naive Bayes model guesses its genre.**

An interactive Streamlit app that predicts a movie's genre from its description, trained on the Netflix Movies and TV Shows dataset. It shows the prediction, how confident the model is, and which words influenced the answer.

👉 **[Try the live app](https://YOUR-APP-LINK.streamlit.app)**

<!-- Add a screenshot: take one of the running app, save it as screenshot.png in this repo, and it will show here -->
![App screenshot](screenshot.png)

---

## What it does

- **Predicts a genre** from any plot description you type
- **Shows probabilities** for every genre, so you can see when the model is unsure
- **Explains its answer** by listing the words that pushed it toward that genre
- **Lets you adjust the prior** (alpha) with a slider and see how accuracy changes
- **Shows what each genre "sounds like"**, meaning the words most typical of that genre

## How it works

1. **Data:** Netflix movie descriptions and their genres. I kept movies only, used each movie's first real genre (skipping labels like "International Movies"), and focused on the 6 most common genres.
2. **Text to numbers:** Each description is turned into word counts with `CountVectorizer`, removing common words like "the" and "and".
3. **Model:** A Multinomial Naive Bayes classifier learns how often each word appears in each genre. For a new plot, it uses **Bayes' theorem** to work out which genre is most likely given those words.
4. **The prior:** The `alpha` setting is a Bayesian prior. It assumes every word *could* appear in any genre, so one rare word can't completely decide the prediction.

## Results

| Model | Test accuracy |
|---|---|
| Always guess the most common genre (baseline) | XX% |
| Naive Bayes (alpha = 1.0) | XX% |

<!-- Fill these in with the numbers shown in the app's sidebar -->

**What I noticed:**
- <!-- e.g. Documentaries and horror were easiest to predict because they use very distinctive words -->
- <!-- e.g. Dramas and comedies were often confused with each other -->
- <!-- e.g. Very small alpha values made accuracy worse / better because... -->

## Limitations

- Descriptions are short (one or two sentences), so there isn't much text to learn from.
- Many movies belong to several genres, but the model only predicts one.
- Naive Bayes treats each word independently, so it misses context (e.g. "not scary").

## Run it locally

```bash
git clone https://github.com/YOUR-USERNAME/YOUR-REPO.git
cd YOUR-REPO
pip install -r requirements.txt
streamlit run app.py
```

## Tech used

Python · pandas · scikit-learn · Streamlit

## Data

[Netflix Movies and TV Shows](https://www.kaggle.com/datasets/shivamb/netflix-shows) dataset from Kaggle.

## Retrieval failure analysis (Phase 1)

The repository also contains a BM25 retrieval research track, separate from the
genre classifier. Its question is whether most retrieval failures originate in
data quality, tokenisation, field coverage, or vocabulary mismatch rather than
BM25 scoring itself. This is a hypothesis to test, not an established result.

Phase 1 adds an audited loader for all 8,807 movie and TV records and a
configurable in-memory BM25 index. The classifier and exploratory notebook are
unchanged. The legacy `search.py` remains available, with its failing list
debug-print removed.

```powershell
python -m pip install -r requirements.txt
python retrieval.py --audit-only
python retrieval.py --query "space adventure"
python retrieval.py --fields title description listed_in --weight title=3 --tokenizer words --stem --query "space adventure"
python -m pytest -q
```

See [DATA.md](DATA.md) for all cleaning decisions, actual missing-field counts,
tokenisation/stemming choices, and field-weight semantics. Weights repeat tokens
and therefore affect both term frequency and document length; this is not BM25F.
Run `python retrieval.py --help` for all options.

Judgment tooling, a frozen query split, metrics, failure analysis, and the research
write-up belong to later phases and have not been implemented. There are no
retrieval results or human relevance labels yet, and no tuning has been performed.

## Future ideas

- A game mode where you compete against the model to guess the genre of real Netflix movies
- Predict multiple genres per movie
- Compare Naive Bayes with other models like logistic regression
