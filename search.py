
import pandas as pd
from rank_bm25 import BM25Okapi

CSV_PATH = "netflix_titles.csv"


def load_documents(path=CSV_PATH):
    """Turn each row into one searchable text blob."""
    df = pd.read_csv(path).fillna("")

    # These are the fields we let people search over.
    fields = ["title", "description", "listed_in", "cast", "director"]

    df["doc"] = df[fields].agg(" ".join, axis=1)
    return df


def tokenize(text):
    """Crude on purpose. Improving this is experiment #2."""
    return text.lower().split()


def build_index(df):
    corpus = [tokenize(doc) for doc in df["doc"]]
    return BM25Okapi(corpus)


def search(query, bm25, df, k=10):
    """Return the top-k rows for a query, with their BM25 scores."""
    scores = bm25.get_scores(tokenize(query))
    df = df.assign(score=scores)
    return df.nlargest(k, "score")[
        ["show_id", "title", "type", "release_year", "listed_in", "score"]
    ]


def main():
    print("Loading...")
    df = load_documents()
    bm25 = build_index(df)
    print(f"Indexed {len(df)} titles.\n")

    while True:
        query = input("query> ").strip()
        if query.lower() in {"quit", "exit", "q", ""}:
            break

        results = search(query, bm25, df)
        print()
        for i, (_, row) in enumerate(results.iterrows(), 1):
            print(f"{i:2}. [{row.score:6.2f}] {row.title} ({row.release_year})")
            print(f"     {row.listed_in}")
        print()


if __name__ == "__main__":
    main()
