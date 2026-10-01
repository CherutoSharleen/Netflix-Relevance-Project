
import pandas as pd
from rank_bm25 import BM25Okapi

CSV_PATH = "netflix_titles.csv"


def load_search_rowuments(path=CSV_PATH):
    netflix_movies = pd.read_csv(path).fillna("")
    # fields to search over
    search_fields = ["title", "description", "listed_in", "cast", "director", "country"]
    # make it a blob of text juu bm25 likes text removing structure
    netflix_movies["search_row"] = netflix_movies[search_fields].agg(" ".join, axis=1)
    # print(netflix_movies["search_row"].head(10))
    return netflix_movies


def tokenize(text):
    """Crude on purpose. Improving this is experiment #2."""

    return text.lower().split()


def build_index(netflix_movies):
    corpus = [tokenize(word) for word in netflix_movies["search_row"]]
    print(corpus.head(10))

    return BM25Okapi(corpus)


def search(query, bm25, netflix_movies, k=10):
    """Return the top-k rows for a query, with their BM25 scores."""
    scores = bm25.get_scores(tokenize(query))
    netflix_movies = netflix_movies.assign(score=scores)
    return netflix_movies.nlargest(k, "score")[
        ["show_id", "title", "type", "release_year", "listed_in", "score"]
    ]


def main():
    print("Loading...")
    netflix_movies = load_search_rowuments()
    bm25 = build_index(netflix_movies)
    print(f"Indexed {len(netflix_movies)} titles.\n")

    while True:
        query = input("query> ").strip()
        if query.lower() in {"quit", "exit", "q", ""}:
            break

        search_results = search(query, bm25, netflix_movies)
        print()
        for i, (_, row) in enumerate(search_results.iterrows(), 1):
            print(f"{i:2}. [{row.score:6.2f}] {row.title} ({row.release_year})")
            print(f"{row.listed_in}")
        print()


if __name__ == "__main__":
    main()
