# Netflix retrieval failure analysis

BM25 retrieval over the Netflix catalogue, with human relevance judgments and
human-assigned failure causes. A combined Streamlit app includes this research
track and the existing **Guess the Genre** classifier.

**Status:** implementation is available; research queries, relevance labels, and
failure assignments have not been supplied. All effectiveness results below
remain explicit placeholders. No empirical conclusion has been established.

## Question

Do most retrieval failures originate upstream of the ranking function, in data
quality, tokenisation, field coverage, or query-document vocabulary mismatch,
rather than BM25 scoring itself?

This is a thesis to test, not an assumption used to label failures. Only the
researcher assigns relevance and failure categories.

## Method

1. Audit and conservatively clean the source data while preserving stable IDs.
2. Freeze human-authored queries into development and held-out partitions
   **before tuning**. Use only development queries to choose settings.
3. Predeclare retrieval configurations and pool their candidates. Add
   human-selected records to improve judgment coverage.
4. Enter grades 0-3 yourself, then blindly re-label a saved sample later.
5. Calculate metrics from their definitions; compare development ablations.
6. Review every query below a chosen nDCG threshold and assign a primary cause.
7. Freeze one configuration and metric protocol before final held-out evaluation.

The failure distribution and metric deltas are evidence for discussion, not
automatic proof that a particular mechanism caused a failure.

## Dataset and cleaning

The checked-in [Netflix Movies and TV Shows dataset](https://www.kaggle.com/datasets/shivamb/netflix-shows)
contains **8,807 records: 6,131 movies and 2,676 TV shows**. Retrieval uses both
types, unlike the movies-only genre classifier.

Cleaning normalizes Unicode/whitespace, preserves missing text as empty, checks
duplicates/identifiers, and narrowly repairs three misplaced duration values.
No rows are removed from the current source, and the original CSV is unchanged.
See [DATA.md](DATA.md) for every decision and measured missing-field counts.

BM25 defaults search title, description, genres, cast, director, and country.
Fields, integer token-repetition weights, tokenisation, case handling, English
Porter stemming, and BM25 parameters are configurable. Repetition weights change
document lengths too; they are not BM25F.

## Labelling protocol and self-consistency

The CLI shows a query and complete cleaned record, without scores, ranks, prior
grades, or a suggested label. It accepts only your explicit integer grade 0-3
and records a UTC timestamp immediately. There is no default grade.

Write down your interpretation of the scale and query intent before labelling.
Binary metrics require you to explicitly choose which grades count as relevant.
The judgment list size is **[RESULT: number of judged query-record pairs]** over
**[RESULT: number of research queries]** queries.

Blind re-labelling samples already graded pairs without replacement, saves that
exact queue, and hides earlier grades. Choose a delay and record it yourself.
There is no enforced waiting period; the tool cannot prevent memory or reading
your own original CSV. See [LABELING.md](LABELING.md).

| Self-consistency item | Result |
|---|---|
| Re-labelled pairs | [RESULT: number of re-labelled pairs] |
| Delay between passes | [RESULT: re-labelling delay] |
| Exact agreement | [RESULT: exact self-agreement] |
| Quadratic-weighted Cohen's kappa | [RESULT: quadratic-weighted Cohen's kappa] |

Undefined kappa is reported explicitly, not replaced with a perfect score.

## Metrics

`metrics.py` implements precision@k, recall@k, reciprocal rank/MRR, and nDCG@k
using the standard library. Tests include hand-worked examples and a separate
`ir-measures` cross-check. See [METRICS.md](METRICS.md).

Precision divides by k even for short runs. Recall is relative to the supplied
human judgment list. nDCG uses **linear grade gain** with base-2 log discounts.
The analysis runner reports **MRR@k**, since it evaluates runs truncated at k.
All query averages weight queries equally.

Missing labels are errors, never implicit zero grades. Undefined recall/nDCG
cases stop evaluation rather than silently dropping queries.

| Measure | Development | Held-out, frozen configuration |
|---|---|---|
| Precision@10 | [RESULT: precision@10 dev] | [RESULT: precision@10 heldout] |
| Recall@10, judged list | [RESULT: recall@10 dev] | [RESULT: recall@10 heldout] |
| MRR@10 | [RESULT: MRR@10 dev] | [RESULT: MRR@10 heldout] |
| nDCG@10, linear gain | [RESULT: nDCG@10 dev] | [RESULT: nDCG@10 heldout] |

The table's cutoff is a reporting placeholder, not a chosen study protocol.
Document the actual cutoff, binary threshold, and failure threshold before use.

## Failure breakdown and ablations

For each query below your nDCG threshold, the workflow exports top results,
missed judged-relevant records, original field text, missing fields, matching
fields excluded from the index, and literal overlap under the configured,
word-token, and stemmed representations.

You fill every primary cause in `assignments.csv`. Proposed categories do not
pre-assign any failure. A complete assignment file produces per-experiment
counts, metric-delta tables, and one baseline failure-count chart.

| Baseline primary cause | Count |
|---|---|
| Tokenisation | [RESULT: baseline tokenisation failures] |
| Missing or poor field data | [RESULT: baseline data-quality failures] |
| Field coverage | [RESULT: baseline field-coverage failures] |
| Vocabulary mismatch | [RESULT: baseline vocabulary failures] |
| Field weighting | [RESULT: baseline field-weighting failures] |
| Ambiguous judgment | [RESULT: baseline ambiguous judgments] |
| True ranking failure | [RESULT: baseline ranking failures] |
| Other, explained in notes | [RESULT: baseline other failures] |

Development ablation deltas: **[RESULT: ablation metric-delta table]**.
Baseline failure-count chart: **[RESULT: human-assigned failure-count chart]**.
Thesis assessment: **[RESULT: interpretation after human failure review]**.

See [ANALYSIS.md](ANALYSIS.md) for the full pooling, evaluation, review, summary,
and held-out freeze procedure. The shipped `experiments.json` contains untuned
mechanical examples, not recommended winning configurations.

## Limitations

- **Single annotator:** relevance and cause assignments reflect one person's
  interpretation. Self-consistency is not inter-annotator agreement.
- **Judgment-list size and coverage:** [RESULT: judgment-list size and coverage].
  Pooling can omit relevant records; judged-list recall is not exhaustive corpus
  recall. Unknown relevance is never silently treated as irrelevant.
- **Development-set overfitting:** repeated ablations can adapt to development
  queries and judgments. Freeze settings before final held-out evaluation.
- Queries can leak through paraphrases even when duplicate text is rejected.
  The software guards workflow inputs, not deliberate manual inspection.
- Short, multilingual metadata can limit retrieval. English stemming can damage
  names or non-English text. The catalogue is a historical snapshot.
- Failure causes can overlap; a single primary category simplifies them.
  Literal token evidence and ablation improvements alone do not prove causation.
- A re-label sample can be small or affected by memory, timing, and selection.
- MRR is depth-limited in the analysis runner. Undefined metrics require a
  transparent protocol decision rather than hidden query exclusion.
- Community Cloud sessions/files are not durable storage for research artifacts.

## Reproduce locally

Use Python 3.12 or newer in a virtual environment. From this checkout:

```powershell
python -m pip install -r requirements.txt
python retrieval.py --audit-only
python -m streamlit run streamlit_app.py
```

The combined app offers **Film search**, **Retrieval research**, and **Guess the
genre**. Search needs no labels; research evaluation stops until you upload your
own valid frozen split, pool, and judgments. Hosted outputs must be downloaded
and backed up. Human labelling remains local through `label_cli.py`.

Create `judgment_data\queries.csv` yourself with header `query_id,query`, then:

```powershell
python label_cli.py split --queries judgment_data\queries.csv --output judgment_data\split.json --heldout-fraction 0.2 --seed 42
python analysis_cli.py prepare --split-file judgment_data\split.json --experiments experiments.json -k 10 --output judgment_data\dev_pool.json
python label_cli.py label --split-file judgment_data\split.json --pool judgment_data\dev_pool.json --labels judgment_data\judgments.csv
```

Choose thresholds explicitly before the evaluation commands documented in
ANALYSIS.md. Complete blind re-labelling as documented in LABELING.md. Do not
commit live judgments accidentally: `judgment_data/` is gitignored, not backed up.
Explicitly version finalized, non-sensitive research artifacts when appropriate.

```powershell
python -m pip install -r requirements-test.txt
python -m pytest -q
```

All fixture grades are artificial test data, never research judgments.

## Deploy the combined app free

Push this branch to GitHub. In [Streamlit Community Cloud](https://share.streamlit.io/),
sign in with your account, create an app from
`CherutoSharleen/Netflix-Relevance-Project`, select the branch containing these
files, and set the main file to **`streamlit_app.py`**. Choose Python 3.12 in
advanced settings. `requirements.txt` contains runtime dependencies; the
independent IR cross-check dependency is only in `requirements-test.txt`.

The repository root is the app working directory. No secrets or external data
services are required. The source CSV ships with the repository. Never put
credentials in source control; `.streamlit/secrets.toml` is ignored.

Deployment URL: **[RESULT: verified Streamlit deployment URL]**.
GitHub push alone is not a Streamlit deployment: creation requires your
authenticated Community Cloud session. Point an existing genre-only deployment
at the new entry point only if you intend to replace its hosted interface.

## Existing genre classifier

The original `app.py` is preserved and remains independently runnable:

```powershell
python -m streamlit run app.py
```

It uses movie descriptions, word counts, and Multinomial Naive Bayes to predict
a main genre, with probabilities, influential words, and an adjustable alpha.
Its movies-only preparation and classification task are distinct from BM25
retrieval; its classifier accuracy is not a retrieval result. The combined
navigation runs the same original entry point without rewriting that model.
