# Failure analysis and ablations

This workflow implements the research procedure; it does not supply the research
queries, labels, failure causes, or results. All failure causes are assigned by
you. The thesis about upstream failures remains untested.

## Before collecting judgments

Follow [LABELING.md](LABELING.md) to create a human-authored query CSV and freeze
the development/held-out split. Inspect the proposed, **untuned**
`experiments.json` and predeclare the runs you intend to compare. Its first entry
is the baseline for metric deltas and the failure-count chart. Every entry has
a unique `name` and a `config` using Phase 1's `IndexConfig` fields.

The examples separately change tokenisation, stemming, title repetition weight,
and field selection. These are mechanics, not claims about better settings.
Weights change document length as well as term frequency. Keep the judgment
protocol, relevance threshold, query set, and cutoff fixed across comparisons.

Prepare a union of top-k candidates from all predeclared runs, rather than only
the baseline:

```powershell
python analysis_cli.py prepare --split-file judgment_data\split.json --experiments experiments.json -k 10 --output judgment_data\dev_pool.json
python label_cli.py label --split-file judgment_data\split.json --pool judgment_data\dev_pool.json --labels judgment_data\judgments.csv
```

`--additional-pairs` accepts an optional ungraded `query_id,show_id` CSV for
human-selected records. The union is de-duplicated and shuffled with recorded
seed 42. Every query in the chosen partition receives candidates.

Use the blind re-label/self-consistency procedure in LABELING.md on this same
saved pool. Re-label grades are never substituted for initial grades in
effectiveness calculations. All pool/split/corpus fingerprints are checked.

An existing baseline-only pool is still supported. If an ablation retrieves an
unjudged record, evaluation stops and identifies the query and missing records.
It never treats unjudged records as irrelevant. Do not edit a pool already tied
to grades: create a new versioned pool/log and label it explicitly if the planned
coverage changes. The current tool does not migrate grades across changed pools.

## Evaluate development runs

Choose the binary relevance threshold and failure threshold yourself. The
following **illustrates syntax only**, not a recommended or tuned protocol:

```powershell
python analysis_cli.py evaluate --split-file judgment_data\split.json --pool judgment_data\dev_pool.json --labels judgment_data\judgments.csv --experiments experiments.json -k 10 --relevant-grade 1 --failure-threshold 0.5 --output judgment_data\dev_report.json
```

There are no defaults for those two thresholds in development evaluation. The
binary threshold is inclusive. A query fails the review threshold when its
**nDCG@k is strictly below** the chosen value.

The report records the full configurations, input fingerprints, selected
partition, k, relevance threshold, failure threshold, per-query metrics,
retrieved IDs/scores/human grades, and judged relevant records missed at k.
Aggregate precision, recall, and nDCG are equal-query means. Reciprocal ranks
are averaged into `mrr_at_k`: this evaluates only the first k returned records,
not the complete corpus ranking. Explicitly report this truncation.

Recall and ideal nDCG depend on the supplied judged list, not an exhaustive
corpus-wide relevance census. See [METRICS.md](METRICS.md) for formulas and
undefined cases. A query with no judged positives at the chosen threshold makes
recall undefined; all-zero gains make nDCG undefined. Evaluation stops explicitly,
without omitting queries, changing grades, or writing a partial numerical report.
Review such queries as protocol issues before choosing a transparent study policy.

## Export evidence, then assign causes yourself

```powershell
python analysis_cli.py review --report judgment_data\dev_report.json --output judgment_data\dev_review
```

The new output directory contains:

| File | Purpose |
|---|---|
| `review.csv` | Query text, top-k results, missed judged-relevant records, human grades, and field evidence for each below-threshold query/run |
| `assignments.csv` | One row per failing experiment/query, with **blank** cause and notes for you to fill |
| `taxonomy.json` | Proposed category names and descriptions; no assigned causes |

Review rows include all cleaned record fields, missing fields, literal query
token overlaps in each field, matching fields excluded from that index, and
reference word-token/Porter-stem overlaps. This is the evidence for fields that
could match under those explicit representations. It is not proof of semantic
relevance, a reconstructed missing description, or an automatically diagnosed
cause. The reference alternatives do not re-rank results in the review sheet.

Proposed primary causes are `tokenisation`, `missing_or_poor_field_data`,
`field_coverage`, `vocabulary_mismatch`, `field_weighting`, `ambiguous_judgment`,
`true_ranking_failure`, and `other`. Assign one primary category per failing
query/experiment, using notes for mixed or uncertain mechanisms. Explain `other`
in notes. The tool never chooses a cause, even when token evidence looks suggestive.

CSV text may include user-authored queries and original dataset text. When using
a spreadsheet, import text columns as text instead of executing cell formulas.

## Summarize only human-completed assignments

```powershell
python analysis_cli.py summarize --report judgment_data\dev_report.json --assignments judgment_data\dev_review\assignments.csv --output judgment_data\dev_summary
```

Blank, duplicate, unknown, extra, or missing assignments are errors. No missing
category is silently counted as zero. After all expected assignments exist,
the new directory contains:

- `failure_counts.csv`: counts grouped by experiment and your primary cause.
- `ablation_deltas.csv`: metric values and absolute differences from the first
  experiment, using the same queries and judgment list.
- `failure_counts.svg`: **one chart for the baseline's human-assigned causes**.
  Ablations' correlated failures are not pooled into the baseline distribution.

The count unit is a failing **query/experiment pair**, not a missed record or an
independent annotator. If a valid report has no failures, the summary explicitly
records that situation rather than inventing causes. Determine what counts as an
upstream cause in your study protocol before interpreting the distribution.
Do not treat improved metrics or literal overlaps alone as proof of causation.

## Freeze one choice before opening held-out results

Select an experiment yourself from a development report:

```powershell
python analysis_cli.py freeze --report judgment_data\dev_report.json --experiment baseline --output judgment_data\frozen.json
python analysis_cli.py prepare --partition heldout --split-file judgment_data\split.json --frozen-config judgment_data\frozen.json --output judgment_data\heldout_pool.json
python label_cli.py label --split-file judgment_data\split.json --pool judgment_data\heldout_pool.json --labels judgment_data\judgments.csv
python analysis_cli.py evaluate --partition heldout --split-file judgment_data\split.json --pool judgment_data\heldout_pool.json --labels judgment_data\judgments.csv --frozen-config judgment_data\frozen.json --output judgment_data\heldout_report.json
```

The frozen file includes your chosen configuration, original development report
fingerprint, corpus/split identities, k, and metric thresholds. Held-out commands
require this file, accept one configuration only, and reject changes to the
frozen settings or thresholds. No held-out ablation sweep or automatic selection
is implemented. A held-out report cannot be used to freeze a new choice.

Exclusive-create snapshots prevent accidental overwrites, not deliberate file
editing or repeated experimentation under new filenames. The tool cannot stop
you manually inspecting held-out queries through other code. Protect the held-out
set as part of the research protocol and document any leakage honestly.

## Combined app and local durability

`python -m streamlit run streamlit_app.py` opens search, research workflows, and
the original genre classifier. The research page performs the same validations
using uploaded files, offers downloads/ZIPs, displays metric/delta tables only
after valid evaluation, and draws the chart only after you supply completed
failure assignments.

The CLI remains the durable labelling interface. The hosted app does not collect
anonymous grades or maintain a shared judgment database. Uploads use per-operation
temporary directories with fixed server-chosen filenames; outputs stay in your
Streamlit session until downloaded. They are not globally cached or committed.
Session resets, app restarts, and Community Cloud sleep can lose these outputs.
Keep local backups; do not upload sensitive data to a public demo.

To keep the shared free-hosted demonstration bounded, its limits are 8
configurations, 100 queries, depth 100, and repetition weights no greater than 5.
Larger studies/weights remain available through the local CLI; the app raises
explicit errors rather than silently truncating them.
