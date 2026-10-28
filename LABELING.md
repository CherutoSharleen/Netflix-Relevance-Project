# Human judgment protocol (Phase 2)

The CLI never generates, suggests, or pre-fills relevance grades. Only you enter
integers 0-3. The research query list and real judgments have not been supplied.
Do not treat the synthetic test fixtures as research data.

## 1. Author queries and freeze their split before tuning

Create a local `judgment_data` directory. Inside it, write `queries.csv` in UTF-8
with exactly this header and then your own query rows:

```csv
query_id,query
```

Give each query a stable, unique ID. Quote query text containing commas using
normal CSV quoting. Document your intent and your own interpretation of the
0-3 relevance scale before labelling; use that protocol consistently. No default
grade is provided. Do not tailor the queries to a tuned system's successes.

```powershell
python label_cli.py split --queries judgment_data\queries.csv --output judgment_data\split.json --heldout-fraction 0.2 --seed 42
```

The split is by **query**, never by candidate record. IDs are sorted then shuffled
with the recorded seed; the held-out size is the ceiling of the requested fraction,
bounded to keep at least one query in each partition. At least two queries are
required. Query text is NFC/whitespace-normalized; duplicate IDs and
case-insensitively identical normalized text are rejected to reduce leakage.
You must also check for semantically equivalent queries and paraphrases yourself:
the tool cannot detect them.

`split.json` contains the complete normalized query inventory, partition
assignments, source-query CSV fingerprint, corpus fingerprint, seed, and requested
fraction. It is an exclusive-create snapshot: the CLI refuses to overwrite it.
Later commands use this snapshot, not a possibly edited source CSV.

**All tuning must use only `dev`.** Do not inspect held-out effectiveness or use
held-out self-consistency to select retrieval settings. Freeze your settings
before final held-out evaluation. These CLI tools do not implement a tuner:
candidate generation uses only the fixed Phase 1 default BM25 representation,
not configurable or automatically optimized settings. The free-form Phase 1
search CLI is not a guard against manually querying held-out text; respecting
the boundary remains part of the research protocol.

## 2. Prepare an ungraded candidate pool

```powershell
python label_cli.py pool --split-file judgment_data\split.json --partition dev -k 10 --output judgment_data\dev_pool.json
```

This collects baseline top-k records for every query in that partition, then
shuffles the query-record pairs with a recorded seed (default 42). Scores and
ranks are not written to the candidate queue or shown during labelling.
Candidate selection is not a relevance suggestion; no record gets a label
until you explicitly enter one.

For broader coverage, you may prepare an additional UTF-8 CSV with exactly the
header `query_id,show_id` and human-selected candidate pairs, without any grades.
Pass it with `--additional-pairs judgment_data\extra_dev_pairs.csv` when creating
the pool. Its pairs must belong to the chosen partition and existing corpus;
the union is de-duplicated and shuffled. Its source fingerprint is recorded.

**Pool limitations:** baseline top-k alone is not an exhaustive judgment list.
It can miss relevant records that every baseline result omits. Human-supplied
candidates improve coverage but do not establish completeness. Later recall and
nDCG results must describe the judged pool and this incompleteness; never
silently assume unjudged records are irrelevant. Further pooling across
predeclared configurations is not implemented here.

Pools cannot be overwritten by the CLI. Keep them unchanged after labelling
starts. Corpus, split, and pool fingerprints bind the grades to their inputs;
editing these files makes the old labels incompatible rather than silently
reassigning them.

## 3. Enter grades and resume safely

```powershell
python label_cli.py label --split-file judgment_data\split.json --pool judgment_data\dev_pool.json --labels judgment_data\judgments.csv
```

Each prompt shows the query text, stable IDs, and all cleaned CSV record fields.
It shows no retrieval score, rank, suggested grade, or previous grade. Submit
your own integer 0, 1, 2, or 3. Blank input, decimals, and other numbers are
rejected; there is no default. Enter `q`, send EOF, or press Ctrl+C at a grade
prompt to leave without grading that pair.

Each submitted grade is appended immediately, flushed, and synced to disk:

| CSV column | Meaning |
|---|---|
| query_id, show_id | Stable query-record identity |
| grade | Your explicitly entered integer 0-3 |
| timestamp | UTC ISO-8601 time of submission |
| pass_id | `initial` or a unique blind re-label pass ID |
| split_sha256, corpus_sha256, pool_sha256 | Input identity for that grade |

Use the same command to resume: already graded pairs are skipped without showing
their grades. The tool does not overwrite existing grades or allow duplicate
pair/pass entries. Corrupt logs, invalid grades, or changed inputs cause an
explicit error. A new log is not created until you enter the first valid grade.
Run **one writer at a time**, do not edit inputs during a session, and back up the
JSON/CSV files. These are local files, not a concurrent or hosted database.

The same workflow can prepare a separate pool with `--partition heldout` and
record its grades in the same judgment CSV. That does not change the frozen
split or authorize tuning on held-out queries.

## 4. Save a blind subset and return later

After completing the initial development pool, choose how many already labelled
pairs to re-label. The following illustrates a sample size of one only; choose
your study's size before examining agreement.

```powershell
python label_cli.py sample --split-file judgment_data\split.json --pool judgment_data\dev_pool.json --labels judgment_data\judgments.csv --size 1 --seed 43 --output judgment_data\dev_repeat.json
```

The command requires every pair in the original pool to have an initial grade.
It samples pairs uniformly without replacement, not queries or relevance
classes. It saves the selected order, a unique pass ID, and the original pool's
fingerprint. It stores **no previous grades** in the sample file. This is a
pair-level sample, so queries with more candidates can contribute more pairs.

Return after a delay you choose and record in your study notes:

```powershell
python label_cli.py label --split-file judgment_data\split.json --pool judgment_data\dev_repeat.json --labels judgment_data\judgments.csv
```

The CLI hides earlier grades, and it resumes the same saved subset after an
interruption instead of drawing a new sample. The original CSV still contains
your earlier grades; do not open it during the blind pass. A single-annotator
local tool cannot prevent you from reading your own files or remembering
previous decisions. There is no enforced waiting period. Keep each saved sample
for one pass; generating additional samples can bias reported agreement.

## 5. Compute self-consistency only after both passes are complete

```powershell
python label_cli.py consistency --split-file judgment_data\split.json --pool judgment_data\dev_repeat.json --labels judgment_data\judgments.csv
```

Missing labels are an error, not zero grades. The tool reports **no result**
until all sampled pairs have both initial and re-label grades. Grades are joined
by query/record IDs, never CSV row position, and each grade must match its
original pool fingerprint.

Exact agreement is the proportion of paired grades that are identical.
Quadratic-weighted Cohen's kappa treats the grades as ordinal: disagreement
weight is `(first - second)^2 / 9`; observed disagreement is its average over
paired grades; expected disagreement uses the product of the two empirical
marginal grade frequencies. Kappa is `1 - observed / expected`. This is a
from-definition implementation, with a separate scikit-learn cross-check test.
If both passes contain only the same single grade, expected disagreement is zero:
the JSON reports `null` for kappa with an explicit undefined note, not a fabricated
perfect kappa. A very small sample gives unstable agreement estimates.

Self-consistency is not retrieval effectiveness and is not inter-annotator
agreement. There is still only one annotator. No self-consistency number exists
for this project yet: **[RESULT: exact self-agreement]** and
**[RESULT: quadratic-weighted Cohen's kappa]**.

## Files and phase boundary

`query_sets.py` protects the frozen inventory; `judgments.py` handles queues,
human grade storage, and agreement; `label_cli.py` exposes the commands.
`tests/test_judgments.py` uses only artificial records and synthetic fixture
grades, including hand-computed agreement examples.

`judgment_data/` is gitignored to avoid accidentally committing live judgment
logs. Keep your own backups and explicitly version the finalized research
artifacts when ready; ignoring is not backup or access control.

Phase 3 [retrieval metric definitions](METRICS.md) are implemented independently
of these files. Real evaluation still requires your queries and complete human
judgments. Phase 4 failure assignments and Phase 5 final write-up are not
implemented. Missing research queries and labels must be supplied by you; the
tool never fills those gaps.
