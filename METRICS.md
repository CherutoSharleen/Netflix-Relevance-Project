# Retrieval metric definitions (Phase 3)

`metrics.py` implements precision@k, recall@k, reciprocal rank, mean reciprocal
rank (MRR), and nDCG@k using only the Python standard library. The formulas are
visible in small functions; no IR metrics library is used by the implementation.
The independent cross-check is in `tests/test_metrics_library.py`.

These are measurement tools, **not project results**. Human queries and
judgments are still missing. No evaluation, tuning, or failure classification
has been performed.

## Input contract

Per-query functions take an ordered sequence of unique document IDs and a
nonempty mapping from document IDs to your integer grades 0-3. Use stable
`show_id` values when working with the real corpus.

- Every document in the **entire supplied ranking** must have a human judgment,
  even if it is beyond the requested cutoff. Missing grades raise an error:
  they are never filled with zero, omitted, or guessed.
- Judgments may contain additional documents not retrieved. Those records count
  toward recall's denominator and the ideal nDCG ranking.
- Duplicate returned IDs, blank IDs, invalid grades (including booleans and
  floats), and nonpositive/noninteger cutoffs are rejected.
- An explicitly empty ranking is a valid run with no returned hits. An empty
  judgment mapping is missing evidence and is rejected, not a zero result.
- Binary metrics require an explicit keyword argument `relevant_grade`, which
  must be 1, 2, or 3. A grade meets the threshold inclusively. There is no default
  interpretation of your scale; choose and document it before evaluating, and
  keep it fixed across comparisons.

## Definitions and conventions

### Precision@k

`precision_at_k(ranking, judgments, k, relevant_grade=...)`

Count returned documents at positions 1 through k whose grades meet the chosen
threshold, then divide by **k**. If fewer than k documents are returned, the
denominator is still k, not the shorter run length. Unfilled positions are not
retrieved hits; this does not assign grades to any unjudged document.

### Recall@k

`recall_at_k(ranking, judgments, k, relevant_grade=...)`

Divide the same top-k relevant-hit count by the total number of documents
meeting the threshold in the supplied judgment list, including relevant
documents that were not retrieved.

This is **recall relative to the human judgment list**, not a claim of exhaustive
recall over all 8,807 records. An incomplete list can omit relevant records.
The tool cannot discover that missing evidence automatically.

If no judged document meets the threshold, recall's denominator is zero. The
function raises an explicit undefined-metric error rather than reporting zero,
one, or silently excluding the query. A later evaluation protocol must decide
how to report such cases transparently.

### Reciprocal rank and MRR

`reciprocal_rank(ranking, judgments, relevant_grade=...)`

Find the first returned document meeting the threshold. If its one-based
position is r, reciprocal rank is `1 / r`. If no relevant document appears in
the supplied ranking, reciprocal rank is zero.

`mean_reciprocal_rank(rankings, judgments, relevant_grade=...)`

MRR is the arithmetic mean of reciprocal ranks across queries, with each query
weighted equally regardless of its run length or judgment-list size. It uses
the entire supplied ranking; it is not silently truncated to the k used by
other metrics. A deliberately truncated input run limits how far a relevant
hit can be found and must be described in the study.

The two query-ID mappings must contain exactly the same nonempty query set.
A missing query, missing query judgment list, or invalid constituent run is an
error, never an omitted query. An explicit empty run still contributes zero to
the mean. Recall's undefined-denominator policy does not affect reciprocal
rank: no relevant hit means zero even if all supplied judgments are grade zero.

### nDCG@k

`ndcg_at_k(ranking, judgments, k)`

This implementation uses **linear gain**, equal to the original relevance
grade, and base-2 logarithmic discount. It does **not** use exponential gain
`2^grade - 1`. Linear gain matches the `ir_measures`/trec_eval nDCG convention
used by the separate cross-check; report this choice with study results.

For each returned position r from 1 to k, add `grade / log2(r + 1)` to obtain
DCG. Sort **all supplied judgments** by descending grade, take the best k, and
apply the same discounted sum to obtain ideal DCG. Divide DCG by ideal DCG.

The top-ranked document has no discount because `log2(2)` is one. Binary
thresholds are not applied to nDCG: it retains the differences between grades
1, 2, and 3. Relevant records missing from the run can still improve ideal DCG
and therefore lower the run's normalized score.

If every judged gain is zero, ideal DCG is zero and nDCG is undefined. The
function raises explicitly. If ideal DCG is positive but the run has no positive
gain, nDCG is zero. A perfect judged ordering has nDCG one.

## Verification and reproduction

The definition tests use artificial IDs and synthetic grades, never labels for
the Netflix corpus. Their comments show hand calculations for precision, recall,
MRR, and discounted graded gain, including a relevant record missed by the run.
They also check short/empty rankings, missing labels, malformed grades, duplicate
documents, and undefined cases.

Run the hand-worked tests without an IR library:

```powershell
python -m pip install -r requirements.txt
python -m pytest -q tests\test_metrics.py
```

For the independent cross-check and the complete suite:

```powershell
python -m pip install -r requirements-test.txt
python -m pytest -q tests\test_metrics_library.py
python -m pytest -q
```

`requirements-test.txt` includes the existing requirements and pins
`ir-measures==0.4.3`. This additional package is only imported by the separate
cross-check test, not the search, labelling, or metric implementation modules.
No new dependency is required to deploy or run the existing apps.

The cross-check compares all four metrics (including per-query reciprocal rank
and aggregate MRR) at multiple cutoffs and binary thresholds. Its artificial
runs use unique descending scores to preserve the intended order. Every returned
record has a grade, so the library's treatment of unjudged records cannot mask
our stricter missing-label policy. Undefined recall/nDCG cases are tested
separately because libraries may adopt a different zero-denominator convention.

## Using the real judgment protocol

Use the frozen partition from [LABELING.md](LABELING.md) to select queries.
Only original human `initial` grades belong in a retrieval judgment mapping;
re-label passes are for measuring consistency, not extra judgments or a way to
choose whichever grade improves a metric. An evaluation adapter must preserve
the Phase 2 corpus/split/pool fingerprint checks before constructing mappings.

These pure functions do not load or choose research data, merge label passes,
enforce file fingerprints, select a query partition, or run tuning. They must
not be used to bypass those protocol checks. The Phase 4 [evaluation/review
workflow](ANALYSIS.md) connects rankings with saved human judgments, validates
input fingerprints, and reports MRR at the supplied retrieval depth.

All tuning remains development-only. Do not mix development and held-out queries
in a reported average. Final results remain placeholders:
**[RESULT: precision@10 dev]**, **[RESULT: recall@10 dev]**,
**[RESULT: MRR dev]**, and **[RESULT: nDCG@10 dev]**.
