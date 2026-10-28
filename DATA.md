# Retrieval data and index decisions (Phase 1)

## Source and scope

The checked-in `netflix_titles.csv` is the Netflix Movies and TV Shows dataset
linked from the existing README:
<https://www.kaggle.com/datasets/shivamb/netflix-shows>.
The retrieval corpus keeps **all 8,807 records: 6,131 movies and 2,676 TV shows**.
It is not a movies-only corpus, unlike the existing genre classifier. No genres,
languages, countries, years, or record types are filtered out.

The source CSV is never overwritten. `film_data.load_records()` returns cleaned
records and an audit report. `python retrieval.py --audit-only` prints that report,
the source path and SHA-256, and the index configuration as JSON. These counts
describe the actual checked-in data, not retrieval effectiveness results.

## Every automatic cleaning decision

1. Read with Python's CSV parser, UTF-8 with optional BOM, preserving quoted
   commas and embedded newlines. Require the existing 12-column schema and its
   order. Reject malformed records, an empty file/corpus, missing identifiers,
   or conflicting records with the same identifier rather than silently skipping.
   Missing cells are represented by empty strings, not `"nan"` or `"Unknown"`.
   Literal strings such as `"NA"` are not interpreted as missing values.
2. Normalize every cell to Unicode NFC, trim surrounding whitespace, and replace
   runs of Unicode whitespace (including embedded newlines) with one space.
   Preserve case, accents, punctuation, and non-Latin text in stored records.
   Do not transliterate, remove HTML, spell-correct, or infer missing content.
   Normalized-cell counts report changes before duplicate removal or metadata
   repair. On the source, whitespace changes affect 9 title, 7 cast, 88 date-added,
   and 16 description cells; NFC also changes one cell. The audit reports the
   combined count per field, without double-counting a cell changed by both.
3. If `duration` is empty and `rating` consists exactly of digits followed by
   `" min"` after normalization, move that value to `duration` and clear `rating`.
   This narrow repair applies to `s5542` (`74 min`), `s5795` (`84 min`), and
   `s5814` (`66 min`). Their actual maturity ratings remain unknown. No other
   metadata repairs, date parsing, numeric conversion, or duration conversion
   are performed. These fields are not searched by default.
4. Remove only identical complete records after normalization and the narrow
   duration repair; retain the first occurrence in CSV order. A conflicting
   duplicate `show_id` is an error. Never deduplicate by title or content alone:
   remakes and separately identified records must remain distinct. The checked-in
   source has no duplicate rows, duplicate IDs, or repeated raw titles; no rows
   are removed.
5. Keep records with missing searchable metadata; missing fields contribute no
   tokens. Do not invent directors, cast, countries, descriptions, or relevance
   labels. Preserve stable `show_id` values and input order for later judgments.

| Field | Missing before cleaning | Missing after cleaning |
|---|---:|---:|
| show_id | 0 | 0 |
| type | 0 | 0 |
| title | 0 | 0 |
| director | 2,634 | 2,634 |
| cast | 825 | 825 |
| country | 831 | 831 |
| date_added | 10 | 10 |
| release_year | 0 | 0 |
| rating | 4 | 7 |
| duration | 3 | 0 |
| listed_in | 0 | 0 |
| description | 0 | 0 |

## Index representation and ablations

`retrieval.py` builds an in-memory BM25Okapi index with `rank-bm25==0.2.2`.
It does not persist a binary index; rerunning the command rebuilds it from the
CSV and emits its configuration. Save stdout to retain a run's audit and scores.

Defaults mirror the legacy `search.py` representation: combine `title`,
`description`, `listed_in`, `cast`, `director`, and `country`; lowercase; split on
whitespace; no stemming or stop-word removal. Cleaning is now explicit, so this
is a representation baseline, not a claim of byte-for-byte identical raw input.
Comma-separated genres and names are not parsed into entities.

- `--fields` selects any nonempty, unique subset of the CSV fields. Excluded
  fields contribute nothing. Metadata fields such as year are opt-in.
- `--weight title=3` repeats title tokens three times before combining fields.
  Unspecified weights are one; weights must be positive integers. Exclude a field
  with `--fields`, not a zero weight. This is **token repetition, not BM25F or a
  weighted sum of field scores**: it changes term frequencies and document
  lengths together and remains subject to BM25 saturation. Interpret weighting
  ablations with this confound in mind.
- `--tokenizer whitespace` preserves attached punctuation. `--tokenizer words`
  extracts Unicode alphanumeric runs, splitting on punctuation and underscores:
  `sci-fi` becomes `sci`, `fi`; `O'Brien` becomes `o`, `brien` when lowercased.
  Accents and non-Latin letters remain; this is not language-aware segmentation.
- `--preserve-case` disables lowercasing. `--stem` enables NLTK's English Porter
  stemmer (default NLTK extensions); it needs no corpus downloads. It is not a
  multilingual stemmer and can harm names and non-English text. The same
  normalization, case handling, tokenizer, and stemmer apply to documents and
  queries. Query terms are never field-weighted.
- `--k1`, `--b`, and `--epsilon` expose the BM25Okapi defaults 1.5, 0.75, and
  0.25, respectively. `rank-bm25` floors negative IDF values using epsilon times
  average IDF; scores can still be negative for some corpora. Do not treat
  a score as a relevance grade or remove negative/zero scores implicitly.

Each retained row remains an indexed document, even if its selected fields are
empty. An entirely empty vocabulary is rejected explicitly. All documents are
ranked by descending score, with ties broken by original CSV order. Queries
with no indexed terms produce zero scores and a CLI notice; the returned order
is a tie-break, not evidence of relevance. `k` must be positive and results are
capped at corpus size.

## Reproduction

From the repository root, using the same Python environment:

```powershell
python -m pip install -r requirements.txt
python retrieval.py --audit-only
python retrieval.py --query "space adventure" -k 10
python retrieval.py --fields title description listed_in --tokenizer words --stem --weight title=3 --query "space adventure"
python -m pytest -q
```

Omit `--query` for an interactive prompt; submit an empty query or `q`, `quit`,
or `exit` to leave. The old `python search.py` remains available as a legacy
baseline and does not use this new cleaning/configuration pipeline.

## Research boundary

The thesis that failures arise primarily upstream of scoring is **untested**.
No judgments, metrics, tuning, failure assignments, or empirical effectiveness
claims are produced in Phase 1. The judgment list is not yet available; evaluation
must stop if required human labels are missing. Phase 2's [human judgment
tooling](LABELING.md) freezes a development/held-out query split before any tuning.
Configuration examples above illustrate mechanics only and are not recommended
or tuned settings.
