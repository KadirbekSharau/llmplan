# M2 implementation notes

Structure per docs/DEFINITION_OF_DONE.md section 6. Each entry names the implementation step
(1 to 5 of M2_DESIGN.md section 10) that introduced it.

## Implementation notes

- **Step 1 — Dependencies.** Runtime: `pandas>=3.0`, `numpy>=2.0` (section 11 allows both;
  lower bounds are the major versions tested, 3.0.6 and 2.4.6; pandas 3 changed string
  dtypes, so 2.x is not claimed). Dev: `pandas-stubs>=3.0`, needed for `mypy --strict`
  because pandas ships no inline types; it is a type-only package and not a runtime
  dependency.
- **Step 1 — `WorkloadFormatError` subclasses `ValidationError`** (CLI exit 2): a trace
  that does not match its format is bad user input, not catalog data. It sits after
  `ValidationError` in `errors.py`.
- **Step 1 — `model` and `tenant` use pandas' nullable `"string"` dtype with `pd.NA`.**
  pandas 3 also has a `str` dtype whose missing value is `NaN` and which compares equal to
  `"string"`; the `Workload` validator checks `na_value is pd.NA` so only the dtype named
  in the design is accepted.
- **Step 1 — `Workload` also requires a default `RangeIndex` and at least one row.** The
  design's "first value == 0.0" needs a first row, and positional access downstream should
  not depend on a parser's index.
- **Step 1 — The frame is not deep-frozen.** Pydantic freezes the model's fields, not the
  DataFrame's buffers; the docstring tells callers to treat it as read-only.
- **Step 1 — Row validation details (section 4.4).** Token values are coerced numerically;
  non-integral values (`1.5`) and counts above 2**31 - 1 are dropped as invalid, like NA and
  out-of-range values. The note is added when strictly more than 5% of rows are dropped and
  the error raised when strictly more than 50% are. Missing `model`/`tenant` values are not
  invalid (both columns are optional).
- **Step 1 — Parsing is chunked** (1,000,000 rows per chunk) so the 1.1 GB Azure 2024
  conversation trace never holds its timestamp strings in memory at once; each chunk is
  reduced to numpy arrays before the next is read. Datetimes are kept as int64 nanoseconds
  until the final `min` is subtracted, so relative seconds carry no float cancellation.
- **Step 1 — `read_csv(float_precision="round_trip")`.** pandas' default float parser is
  not round-trip exact (`0.30000000000000004` reads back as `0.3`), which would break the
  `synth` -> `load_workload` round trip.
- **Step 1 — Generic CSV `timestamp` values are parsed as ISO-8601 with `utc=True`**:
  values without an offset are taken as UTC, values with offsets are converted, so mixed
  files still give correct relative seconds.
- **Step 1 — `load_workload` gained `max_bytes` (default 2 GiB).** Section 4 makes the cap an
  argument of the parsers; ARCHITECTURE.md section 1 caps UI uploads at 200 MB, so M6 needs
  to pass its own value. A file above the cap, a missing file, or a directory raises
  `ValidationError` (exit 2).
- **Step 2 — Headers verified on 2026-09-30** by reading the dataset READMEs and the first
  bytes of each file. Azure 2023 (`AzureLLMInferenceTrace_code.csv`, `_conv.csv`) and Azure
  2024 (`AzureLLMInferenceTrace_code_1week.csv`, `_conv_1week.csv`) both have exactly
  `TIMESTAMP,ContextTokens,GeneratedTokens`. 2023 timestamps look like
  `2023-11-16 18:17:03.9799600` (naive, 7 fractional digits); 2024 timestamps look like
  `2024-05-10 00:00:00.009930+00:00` (UTC offset, 6 digits). BurstGPT release v2.0
  `BurstGPT_1.csv` has `Timestamp,Model,Request tokens,Response tokens,Total tokens,Log Type`
  with integer-second timestamps; the README says newer files add `Session ID` and
  `Elapsed time`, which the parser ignores (only its four used columns are read, and
  detection requires the six listed columns).
- **Step 2 — Telling azure2023 from azure2024.** The headers are identical, so `detect()`
  also reads the first data row: a `TIMESTAMP` ending in a UTC offset (`+00:00`, `Z`,
  `-0700`) is `azure2024`, anything else `azure2023`. A header-only Azure file is reported
  as ambiguous. Both keys share one parser; naive timestamps are read as UTC.
- **Step 2 — Detection requires exactly one match**; a header that satisfies two formats
  (e.g. BurstGPT columns plus generic CSV columns) raises `WorkloadFormatError` listing both.
  Column names are matched exactly (only a UTF-8 BOM is stripped).
- **Step 2 — BurstGPT failed rows.** In the real `BurstGPT_1.csv` (1,429,737 rows), 25,443
  rows have `Response tokens == 0` and 25,427 of those also have `Request tokens == 0`.
  Section 4.4 drops `input_tokens < 1`, so those 25,427 are dropped (1.8%, below the 5% note
  threshold) and 16 zero-output rows are kept and reported in the `zero_output_rows` note.
  `Workload` requires `input_tokens >= 1`, so keeping them would need an invented value;
  a zero-token failed request also costs no prefill or decode work.
- **Step 2 — `Log Type` is not mapped to `tenant`**; the design maps only `Model`.
- **Step 2 — Fixtures** (`tests/fixtures/workload_{csv,azure2023,azure2024,burstgpt}_50.csv`)
  were generated with a seeded script using invented values in each verified layout; no row
  is copied from a dataset.
- **Step 2 — Real-trace check (not part of the test suite, no data committed).** Parsing the
  downloaded files: Azure 2023 code 8,819 rows and conversation 19,366 rows (under 0.1 s
  each, 0 dropped); BurstGPT_1 1,404,310 rows kept in 1.1 s; Azure 2024 code one-week
  16,803,695 rows in 38 s with 1.6 GB peak RSS, duration 604,799.9 s, `hourly_rps` over 7
  full days. Nearly all of the 2024 time is pandas' ISO-8601 parsing (about 2 s per million
  rows); M6's 60-second preset budget should account for it. The one-week conversation
  file (27,303,999 rows) took 68.5 s with 2.4 GB peak RSS, so the M6 Azure 2024 preset will
  need the code file, a cached parse, or a faster timestamp path.
- **Step 3 — `diurnal` is a relative shape; `rate_rps` is the peak-hour rate.** Section 6
  says multipliers are "applied to `rate_rps`" and arrivals are thinned with probability
  `mult / max(mult)` "after generating at the peak rate". Taking the peak rate as
  `rate_rps * max(mult)` would make test 9.4's "all-equal multipliers leave the count
  unchanged" true only for multipliers equal to 1; generating at `rate_rps` and thinning by
  `mult / max(mult)` makes it true for any constant vector, so that reading was chosen. The
  hour of day of an arrival is `floor(arrival_s / 3600) % 24`.
- **Step 3 — The arrival at 0.0 is always kept under diurnal thinning.** Section 6 fixes the
  first arrival at 0.0 and `Workload` requires it; thinning it away and re-normalizing would
  shift every hour bucket. If `diurnal[0] == 0`, hour 0 therefore holds that single request.
- **Step 3 — Arrivals are those at or before `duration_s`** (the first exponential arrival
  past `duration_s` ends the process and is discarded).
- **Step 3 — Test 9.4's zero-hour vector is zero for hours 6 to 17.** Hour 0 must be
  non-zero (the kept first arrival) and hour 23 must be non-zero, because a trace's duration
  ends at its last arrival: a silent last hour would make the trace shorter than one day
  and `hourly_rps` None. The design does not say which 12 hours.
- **Step 3 — Generator guards:** `seed >= 0` (numpy rejects negative seeds),
  `rate_rps * duration_s <= 50,000,000` expected requests (memory guard for CLI input), and
  the input distribution must have `lo >= 1` (a `Workload` needs `input_tokens >= 1`); each
  raises `ValidationError`. `parse_distribution()` (CLI syntax of section 8) lives in
  `synth.py` so the M6 UI can reuse it.
- **Step 4 — Manifest values.** All five files were downloaded once on 2026-09-30 and
  hashed locally; the Azure 2024 and BurstGPT hashes equal the SHA-256 digests GitHub
  publishes for those release assets. Azure 2023 files live in the repository tree, not in a
  release, so their URLs are pinned to commit `790921015d50dd6aae7f7e47f39ba0e235ad6b08`
  (the last commit touching them) and were re-hashed at that commit. BurstGPT uses the
  release v2.0 asset (52,283,111 bytes), not the copy in the repo's `data/` directory, which
  has a different size (50,853,373 bytes). No entry needed `url: null`. Licenses: both
  datasets are CC-BY-4.0 (GitHub license API). No manifest value is null.
- **Step 4 — Fetch behavior.** Consent is checked in the library (`fetch_trace(..., yes=)`)
  before the manifest is read or any request is built, so the CLI stays a thin adapter and
  exits 2 via `ValidationError`. The temporary `.part` file is created inside `--dest` so the
  final rename is atomic, and it is removed in a `finally` on every failure path. The stream
  stops as soon as it passes the smaller of 2 GiB and the manifest `size_bytes` (a longer
  body cannot match). Every hop, including redirects (GitHub release downloads redirect to
  a CDN host), must be https; hosts are not allow-listed because the SHA-256 check is the
  integrity guarantee. An existing target file is reused if its checksum matches and refused
  otherwise, never overwritten. `--dest` must already exist. Unknown names raise
  `ValidationError` (exit 2) listing the known names.
- **Step 4 — `data/traces/.gitignore`** ignores everything but the manifest, so a fetch with
  `--dest data/traces` cannot be committed by accident.
- **Step 4 — CLI module.** `llmplan/cli_workload.py` holds the M2 sub-apps (as agreed for the
  parallel M3 branch); `llmplan/cli.py` gains one import line and one `add_typer` line per
  sub-app. `cli_workload` reaches `llmplan.cli._run` through a function-level import,
  because `llmplan.cli` imports `cli_workload` at module level.
- **Step 4 — Not verified against the live endpoints with the fetch command itself**
  (network use was limited to one download per file); the HTTP path is covered with
  `httpx.MockTransport`, and the pinned URLs were checked when hashing.

## Deviations from the design doc

- **Step 1 — `hourly_rps` is present when the trace covers 24 hour windows, not when
  `duration_s >= 86_400`.** Section 9.4 requires `hourly_rps` for a synthetic run with
  `duration_s=86_400`, but a Poisson trace generated over `[0, 86_400]` ends a fraction of a
  second before 86,400 s (the next arrival would fall past the end), so `duration_s` is
  always slightly below 86,400 and the literal rule would return None. The same happens for
  the real Azure 2024 one-week files, which end at 23:59:59.9 on their seventh day. Hours
  now follow the window convention of section 3: the trace covers
  `floor(duration_s / 3600) + 1` hour windows (the last may be partial), `full_days` is that
  count // 24, `hourly_rps[h]` is the count of arrivals in hour-of-day `h` within those full
  days divided by `full_days * 3600`, and arrivals in a trailing partial day are not
  bucketed. Consequence: a trace between 23 and 24 hours long gets an `hourly_rps` whose
  hour 23 is under-counted, exactly like the partial last rate window.

- **Step 3 — Test 9.6's `peak_window_rps >= mean_rps` is false; the test asserts the true
  bound instead.** With section 3's windows the last window is partial but divided by the
  full `window_s`, while `mean_rps` divides by `duration_s`. Section 9.2's own expected values
  break it: `peak_window_rps = 6 / 60 = 0.1 < mean_rps = 10 / 90 = 0.111`. More generally,
  with `n_windows = floor(duration_s / window_s) + 1`, the counts sum to `n_requests` over
  a span of `n_windows * window_s > duration_s`, so only
  `max_count >= n_requests / n_windows` is guaranteed (e.g. two requests 10 s apart with
  60 s windows: peak `2/60`, mean `2/10`). Test 9.6 asserts
  `round(peak_window_rps * window_s) * n_windows >= n_requests`, i.e. the peak window rate
  is at least the mean rate over the windows the trace spans. The other two properties are
  unchanged. No expected value of 9.2 was changed.

## Questions for founder

None.
