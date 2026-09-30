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

## Questions for founder

None.
