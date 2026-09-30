# M2 Design — Workload Ingestion and Characterization

Status: ready for implementation once M1 is merged. Assignee: developer agent.
Prerequisites: docs/PLAN.md, docs/ARCHITECTURE.md, docs/DEFINITION_OF_DONE.md, and a
merged M1 on `main`. Branch: `m2-workload`.

Definition of done: DEFINITION_OF_DONE.md plus every test in section 9.

---

## 1. Goal

Turn a request trace (public dataset, user CSV, or synthetic) into a validated `Workload`
and a `WorkloadStats` summary that later milestones size against: peak request rate, token
length distributions, peak token rates, and a diurnal profile.

## 2. Deliverables

1. `llmplan/workload/schema.py`: `Workload`, `WorkloadStats`, `Distribution`.
2. `llmplan/workload/formats/`: registry with `csv`, `azure2023`, `azure2024`, `burstgpt`.
3. `llmplan/workload/stats.py`: `compute_stats(workload, window_s) -> WorkloadStats`.
4. `llmplan/workload/synth.py`: deterministic synthetic generator.
5. `llmplan/workload/__init__.py`: `load_workload(source, *, format=None) -> Workload`.
6. `llmplan/workload/fetch.py` + CLI `llmplan traces fetch`: consented, checksum-verified
   download of the public traces into a user-chosen directory.
7. CLI: `llmplan workload stats`, `llmplan workload synth`, `llmplan traces fetch`.
8. Fixtures: a 10-row CSV with hand-computed stats (section 9.2), and a 50-row sample in
   each public format's column layout (hand-made, not copied).
9. Tests, README usage, CHANGELOG, `M2_NOTES.md`.

## 3. Data models

```python
class Workload(BaseModel, frozen=True, arbitrary_types_allowed=True):
    source: str                      # path or "synthetic:<seed>"
    format: str                      # registry key
    frame: pd.DataFrame              # validated columns below, sorted by arrival_s
    dropped_rows: int                # rows removed during parsing (see 4.4)
    notes: tuple[str, ...]

# frame columns (exact names and dtypes, validated in a model_validator):
#   arrival_s: float64, >= 0, non-decreasing, first value == 0.0
#   input_tokens: int64, >= 1
#   output_tokens: int64, >= 0
#   model: string (pandas "string" dtype), may be all <NA>
#   tenant: string, may be all <NA>

class Distribution(BaseModel, frozen=True):
    kind: Literal["fixed", "lognormal", "uniform"]
    # fixed: value; lognormal: mean, sigma (of the underlying normal), clipped to [lo, hi];
    # uniform: lo, hi (inclusive integers)
    value: int | None = None
    mean: float | None = None
    sigma: float | None = None
    lo: int = 1
    hi: int = 131_072

class WorkloadStats(BaseModel, frozen=True):
    n_requests: int
    duration_s: float                # max(arrival_s) - min(arrival_s)
    window_s: float
    n_windows: int
    mean_rps: float                  # n_requests / duration_s (0.0 if duration_s == 0)
    peak_window_rps: float           # max over windows of count / window_s
    peak_window_index: int
    input_tokens_p50: float
    input_tokens_p95: float
    input_tokens_p99: float
    input_tokens_mean: float
    input_tokens_max: int
    output_tokens_p50: float
    output_tokens_p95: float
    output_tokens_p99: float
    output_tokens_mean: float
    output_tokens_max: int
    peak_input_tokens_per_s: float   # max over windows of sum(input_tokens) / window_s
    peak_output_tokens_per_s: float
    hourly_rps: tuple[float, ...] | None   # 24 entries when duration_s >= 86_400, else None
```

Percentiles use `numpy.percentile` with the default `"linear"` method. Windows are
half-open `[k*window_s, (k+1)*window_s)` starting at `min(arrival_s)`; the last partial
window is included and its rate is divided by the full `window_s` (conservative: it
under-counts, which is fine for peak detection; note this in the docstring).

## 4. Formats

Registry in `llmplan/workload/formats/__init__.py`: `register(key)`, `get(key)`,
`detect(path) -> str` (by header inspection; raise `WorkloadFormatError` if ambiguous).
Each format module implements `parse(path: Path) -> Workload`.

All parsers: read with `pandas.read_csv(usecols=..., dtype=...)`; reject files above
`max_bytes` (default 2 GiB, argument); normalize `arrival_s` so the first request is 0.0;
sort by `arrival_s` (stable); drop and count invalid rows per 4.4.

### 4.1 `csv` (generic)
Columns: `arrival_s` (float seconds) **or** `timestamp` (ISO-8601, converted to seconds
from the first row); `input_tokens`; `output_tokens`; optional `model`, `tenant`.
Both `arrival_s` and `timestamp` present -> `WorkloadFormatError`.

### 4.2 `azure2023` and `azure2024`
Microsoft Azure LLM inference traces (repo: https://github.com/Azure/AzurePublicDataset,
files `AzureLLMInferenceTrace_code.csv`, `AzureLLMInferenceTrace_conv.csv` for 2023 and the
`_1week` variants for 2024). Columns: `TIMESTAMP` (datetime), `ContextTokens` (input),
`GeneratedTokens` (output). Verify the exact header against the dataset README and record
it in `M2_NOTES.md`; the 50-row fixture uses the verified header. Both keys share one
parser parameterized by key.

### 4.3 `burstgpt`
BurstGPT (repo: https://github.com/hpmll/burstgpt). Columns: `Timestamp` (seconds offset),
`Model`, `Request tokens`, `Response tokens`, `Total tokens`, `Log Type`. Map `Model` to
`model`. Verify the header the same way. Rows with `Response tokens == 0` are kept (they
are real failed or empty responses) but counted in a `zero_output_rows` note.

### 4.4 Row validation (all formats)
Drop, and count in `dropped_rows`, any row where: `input_tokens < 1`, `output_tokens < 0`,
timestamp unparsable, or any required column is NA. If more than 5% of rows are dropped,
add a note; if more than 50%, raise `WorkloadFormatError` (the file is probably the wrong
format).

## 5. Statistics

`compute_stats(workload: Workload, window_s: float = 60.0) -> WorkloadStats` per section 3.
`window_s` must be > 0. `hourly_rps`: bucket by `floor(arrival_s / 3600) % 24`, mean rps
per bucket over the number of full days covered; only when `duration_s >= 86_400`.

## 6. Synthetic generator

```python
def generate(*, rate_rps: float, duration_s: float, input_tokens: Distribution,
             output_tokens: Distribution, seed: int,
             diurnal: tuple[float, ...] | None = None) -> Workload
```
- Poisson process: inter-arrival times `numpy.random.default_rng(seed).exponential(1/rate)`
  until `duration_s` is exceeded; first arrival at 0.0.
- `diurnal`: optional 24 multipliers applied to `rate_rps` by hour of day (thinning: accept
  each arrival with probability `mult/ max(mult)` after generating at the peak rate).
- Token lengths drawn from the `Distribution` with the same generator, rounded, clipped.
- `source = f"synthetic:{seed}"`, `format = "synthetic"`. Byte-identical output for the same
  arguments.

## 7. Public trace fetch

`data/traces/manifest.yaml`: entries `{name, url, sha256, size_bytes, as_of, license_url}`.
On first implementation, the agent downloads each file once, records the SHA-256 and size
it observed, and commits the manifest (not the data). `llmplan traces fetch <name> --dest DIR
--yes` refuses to run without `--yes`, streams to a temp file with a 2 GiB cap, verifies
the SHA-256, then moves it into place. Mismatch -> `FetchError` and the temp file is deleted.
Names: `azure2023-code`, `azure2023-conv`, `azure2024-code`, `azure2024-conv`, `burstgpt-1`
(verify actual file names and URLs; if the dataset is hosted behind a page that forbids
direct download, record that in the notes and leave the entry with `url: null`).

## 8. CLI

```
llmplan workload stats --trace PATH [--format auto|csv|azure2023|azure2024|burstgpt]
                       [--window 60] [--format-out text|json]
llmplan workload synth --rps 5 --duration 3600 --in-tokens lognormal:6.2:0.8
                       --out-tokens lognormal:5.5:0.9 --seed 1 --out PATH.csv
llmplan traces fetch NAME --dest DIR --yes
```
Distribution CLI syntax: `fixed:N`, `lognormal:MEAN:SIGMA[:LO:HI]`, `uniform:LO:HI`.
`synth` writes the generic `csv` format so it round-trips through `load_workload`.

## 9. Acceptance tests (`tests/acceptance/test_m2.py`)

**9.1 Fixture round-trips.** Each of the four format fixtures (50 rows) loads, has the exact
column set and dtypes, `arrival_s[0] == 0.0`, is sorted, and `dropped_rows == 0`. `detect()`
returns the right key for each.

**9.2 Hand-computed stats on `tests/fixtures/workload_10.csv`:**
```
arrival_s:     0,10,20,30,40,50,60,70,80,90
input_tokens:  100,200,300,400,500,600,700,800,900,1000
output_tokens: 10,20,30,40,50,60,70,80,90,100
```
With `window_s=60`: `n_requests 10`, `duration_s 90.0`, `n_windows 2`, `mean_rps 10/90`,
`peak_window_rps 0.1` (6 requests in window 0), `peak_window_index 0`,
`input_tokens_p50 550.0`, `input_tokens_p95 955.0`, `input_tokens_p99 991.0`,
`input_tokens_mean 550.0`, `input_tokens_max 1000`, `output_tokens_p50 55.0`,
`output_tokens_p95 95.5`, `output_tokens_p99 99.1`, `peak_input_tokens_per_s 3400/60`
(window 1: 700+800+900+1000), `peak_output_tokens_per_s 340/60`, `hourly_rps None`.
Compare floats with `pytest.approx(rel=1e-9)`.

**9.3 Validation.** A CSV with both `arrival_s` and `timestamp` raises `WorkloadFormatError`.
A CSV where 60% of rows have `input_tokens=0` raises `WorkloadFormatError`; one where 10% do
loads with `dropped_rows` equal to that count and a note present. `window_s=0` raises
`ValidationError`.

**9.4 Synthetic.** `generate(rate_rps=5, duration_s=3600, seed=1, ...)` twice gives
byte-identical CSV output; `n_requests` within 5% of 18,000; `arrival_s[0] == 0.0`; token
columns respect `[lo, hi]`. With `diurnal` all-equal multipliers the count is unchanged
(within the same 5%) and with a multiplier vector that is zero for 12 hours over a
`duration_s=86_400` run, `hourly_rps` has zeros in exactly those buckets.

**9.5 Fetch safety.** `traces fetch` without `--yes` exits 2 and performs no request
(stubbed transport). A stubbed download whose bytes do not match the manifest SHA-256 raises
`FetchError` and leaves no file in `--dest`.

**9.6 Property tests.** For any generated workload: `arrival_s` non-decreasing;
`peak_window_rps >= mean_rps`; `input_tokens_p99 >= input_tokens_p95 >= input_tokens_p50`.

## 10. Implementation order

1. schema + generic csv parser + 10-row fixture + 9.2 stats test.
2. Public format parsers + 50-row fixtures + detect + 9.1, 9.3.
3. Synthetic generator + 9.4, 9.6.
4. Fetch + manifest + 9.5.
5. CLI + README + CHANGELOG + notes.

## 11. Dependencies allowed
Runtime adds: `pandas`, `numpy`. Nothing else.

## 12. Questions for founder (agent appends to M2_NOTES.md as needed)
- None known at design time.
