Hand-written M8 fixtures (no value is copied from a real run):

- `benchmarks_3_rows.csv`: llmplan benchmark CSV (the `BenchmarkRow` columns without
  `source_url` and `as_of`); row 1 claims an impossible 50,000 output tokens/s at
  concurrency 32 on one H100.
- `vllm_bench_serve.json`: one result object in the layout `vllm bench serve --save-result`
  writes without `--save-detailed`, field names checked against vLLM v0.30.0
  (`vllm/benchmarks/serve.py`); the default `--metric-percentiles 99`, so no `p95_*` keys.
