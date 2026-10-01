# Security policy

## Reporting a vulnerability

Please report security problems privately, not in a public issue:

- email the maintainer, Kadirbek Sharau, at sharaukadr2001@gmail.com, or
- use GitHub's private vulnerability reporting:
  https://github.com/KadirbekSharau/llmplan/security/advisories/new

Include what you found, how to reproduce it, and the version (`pip show llmplan`, or the
commit). You will get a reply within a week. Please give us a reasonable time to fix the
problem before you disclose it. There is no bug bounty.

## Supported versions

Only the latest release (and `main`) receives fixes.

## What llmplan does and does not do

llmplan is an offline planner: it never connects to a cluster, a GPU or an inference
server. Its only network access is fetching a model's `config.json` from
`https://huggingface.co` (validated repository ids, https only, 1 MiB cap), and
`llmplan traces fetch`, which downloads only with `--yes` and verifies a recorded SHA-256.
A Hugging Face token is read from the `HF_TOKEN` environment variable only and is never
logged or written. Uploaded traces and benchmark files in the web UI are size-capped,
parsed with a fixed schema in memory, never written to disk and never logged.
