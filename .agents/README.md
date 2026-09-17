# Agent memory

This directory is the durable project memory for future work in this folder.

## Reading order

1. Read `MEMORY.md` for the current cross-version picture and active queue.
2. Read the relevant file under `versions/` before analyzing or changing a version.
3. Verify important claims against the named archive artifact; these notes are an index, not a replacement for primary evidence.

## Maintenance rules

- Keep observations version-scoped. Do not silently project a later result backward.
- Separate frozen/test results from development-selection results.
- Preserve negative findings, caveats, and protocol amendments.
- Record archive SHA-256 values so replacement archives are detectable.
- Update `MEMORY.md` after each version review, and keep detailed evidence in `versions/vNN.md`.
- Do not extract large data/model payloads into the repository unless a task needs them.

