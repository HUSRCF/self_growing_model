# Viability-only witness path — 2026-09-17

Previous turn produced a verified probe/cache checkpoint. This experiment
tests a pure implementation optimization, not a new controller or scorer.

`fast_probe=True` calls `_expand(..., viability_only=True)` only for shallow
commit probes. Event read, event/q pair order, batched F input, exact F-cache
key, checker inputs, admissibility and full fallback are unchanged. The
first retained q is returned as a singleton witness instead of constructing
all child histories, diagnosing q membership, accumulating unused scores,
and allocating child Nodes/hidden copies. No probe RNG is introduced.

The implementation intentionally retains batched candidate F/checking. An
edge-at-a-time early exit would change cache keys and numeric batch shapes;
it might lose vectorization benefits and needs a separate experiment.
Normal root and lookahead expansion still returns fully scored Nodes.
Original defaults remain unchanged; generic CLI requires `--commit-probe`
with `--fast-probe`; evaluator `--mode probe_fast` also enables F cache4096.

35 tests pass, including first-witness agreement with full expansion under
hard/soft checking, no unused state diagnosis, widening with witness-type
preservation, numeric-invalid exhaustion, and all prior search/learning
tests. Four models (original + all three TRAIN-selected stateless adapters),
three rollout seeds,32DEV windows,8particles,301 internal/300 scored steps.
Summary checks prediction/failure/q/boundary bitwise versus uncached probe,
and truth/video/window alignment. No TEST use or fitting this round.

`benchmark_fast_probe.py` additionally alternates old/new execution order
over two repeats of fixed windows0/8/16/24 (one per DEV video), original
policy, seed1729. It requires exact returned arrays, checker audit and all
search counters except timing, not just equal headline errors. CPU times
include independent post-rollout checker verification. The repeats are a
timing diagnostic, not independent generalization evidence.

## Paired timing outcome

Full32DEV evaluation completed: all four models × three rollout seeds
are bitwise identical in predictions/failures/q/terminal boundary; every
checker audit and non-timing search counter matches the cached reference.
Original-policy CPU158.926→157.524s/run (0.88%); adapter-family
176.989→172.276s/run (2.66%). Original0/768 and adapters0/2304 failures,
with unchanged errors. Historical-run timing comparisons are descriptive;
the alternating paired control below gives the more local speed estimate.

Eight paired measurements total40.289s cached vs39.647s fast, a1.59% CPU
reduction. Seven of eight pairs favor fast, but one reverses. All arrays,
checks and search counters match exactly. This is a small optimization,
not evidence of a major bottleneck removal or a new predictive improvement.
The earlier profile's event-read/checker/F costs remain the main targets;
do not infer that general Python allocation is the dominant cost from this.

Next priority should not be a long sequence of tiny Node optimizations.
A pure event-read cache needs full history/q/hidden in its key and frozen
adapter identity; a checker cache needs its actual continuous inputs and
immutable calibration. Any new cache must demonstrate end-to-end exactness
and measured CPU benefit. For model quality, retained hypotheses are
causal value feedback aligned with actual deployed continuations and a
learned persistent residual distribution; present speed-only evidence
neither supports nor refutes those hypotheses. No default-policy promotion.

Reproduce:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python evaluate_adapted_checked.py --mode probe_fast --model-seed 0 --workers 3
# Repeat with model seeds1901,2718,3141.
python summarize_commit_probe.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python benchmark_fast_probe.py
```
