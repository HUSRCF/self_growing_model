# Pre-commit successor viability probe — 2026-09-17

Previous goal turn made progress: stateless adapter/checker integration,
strict endpoint validation, and exact sparse-failure replay. Selected-window
rescues by every-step depth2 motivated this full-window experiment.

## Mechanism

`AdaptiveBeam(commit_probe=True)` affects scheduled shallow decisions:

1. Sample a root under the original shallow score and matched temperature.
2. Expand ONLY that root to seek an admissible successor, preserving the
   event→q→F virtual-edge order. If the narrow set is empty, enumerate all.
3. If none exists, discard all same-q root event edges locally and resample
   the remaining roots. Equal-q equivalence holds only for the original or
   stateless no-feedback controller used here. Parent memory is untouched.
4. If a successor exists, commit the original root, not the successor.
   Its score never reranks roots and never changes the scoring temperature.

Scheduled depth2 calls already have a successor and do not probe again.
Audit counters record probes/full fallbacks/rejected event-root edges;
`scoring_depth` distinguishes shallow ranking from actual probe depth.
This is one-step viability, not a global feasibility guarantee.

## Comparison protocol

Sparse period5 depth2 versus period5 plus commit probe versus every-step
depth2. Same event_top2/next_top2/beam3, temperature1 with depth-invariant
scale matching, frozen checker/F, shared writer, rollback budget300 and
two-step revision limit.301 internal outputs,300 returned, every valid
point including endpoint independently checked. Same32DEV windows and
3 rollout seeds; original policy plus all3 TRAIN-selected adapter models.

All modes use the same initial windows and seed conventions, but rejection
and resampling can change later RNG consumption. Full depth2 includes
future scores; the probe deliberately only filters viability. Compare
those distinct mechanisms, not just a scalar depth budget. CPU sums, not
parallel wall-clock times, are used for compute comparisons.

33 tests pass: local duplicate-q rejection/resampling; full fallback and
exhaustion; unchanged shallow temperature despite second-edge probing;
no ranking by future scores; no extra probe when already deep; all previous
29 adapter/prefix/checker tests. Defaults and frozen package remain unchanged.

## Initial full-benchmark outcome

| Adapter family mode | .5s RMSE | 1s RMSE | 3s RMSE | Failed endpoints | CPU seconds/run |
|---|---|---|---|---|---|
|Sparse period5|.040647|.142315|.387230|9/2304|145.984|
|Commit probe|.034029|.138854|.387192|0/2304|205.631|
|Full depth2 every step|.033657|.121471|.389235|1/2304|369.405|

Probe eliminates the previously observed failures on this full DEV family,
but is not a global feasibility guarantee. Full depth2's single failure
also shows that selected-window recovery from the previous audit did not
guarantee full-benchmark completion. Probe costs~44% less CPU than full
depth2 but~41% more than sparse without probe. These are measured CPU sums,
not a real-time deadline promise. Full depth2 gives substantially better1s
error; probe's slightly better3s does not make it uniformly superior.

Original frozen-policy controls: probe .035507/.130582/.388095,
CPU188.308s,0/768 failures; full .033114/.119322/.392053,CPU331.901s,0/768.
Compared with the matching frozen probe, adapter improves .5s/3s but worsens
1s; no claim of a uniformly better learned controller. All valid returned
points pass unchanged central checks, including endpoint.

Probe turns many delayed failures into pre-commit rejection: frozen-policy
mean rollbacks2321→56; adapter means roughly2305→59. Some trajectories are
unchanged because pre-commit rejection can mirror a later commit-and-pop.
This is partly moving constraint work earlier, not solely increasing search
depth. The two-step revision budget still measures the runner's written
frontier; it is not an actual streaming-publication or wall-clock budget.

## Exact-cache follow-up protocol

`probe_cache` uses the already tested F LRU cache (capacity4096), keyed by
the complete history bytes, q and candidate-r tuple. F is immutable; hidden
state is intentionally excluded because F does not depend on it. Neither
event probabilities nor mutable controller state are cached. Probed F
outputs can be reused on the subsequent actual step; the full rollout is
repeated for every model/seed and checked bitwise against uncached probe.
All checker and policy operations remain active. No new approximate writer.

Completed all four models × three rollout seeds: prediction, failure flags,
q states and terminal boundary arrays are bitwise equal to uncached probe.
Adapter-family CPU205.631→176.989 seconds/run (13.9% reduction); original
188.308→158.926 (15.6%). Cached probe is52.1% cheaper than full depth2,
but21.2% more expensive than sparse without probing. Failure count remains
0/2304 for adapters and0/768 for original; accuracy is unchanged.

Single-window baseline CPU profile (DEV window0, seed1729) confirms exact
prediction parity. Profiled total15.302s includes instrumentation overhead:
event read cumulative4.638s, checker reject3.964s, F candidate execution
3.047s; _expand includes these and is14.256s. These nested cumulative times
must not be summed. This is one original-policy window, not a family-wide
cost estimate. It supports optimizing repeated policy/checker work as well
as F; there is no evidence that F alone explains the remaining cost.

Next bounded candidate: a viability-only expansion that stops at the first
legal successor and avoids unused child scoring/Node construction. Preserve
event→q→F, full fallback, root sampling temperature and RNG consumption;
require full cached-probe parity before claiming a pure speed optimization.
Broader controller-state caching needs hidden-state-aware keys and is not
implemented. Learned feedback and continuous residual experiments remain
separate; this feasibility study does not establish their effectiveness.

## Reproduction

For model seed0/1901/2718/3141 and mode probe/full/probe_cache:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python evaluate_adapted_checked.py --mode probe --model-seed 0 --workers 3
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python evaluate_adapted_checked.py --mode full --model-seed 0 --workers 3
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python evaluate_adapted_checked.py --mode probe_cache --model-seed 0 --workers 3
python summarize_commit_probe.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python profile_commit_probe.py
```

The prior `adapted_sparse_model*.json/.npz` are the matched strict-endpoint
baseline. Do not compare against legacy300-only sparse outputs. No TEST use.
