# Stateless adapter under checked and sparse runtimes — 2026-09-17

Previous goal turn made progress: refreshed-window training and denser DEV
audit yielded a modest long-horizon candidate. This turn checks composition
with constraints/search rather than assuming unchecked gains transfer.

## Integration contract

`stateless_policy_adapter.py` exports/verifies a manifest for the selected
`closed_loop` (NO feedback) arrays. Model SHA256 and actual frozen-head
array fingerprint are checked; event-feedback contracts and reset backbones
are rejected. Adapter adds q logits inside each event's transition law;
event probabilities, F and checker are unchanged. Causal continuous features
are computed once per read. Read is pure; equal parent/history/r gives
equal commit memory for all e, preserving destination-ban equivalence.

`--policy-adapter` is opt-in in the isolated search prototype. Frozen package
files and default model behavior are not changed. Manifests reference local
NPZ artifacts, which remain excluded from Git under prior binary rules.

## Endpoint validity correction in this experiment

The legacy search runner returns H points without a committed extra right
neighbor. The final point is not certified by that returned trajectory;
on a failed next step, the previous point may also be uncertified. The
legacy runner now explicitly labels this scope in JSON.

New `evaluate_adapted_checked.py` runs sparse search for301 internal points,
returns300, shifts failure validity one step back and retains the correct
right boundary for the last valid point. Every claimed-valid output is
independently checked with the original central derivative and its q label,
including the final returned point. Official CheckedMachine already saves
the extra right boundary; it receives the same independent audit.

Adding a step changes later particles' RNG consumption in the legacy shared
per-window stream. Therefore sparse baseline is rerun under this protocol;
do not mix its scores with historical300-only sparse runs.

29 tests pass: trained-controller read parity, pure/non-event-specific
commit, wrong-backbone/feedback-contract/model-hash rejection, missing-right
validity shift, and terminal-point checking, plus previous24 tests.

## Matched runtime comparisons

32 common DEV windows,8particles,3 rollout seeds. All3 TRAIN-selected
optimizer models are retained; no best-seed choice. Checked runtime uses
its original stochastic rollback budget. Sparse uses depth2, period5,
temperature-scale matching, shared writer,300 total rewinds,2-step revision
window and boundary root widening. Their budgets/selection differ; compare
each adapter family against the matching runtime baseline.

| Runtime / policy | .5s RMSE | 1s RMSE | 3s RMSE | Endpoint failure | Mean CPU seconds/run |
|---|---|---|---|---|---|
|Checked frozen|.033691|.144262|.398124|0|3.899|
|Checked adapter family|.033520|.142024|.395217|0|4.151|
|Sparse frozen, new endpoint protocol|.035513|.130583|.388887|0|133.418|
|Sparse adapter family|.040647|.142315|.387230|.00390625|145.984|

Checked adapter modest mean gains at all horizons,~6.5% CPU overhead;
3s energy .473970 vs .476674, but coverage .335938 vs .359375.3s model-seed
means .394819/.391492/.399342; one seed worsens vs frozen. DEV exploratory,
not a robust universal upgrade or equal-cost particle comparison.

Sparse adapter has9 failed endpoint particles out of2304:1 from model2718,
8 from model3141. Baseline0/768. All VALID returned points in all runs pass
independent checks with zero violations; failed suffixes are explicitly
marked/held placeholders and do NOT count as valid completion. Its slightly
lower3s headline RMSE cannot override completion and short-horizon losses.
CPU overhead~9.4%. Do not deploy the sparse composition as tested.

## Dead-end diagnosis

`audit_adapted_deadends.py` exactly reproduces model2718/rollseed1729/window16
and model3141/rollseed2718/window13, including predictions and failure masks.
At terminal revision boundary, full root enumeration has unbanned candidates,
but NONE has a full one-step successor. Banned rootq6 still has short-horizon
support; earlier finite search failure is not proof of global impossibility.
Do not merely erase bans and risk cycling.

Selected-window probes (NOT benchmark-wide improvements):

| Probe | Failed particles, first/second window | Revision limit |
|---|---|---|
|Existing period5 depth2|1 /4|2|
|Widen whenever raw expansion empty|1 /4|2|
|Full root before each backtrack|1 /4|2|
|Depth2 every step|0 /0|2|
|Allow revision3|0 /0|3|

Every-step probing costs~11.4CPU s/window in these diagnostics versus
~4.5s for revision3; these are selected failed windows, not a global speed
comparison. Full-root-before-backtracking also fails to rescue these cases;
simply postponing bans until a broader root search does not solve them.
No claim of minimum required depth or unavoidable three-step latency.

Next concrete experiment: a cheap pre-commit viability probe on the chosen
root, requiring at least one admissible next point before accepting it.
Compare against every-step depth2 and matched sparse baselines, not only
these selected failed windows. This may retain useful early lookahead
without scoring every root continuation; it is NOT yet implemented or
guaranteed. Training on unconstrained rollouts still differs from deployed
checked/search behavior, so any new value targets must respect that policy.

## Reproduce

For each optimizer seed1901/2718/3141, first export its existing TRAIN report:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python stateless_policy_adapter.py adaptive_search_results/windows_uniform_seed1901.json
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python evaluate_adapted_checked.py --mode checked --model-seed 1901
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python evaluate_adapted_checked.py --mode sparse --model-seed 1901
python summarize_adapted_checked.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python audit_adapted_deadends.py
```

Also run model-seed0 for both baseline modes. Frozen artifacts untouched,
no new TEST access, research goal stays active.
