# Refreshed and stratified TRAIN-prefix windows — 2026-09-17

Previous goal turn made progress: accumulation budget controls and initial
motion/state distribution audit. This turn tests state-coverage hypotheses
with fixed optimization/rollout budgets, preserving original model defaults.

## Training controls

`training_window_sampler.py` builds20813 eligible histories from only the
10 fit TRAIN-video prefixes. Starts63..prefix_stop-101; all100 target steps
remain strictly before the midpoint. q and motion are computed only from
the32 observed initial frames. No selection/DEV/TEST video is admitted.

Three samplers:

- Fixed: previous40 linspace starts, reused every update.
- Uniform: each batch draws4 distinct random starts per training video.
- Stratified: for each video, split each observed q into causal-motion
  quartiles; choose4 nonempty cells uniformly, then a start within each.
  Thresholds are fitted only within that video's TRAIN prefix. Cell
  balancing changes the training-state weighting; no importance correction.

Same30 batches/updates,4particles/window,100steps,30×40=1200 initial-window
draws per model.3 training seeds, with/without score feedback;7 checkpoints
selected on unchanged3 TRAIN-video prefixes. Original epoch0 eligible.
DEV evaluation32 windows ×8particles ×3 rollout seeds,300steps, no checker.

## Coverage verification

Uniform refresh reaches1163–1165 distinct starts versus40 fixed; stratified
reaches1141–1158. Control and feedback arms use exactly the same sampled
starts and sampler statistics within each seed.24 tests pass, including
forbidden-video rejection, strict history/target bounds, deterministic
sampling, invariance to changes in the discarded tail, and uniform CELL
selection rather than population-proportional selection.

Uniform sampled motion medians16.50/17.12/16.71rad/s. Stratified medians
17.44/17.63/17.31, not lower. Its q histogram is nearly balanced, but that
upweights rare fast-motion states. Thus q balance is NOT evidence that
the lower-motion DEV-tail distribution has been matched. No DEV thresholds
or weights were used, and no claim of solving domain shift is justified.

## 32-window results

| Sampler | Feedback | .5s RMSE | 1s RMSE | 3s RMSE | 3s energy |
|---|---|---|---|---|---|
|Frozen|—|.039119|.125447|.362147|.413651|
|Fixed|no|.039518|.128932|.364655|.417776|
|Fixed|yes|.039510|.126351|.364116|.416649|
|Uniform refresh|no|.039757|.127525|.358163|.408562|
|Uniform refresh|yes|.039619|.127210|.360171|.410834|
|q/motion strata|no|.038884|.121896|.367124|.420327|
|q/motion strata|yes|.039167|.125182|.364834|.416889|

Uniform-no-feedback improves3s mean RMSE by~1.1% versus frozen, with the
same direction in all3 training seeds (.354624/.361196/.358669), but loses
shorter-horizon means. Stratified-no-feedback improves short horizons but
loses3s. No overall deployment winner, and feedback is not uniformly useful.
Uniform feedback seed2718 and stratified feedback seed3141 select epoch0:
their predictions are verified bitwise equal to frozen, not a learned gain.
All numeric failures0; not checker-valid completion. Summary verifies
baseline bitwise equality, truth/window alignment, and no-op checkpoints.

## Denser DEV follow-up protocol

The uniform-no-feedback family was chosen for a post-hoc denser128-window
DEV audit based on these results. All3 TRAIN-selected optimizer checkpoints
are evaluated; no best-seed selection. Frozen baseline uses matching windows
and rollout RNG seeds. Same4 DEV videos, different denser start grid, NOT
an independent confirmation or untouched test. Script:
`evaluate_refreshed_windows.py --model-seed {0,1901,2718,3141}` (run separately).

Completed denser128-window result:

| Model family | .5s RMSE | 1s RMSE | 3s RMSE | 3s energy | 3s coverage90 |
|---|---|---|---|---|---|
|Frozen|.043282|.160446|.386798|.447307|.559896|
|Uniform refresh, no feedback|.043868|.161529|.382267|.441772|.571398|

3s RMSE improves1.17%, energy improves1.24%; short horizons slightly
worsen. Per-optimizer3s RMSE .379040/.382119/.385643, each below frozen
.386798. Per-video3s improves3/6/8, worsens9. Zero numeric failures.
This is a modest tradeoff, not a statistically established overall winner.
All candidate/baseline window/truth arrays are checked identical within
the128-window audit. It does not turn reused DEV into a fresh holdout.

Next: preserve this TRAIN-selected no-feedback family as a candidate and
test whether benefit survives the mixture checker / sparse-search policy.
No-feedback is important: future state remains identical for equal r at
a fixed node, so destination-only bans remain meaningful. Do NOT use
the event-specific feedback adapter with those bans without revisiting
the state identity. Frozen defaults remain unchanged. Separately, q balance
did not lower motion; any low-motion enrichment needs its own TRAIN-only
definition and coverage audit, not post-hoc DEV distribution matching.

## Reproduction

For each training seed1901/2718/3141 and mode uniform/stratified:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python train_closed_loop_policy.py --window-sampling uniform --train-seed 1901 --output adaptive_search_results/windows_uniform_seed1901.json
python summarize_window_sampling.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m unittest test_training_window_sampler test_closed_loop_policy test_feedback_distribution test_adaptive_search -q
```

Scripts/JSON/notes tracked; NPZ models and trajectories local. No frozen
package edits, no new test-set evaluation. Long-term research goal active.
