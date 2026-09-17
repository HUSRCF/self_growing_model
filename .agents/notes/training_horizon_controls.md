# Training horizon and simulation-budget controls — 2026-09-17

Previous turn completed objective/selection controls and gradient audits.
Local fetch had failed after remote publication; this turn fetched the
published bbe2ed9 checkpoint successfully and synchronized the local index
without discarding files. The research question is100-step fitting versus
300-step evaluation, not another independent DEV parameter sweep.

## Protocol

All arms sample from the SAME fit-TRAIN prefix pool requiring300 future
steps. Short arms truncate only the target array to100; starting histories
and q coverage are not drawn from a larger/easier eligibility pool. Both
feedback controls and all optimizer seeds1901/2718/3141 are retained.

| Arm | Training rollout | Loss horizons | Updates | Batches/update | Simulated training transitions/model |
|---|---|---|---|---|---|
|short|100|50,100|30|1|480000|
|long|300|50,100,300|30|1|1440000|
|short_budget|100|50,100|30|3|1440000|

Each batch:40 refreshed TRAIN histories×4 particles. Energy U with causal
leave-one-trajectory-out baseline; frozen GRU/F, same residual q adapter,
same fixed Adam/KL/gradient scale. No checker/search/noise in training or
DEV evaluation. Only raw belief/margin feedback, not a learned value head.

Common TRAIN-holdout:3 selection videos,8prefix windows/video,8particles,
two rollout seeds,300 steps, mean energy U plus failure cost at50/100/300.
Seven checkpoints including original epoch0 for every arm. Holdout windows
are300-eligible and identical across arms, not the previous100-step starts.
DEV uses original32 common tails,8particles,3 rollout seeds. No TEST use.

Short/long match updates, batches and sampled histories; short_budget/long
match simulated training transitions but short_budget has more independent
histories per update. Validation/evaluation protocol is common. These are
not equal wall-clock-cost controls: energy evaluations, backpropagation and
array overhead differ. Policy loss retains division by rollout length;
100→300 changes its global scale relative to fixed KL, so do not claim a
scale-free isolated causal horizon effect. Gradient norms/clipping recorded.

## Verification

50 tests pass: history/eligibility identity,300-step common holdout, budget
counts, late-target credit assignment, prior energy-gradient and no-leakage
tests. Summary checks same sampler audits for short/long and identical pool
metadata for all three, exact initial holdout objective, simulated-step
counts, window/truth/RMSE alignment, frozen and selected-epoch0 predictions.
Separate default100-step MSE seed1901 replay compares both feedback arms'
complete training traces, selected arrays and all DEV outputs against the
previous checkpoint. Defaults and frozen package remain unchanged.

## Completed result

The full summary verifies all listed controls, including exact default
replay. All arms have zero numeric failures; this is not checker validity.

| Training / feedback | .5s RMSE | 1s RMSE | 3s RMSE | 3s U energy |
|---|---|---|---|---|
|Frozen|.039119|.125447|.362147|.372552|
|short / off|.039247|.127538|.363917|.373673|
|short / on|.039404|.130616|.356663|.365496|
|long / off|.039278|.128500|.360206|.370186|
|long / on|.039405|.129015|.361082|.370514|
|short_budget / off|.038952|.125934|.359645|.368130|
|short_budget / on|.038728|.126167|.358387|.368538|

At equal1.44M training transitions, short_budget has lower mean RMSE at
all three horizons and lower3s U energy than long in BOTH feedback arms.
Long versus short improves3s without feedback, but worsens3s with feedback;
there is no uniform duration benefit. Short+feedback is actually best3s
in this grid, while losing1s quality. No default winner or significance
claim from three optimizer seeds on four reused DEV videos.

Selected epochs (off/on) by optimizer1901/2718/3141:
short5/20,0/25,0/0; long30/30,0/0,5/5;
short_budget30/30,30/30,0/0. Many arms revert to epoch0. Initial common
TRAIN-holdout energy .446980; best short_budget holdout .433936 (2718 off),
but holdout gains alone do not establish universal DEV gains. No gradient
clipping in any update. Different step normalization and history counts
per update remain declared limitations, not proof that horizon is irrelevant.

Next diagnosis should inspect writer realizability, not merely increase
duration again: on TRAIN histories compare the actual routing error with
the privileged best of all F(q,r) candidate centers, and measure residual
corrections still required at that oracle center. This can distinguish
one-step routing headroom from a candidate-output limitation. Oracle uses
truth only as a diagnostic and must never become deployable inference or
be presented as a long-rollout upper bound. Continuous residual learning
is a candidate follow-up, not established necessary by the present study.

Reproduce for each optimizer seed1901/2718/3141:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python train_closed_loop_policy.py --objective-kind energy_u --window-sampling uniform --train-steps 100 --eligible-steps 300 --selection-steps 300 --loss-horizons 50 100 --selection-horizons 50 100 300 --accumulate 1 --train-seed 1901 --output adaptive_search_results/horizon_short_seed1901.json
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python train_closed_loop_policy.py --objective-kind energy_u --window-sampling uniform --train-steps 300 --eligible-steps 300 --selection-steps 300 --loss-horizons 50 100 300 --selection-horizons 50 100 300 --accumulate 1 --train-seed 1901 --output adaptive_search_results/horizon_long_seed1901.json
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python train_closed_loop_policy.py --objective-kind energy_u --window-sampling uniform --train-steps 100 --eligible-steps 300 --selection-steps 300 --loss-horizons 50 100 --selection-horizons 50 100 300 --accumulate 3 --train-seed 1901 --output adaptive_search_results/horizon_short_budget_seed1901.json
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python train_closed_loop_policy.py --window-sampling uniform --train-seed 1901 --output adaptive_search_results/horizon_default_parity_seed1901.json
python summarize_training_horizons.py
```
