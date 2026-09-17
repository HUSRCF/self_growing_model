# Gradient accumulation budget controls — 2026-09-17

Previous goal turn made progress:3-seed on-policy learning plus initial
gradient-noise audit. This turn follows that measured-noise hypothesis.

## Controls

`train_closed_loop_policy.py --accumulate` averages separately backpropagated
rollout gradients at the SAME actor parameters. Clip once, update once.
No multiple optimizer steps hidden inside an accumulation cycle.

| Condition | Updates | Batches/update | Total training batches | Training transitions/model |
|---|---|---|---|---|
|single30, prior baseline|30|1|30|480000|
|accum3_updates10|10|3|30|480000|
|accum3_updates30|30|3|90|1440000|

Same3 optimizer seeds1901/2718/3141, both no-feedback/feedback, same TRAIN
prefix histories,40 initial windows ×4particles ×100steps per batch. Batch
RNG sequence uses42000+batch_index+train_seed-1901, so equal-budget arms
consume the same sequence of random seeds, at different actor states.

Seven validation checkpoints per condition, selected on3 TRAIN videos,
never DEV. Checkpoint update positions scale with total updates. Thus
the sample positions of checkpoints differ slightly between conditions;
equal budget means full training budget, not necessarily the training
consumed by the ultimately selected checkpoint. No best-seed selection.

Same32DEV windows,8particles,3 rollout seeds,300steps; no checker/search.
20 tests pass, including a mocked gradient test proving the arithmetic
mean precedes clipping/update and30 distinct training batches are used
for10×3, with14 validation runs at7 checkpoints. Existing19 tests retained.
Summary verifies baseline bitwise equality and exact truth/window alignment.

## Results

| Condition | Feedback | .5s RMSE | 1s RMSE | 3s RMSE | 3s energy |
|---|---|---|---|---|---|
|Frozen|—|.039119|.125447|.362147|.413651|
|single30|no|.039518|.128932|.364655|.417776|
|single30|yes|.039510|.126351|.364116|.416649|
|accum3_updates10|no|.039584|.126193|.363528|.414633|
|accum3_updates10|yes|.039604|.126441|.362450|.413082|
|accum3_updates30|no|.039578|.125887|.364834|.416738|
|accum3_updates30|yes|.039704|.126051|.363311|.414688|

Arithmetic means over optimizer/rollout seeds on the same windows, not
independent dataset replications or significance evidence. All numeric
failures0; not checker-valid completion. A few energy scores improve
slightly, but no configuration robustly beats frozen across horizons.

The unfavorable optimizer seed2718 remains unfavorable after averaging:
feedback1s for seeds1901/2718/3141 is .120301/.138485/.120536 at equal budget,
and .119264/.138352/.120536 at equal updates. Thus the earlier noise
measurement did not imply that a3× sampling increase would solve the
generalization gap. Do not keep increasing compute without a new testable
hypothesis. Model/default frozen behavior unchanged.

## Read-only motion-distribution audit

`audit_training_motion.py` measures RMS angular velocity over32 initial
observed frames (motion proxy, not physical energy). No fitting or TEST use.
TRAIN tails are used only for this diagnostic, never training/selection.

| Initial windows | Count | Median velocity RMS, rad/s | Mean | q=4 count | q=6 count |
|---|---|---|---|---|---|
|Fit TRAIN prefixes|40|16.7399|17.6341|2|5|
|Selection TRAIN prefixes|24|16.0952|17.7952|6|4|
|Same fit videos, tails|40|9.2309|10.3144|11|11|
|DEV tails|32|8.5568|8.7806|12|11|

There is a measured difference in initial motion/state coverage, including
within the same training videos over time. This is not a causal attribution
of prediction failures, nor a measurement of every free-run state. It
motivates comparing refreshed TRAIN-prefix windows and TRAIN-only q/motion
stratification against the fixed40-window sampler, at equal rollout budget.
Do not train on tails or choose sampling thresholds using DEV statistics.
Per-particle vs ensemble objective mismatch remains another open hypothesis.

Reproduce for each seed1901/2718/3141:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python train_closed_loop_policy.py --train-seed 1901 --accumulate 3 --epochs 10 --output adaptive_search_results/accum3_updates10_seed1901.json
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python train_closed_loop_policy.py --train-seed 1901 --accumulate 3 --epochs 30 --output adaptive_search_results/accum3_updates30_seed1901.json
python summarize_accumulation.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python audit_training_motion.py
```

Scripts/JSON/notes tracked; NPZ models and full trajectories stay local.
