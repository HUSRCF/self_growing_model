# Closed-loop policy training — 2026-09-17

Previous goal turn made progress: confidence/distribution factorial and
temporal-noise controls produced evidence, code, tests and remote checkpoint.
This turn follows the exposure/target mismatch hypothesis with on-policy
training, not another teacher-forced q classifier or arbitrary noise scale.

## Method and scope

`train_closed_loop_policy.py`: frozen original event GRU and F, residual
32-wide adapter adds eight logits inside each event's q transition law.
Event sampled first, then q, then F. No checker/search/rollback or angle
noise in any arm. Mixture violations are diagnostic, not rejected.

Actor actually generates100-step trajectories; REINFORCE uses future
per-particle embedding MSE at50/100 steps (plus numeric failure penalty).
Other particles from the same initial history form leave-one-out baselines;
already-passed horizon errors cannot credit later actions. Fixed gradient
scale .02, AdamW lr .003, local transition KL weight .01, gradient clip1.
KL is a local visited-state regularizer, not an exact differentiated
long-run occupancy KL objective.

Feedback channels: previous selected e/q frozen-backbone probabilities,
event/q entropy, mixture margin/reject flag. Using frozen probabilities
avoids silently dropping gradients through previous trainable scores.
This differs from the earlier teacher-forced adapter feedback definition.
Control uses the same architecture with seven zero feedback channels.

Fit:10 TRAIN-video prefixes,4 windows/video,4 independent particles/window,
new action RNG each epoch. Select:3 other TRAIN-video prefixes,8 windows/video,
8 particles,2 fixed validation RNG seeds.30 epochs, evaluate every5, original
zero-update actor is eligible at epoch0. Original GRU has previously seen
these TRAIN videos: holdout pertains only to new adapters.

Run optimizer/rollout-training seeds1901/2718/3141; select each independently
without looking at DEV. DEV: common32 windows,8 particles,3 trajectory seeds,
300steps. Training/selection target is per-particle MSE, not ensemble RMSE;
3s is beyond the100-step training horizon. Do not conflate these objectives.

## Results

All six trained adapters reduce their selected TRAIN-holdout objective;
selected epochs are15/20/15 for both feedback conditions. The initial
holdout objective is .115206; selected costs range .106518–.108789.

| Method, averaged across optimizer/trajectory seeds | .5s RMSE | 1s RMSE | 3s RMSE | 3s energy | 3s coverage |
|---|---|---|---|---|---|
|Frozen|.039119|.125447|.362147|.413651|.614583|
|Closed-loop, no feedback|.039518|.128932|.364655|.417776|.585069|
|Closed-loop, feedback|.039510|.126351|.364116|.416649|.587674|

All have zero numeric failures, not checker-valid completion. Feedback
reduces1s RMSE relative to the same-capacity closed-loop control in all
three optimizer seeds, but the absolute result is not robust versus frozen:

| Optimizer seed | No-feedback1s | Feedback1s | Frozen1s |
|---|---|---|---|
|1901|.126744|.120252|.125447|
|2718|.139682|.138910|.125447|
|3141|.120370|.119891|.125447|

Seed1901 alone suggested ~4%1s improvement, which does not survive averaging
over all optimizer seeds. Do not publish/select only the winning seed.
9 optimizer/trajectory combinations reuse32 DEV windows; not9 independent
datasets and no statistical-significance claim. At3s, feedback aggregate
improves video6 but worsens videos3/9/8. No deployment change justified.

## Verification and next question

19 tests pass: baseline excludes own rollout; rewards exclude past horizons;
zero actor matches frozen output; changing future truth cannot change
inference predictions; finite nonzero policy gradient leaves original GRU
arrays unchanged; existing search/distribution tests retained.
`summarize_closed_loop.py` verifies frozen predictions bitwise against the
previous pilot for all seeds, and exact window/truth alignment across arms.

This narrows the diagnosis: changing to on-policy training alone is not
sufficient. Next measure gradient noise, then decide whether rollout
sample size / learned baseline is warranted. Training-video vs DEV-video
distribution shift and per-particle vs ensemble objectives also remain
unresolved. Do not claim gradient noise is the sole cause before measuring.

Reproduce (all CPU, no120-thread oversubscription needed):

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python train_closed_loop_policy.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python train_closed_loop_policy.py --train-seed 2718 --output adaptive_search_results/closed_loop_policy_train2718.json
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python train_closed_loop_policy.py --train-seed 3141 --output adaptive_search_results/closed_loop_policy_train3141.json
python summarize_closed_loop.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python audit_policy_gradient_noise.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m unittest test_adaptive_search test_feedback_distribution test_closed_loop_policy -q
```

NPZ models/trajectories remain local under existing binary ignore rules;
scripts, JSON and long-term notes are tracked. Frozen package unchanged.

## Gradient-noise audit completed

`audit_policy_gradient_noise.py`: same initial actor and40 fit-TRAIN
histories,12 independent action RNG batches of4particles/history, no
optimizer updates. Control/feedback mean pairwise gradient cosine
.095879/.097797;28.79% of gradient pairs have negative cosine. Estimated
single-batch gradient signal-to-noise .311598/.316962 after finite-batch
mean-noise correction. These are descriptive12-batch estimates; cosine
pairs are dependent. They isolate trajectory-sampling noise, not video
sampling variance, and apply only at the initial actor.

The gradient has an observable common component but substantial sampling
noise. Next concrete ablation: average several independent rollout batches
per update, reporting both equal-update and equal-rollout-budget controls;
then evaluate a learned causal value baseline if justified. Do not claim
that more samples alone solves the DEV generalization gap. Keep all3
optimizer seeds, no best-seed selection on DEV.
