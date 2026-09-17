# Ensemble-energy policy training — 2026-09-17

Previous turn was progress: measured compute frontier and V/U energy
ranking reversal motivated testing the training objective, not more caches.

## Objective and estimator

Frozen GRU/F; same residual q-logit actor, mandatory event→q→F order.
On-policy rollouts, no checker/search/noise in any arm. Scores fed back are
previous frozen-backbone beliefs/margins, not a newly learned value function.
Targets at50/100 steps,3s only evaluated out of training horizon.

For each group of P trajectories, energy U is mean distance to truth minus
half mean pair distance over distinct particles, plus2×numeric-failure rate.
Distances use the four sine/cosine angle-embedding coordinates. This targets
marginal horizon distributions, not a proper joint-trajectory score or
physical feasibility. Its independent-sample population interpretation is
appropriate to this fixed-size unchecked sampling design; finite P still
has substantial variance. V energy is also reported on DEV outputs.

The REINFORCE baseline for particle i is the energy of the P−1 OTHER paths,
removing all pair terms incident to i. Advantage is P×(whole cost−baseline)
because the loss averages over particle log probabilities; for an additive
MSE cost this convention recovers the original per-particle gradient scale.
Only costs at future horizons credit an action. Truth affects costs, never
inference states or inputs. No differentiation through the frozen NumPy F.

Exhaustive P3 Bernoulli-support{0,2}/truth0.7 test confirms expected energy
`.7−1.4p+2p²` and score-function gradient equal exact differentiated
expectation, including the P factor. Other tests verify baseline exclusion,
particle permutation, frozen inference/truth independence, finite nonzero
energy gradient, unchanged backbone, and independent selection objective.
48 tests pass including all previous search/learning controls.

## Experimental controls

Three pipelines × no-feedback/feedback × optimizer seeds1901/2718/3141:

1. MSE training, MSE TRAIN-holdout selection (reproduce previous uniform run).
2. Energy U training, Energy U selection.
3. MSE training, Energy U selection (common-selection control).

Each: same30 updates/30 rollout batches,4 particles×40 refreshed histories
per batch,10 fit-TRAIN video prefixes;3 other TRAIN prefixes select among
7 checkpoints including epoch0, using8particles and2 fixed rollout seeds.
Same fitting windows by optimizer seed in every pipeline and feedback arm.
Original backbone has previously seen all TRAIN videos; holdout pertains
only to adapters. DEV32,8particles,3 rollout seeds; no TEST use.

MSE rerun fitted arrays/training trace/selected epochs/DEV predictions are
exactly the previous `windows_uniform_seed*` results. Changing only selection
to energy leaves the entire MSE training trace exactly unchanged. Frozen
and selected-epoch0 predictions match the earlier frozen pilot bitwise.
All window/truth/sampler alignment checks pass. Fixed Adam/learning rate,
gradient scale .02 and KL weight .01 are shared, but loss units differ.
Observed mean gradient norms MSE .026–.033 vs energy .056–.062; no update
hits the clip threshold1. Thus clipping is not the cause of this result,
although objective-unit/optimizer effects are not fully isolated.

## Completed DEV results

Means over all three optimizer seeds and three rollout seeds (same videos):

| Training / selection / feedback | .5s RMSE | 1s RMSE | 3s RMSE | 3s U energy |
|---|---|---|---|---|
|Frozen|.039119|.125447|.362147|.372552|
|MSE / MSE / off|.039757|.127525|.358163|.367000|
|MSE / MSE / on|.039619|.127210|.360171|.369296|
|Energy / Energy / off|.040077|.128303|.363496|.374041|
|Energy / Energy / on|.039202|.124964|.364116|.376010|
|MSE / Energy / off|.039619|.127834|.359172|.367873|
|MSE / Energy / on|.039910|.129614|.361566|.371554|

Energy+feedback's1s mean is~0.38% below frozen, but .5s/3s are worse.
Its selected epochs15/0/0 mean only one optimizer seed deploys a learned
update; the others revert to the original. No robust learned upgrade.
Energy without feedback is worse than MSE under the SAME energy selection
criterion at all reported RMSE horizons and3s energy. With feedback there
is a medium-versus-long-horizon tradeoff, not blanket improvement.
The common-selection control shows changes cannot be attributed solely to
using a different holdout metric. Initial TRAIN energy is .243808; energy
selection improves it for some fits but does not guarantee DEV improvement.
No numeric failures in these arms; NOT evidence of checker-valid completion.

## Conditional gradient-noise follow-up

Same initial actor and first uniform TRAIN batch (sample seed57000),12
independent action-RNG replicates,4particles/window, both feedback controls.
MSE estimated corrected single-batch SNR .1660/.1672; energy .3127/.3070.
Mean pairwise cosine .03595/.03674 vs .07346/.07137. Energy's absolute
gradient-noise trace is larger, but its estimated signal is larger too:
this audit does NOT support attributing poorer results to lower relative
gradient quality. Negative-pair fractions remain high (.424 vs .439/.455).
Only one initial state distribution and finite12 replicates, not a causal
explanation of later training or generalization. Earlier fixed-window noise
audit used different histories; compare these newly matched runs instead.

Next bounded question: training horizon/target mismatch. All new losses
still use50/100-step marginals, while3s is beyond supervision. A long-horizon
energy arm should be compared with a short-horizon arm using the same
300-step-eligible TRAIN histories and a common TRAIN-holdout criterion;
separately match total simulated steps, not only optimizer updates. Keep
original epoch0 and all seeds. Do not claim insufficient sampling is the
cause based on this noise audit, and do not tune horizon choices on TEST.

Reproduce:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python train_closed_loop_policy.py --objective-kind mse --window-sampling uniform --train-seed 1901 --output adaptive_search_results/objective_mse_seed1901.json
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python train_closed_loop_policy.py --objective-kind energy_u --window-sampling uniform --train-seed 1901 --output adaptive_search_results/objective_energy_u_seed1901.json
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python train_closed_loop_policy.py --objective-kind mse --selection-objective-kind energy_u --window-sampling uniform --train-seed 1901 --output adaptive_search_results/objective_mse_select_energy_seed1901.json
# Repeat all three with optimizer seeds2718 and3141.
python summarize_objective_training.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python audit_policy_gradient_noise.py --objective-kind mse --uniform-window --output adaptive_search_results/objective_noise_mse.json
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python audit_policy_gradient_noise.py --objective-kind energy_u --uniform-window --output adaptive_search_results/objective_noise_energy_u.json
```

These are isolated opt-ins; MSE defaults, official runtime and frozen ZIP
contents are unchanged. Default policy is not promoted from reused DEV.
