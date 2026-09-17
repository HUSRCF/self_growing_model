# Predictable continuous center correction — 2026-09-17

Previous turn's TRAIN oracle audit separated routing and center error,
including destination inconsistency. This experiment tests actual causal
residual predictability rather than treating oracle headroom as learnable.

## Model and data

Observed TRAIN-prefix chains16starts×64steps/video, same framing as the
writer audit.10 fit videos (10240 histories),3 hold videos (3072 histories).
For every history enumerate all8 F(q,r) candidates AFTER reading p(e),T.
Fit wrapped angle residuals truth−F weighted by frozen marginal p(r), so
each history has total weight1. No oracle destination is supplied as input
or training label. q and r enter as categorical one-hot vectors, not numeric
coordinates. The original backbone already saw all TRAIN videos; new
holdout pertains only to this correction module.

Controls: zero correction; shared weighted-mean constant; linear ridge
with causal continuous_features plus current q/selected r. Feature mean/
scale from fit videos only, standardized features clipped to±8. Ridge
regularization1e-4/.01/1, intercept unpenalized; correction clipped per angle
to fit residual absolute99th percentile. This is a deterministic convex fit,
not a stochastic optimizer needing three training seeds. No added noise.

Ridge strength chosen by hold one-step expected embedding MSE. Then evaluate
zero, constant and that ridge under common free-running300-step TRAIN hold
windows,8particles,seeds12017/12019; choose mean energy U at50/100/300 plus
failure penalty, with zero eligible. Same hold videos used for both stages,
not independent validation. Only a nonzero winner triggers DEV32 three-seed
evaluation against zero; failed ridge is not selected using DEV.

Inference: event→r→original F(q,r)→bounded correction→commit. GRU, F files,
checker calibration untouched. q remains the sampled r, not reclassified
from the corrected angle. All checker/q-consistency numbers are diagnostics;
no checking/rejection/search is enforced in these free-running arms.

## Initial outcome

Hold one-step expected MSE2.207147e-6→2.139137e-6 for ridge1e-4 (~3.08%
improvement); fit2.343817e-6→2.254802e-6 (~3.80%). This is a measured gain
on held videos used for selection, not independent confirmation or removal
of the whole oracle residual. Weighted destination contradiction increases
10.753%→11.033%; current-middle rejection1.009%→.987%. Consistency and
mixture validity are different diagnostics, as in the preceding audit.

Free TRAIN-hold cost: zero .446979874, ridge .457781905 (~2.42% worse),
constant .446929011 (~0.0114% better). Thus ridge fails the free-run gate
despite one-step fit improvement. Constant correction RMS is only
3.72e-6rad; its weak selection advantage needs sampling-sensitivity checks.

DEV only zero versus the preselected constant:

| Writer | .5s RMSE | 1s RMSE | 3s RMSE | Mean50/100/300 U objective |
|---|---|---|---|---|
|Zero|.039119|.125447|.362147|.168089|
|Constant|.039006|.123079|.368501|.168894|

Medium-horizon improvement trades against worse3s and worse common
distribution objective. No deployment/default change justified. Small
changes can alter sampled branches and amplify over trajectories; do not
infer that this constant estimates a substantial physical bias.

## Verification and confirmation design

58 tests pass: holdout perturbations do not change fit parameters; bounded
linear correction; exact zero-writer rollout; no future-truth influence on
corrected inference; all prior tests. Zero free TRAIN outputs match the
original run_policy exactly; zero DEV outputs match the earlier frozen
pilot bitwise. Every model is serialized/reloaded with exact parameter
checks, including the constant (not only the selected ridge). NPZ models
and trajectories remain local under existing binary ignore rules.

Follow-up fixes the selected constant and uses fresh action seeds22017–22024
on the SAME three TRAIN hold videos. No parameters, thresholds or winner
are reselected. Report paired energy differences and conditional RNG
standard error, not a population/video-generalization confidence claim.

Completed fresh-seed check: paired objective difference(candidate−zero)
mean+.00019835, conditional RNG standard error .00609206,5/8 seeds worse.
The original selection improvement was only−.00005086. Thus no stable
benefit is demonstrated; this does not establish a reliably harmful effect
either. It is a sampling-sensitivity failure, not eight independent datasets.
Constant vector[-2.2262e-6,4.7660e-6]rad; ridge caps[.00442047,.00786352]rad.
All evaluated trajectories have zero numeric failures, not checked validity.

Next inspect the fixed ridge correction on FREE-generated TRAIN histories:
does its direction still reduce the aligned next-frame residual, or does
teacher-forced predictability disappear under drift? Compare identical
states before proposing on-policy residual fitting. Do not automatically
increase correction magnitude or add noise; neither follows from this result.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python continuous_residual_pilot.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python validate_residual_gate.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m unittest test_continuous_residual test_writer_headroom test_training_horizons test_ensemble_objective test_particle_frontier test_read_cache test_commit_probe test_stateless_adapter test_training_window_sampler test_closed_loop_policy test_feedback_distribution test_adaptive_search -q
```
