# Boundary repair and confidence/distribution pilot — 2026-09-17

## Boundary repair

Opt-in `--boundary-repair` broadens the current root to all event/q edges
and requests the configured lookahead when normal search has no candidate
and further rollback would cross the committed frontier. Checker thresholds,
F and the revision window remain unchanged. This is finite beam repair,
not exhaustive feasibility or a real-time deadline guarantee.

Reproduced previous failure: DEV window22, seed3141+22, particle2, step46.
Current q4, banned destinations [3,4,5,6]. Normal top-k candidates followed
by bans leave zero roots; full root enumeration exposes legal unbanned
q0/1/2/7. Thus earlier max-three-step withdrawal was strategy dependent,
not proof that a two-step window is impossible.

32 common DEV windows, 8 particles, 300 steps, seeds1729/2718/3141:

| Seed | Repair attempts/successes | Endpoint failures | Max withdrawal | .5/1/3s embedding RMSE |
|---|---|---|---|---|
|1729|0/0|0/256|2|.034277/.129370/.394282|
|2718|0/0|0/256|2|.035102/.135010/.399226|
|3141|1/1|0/256|2|.035842/.109654/.371716|

Without repair, seed3141 had1/256 failed particles. Seeds1729/2718 have
identical summary scores/expansion counts. Changing one branch can shift
later particles' RNG within the same window in this existing rollout.
Do not interpret minor accuracy changes as independent improvement.
11 prototype unit tests pass, including committed-prefix preservation
during synthetic boundary repair.

## User-requested next experiments

1. Feed previous selected event/q confidence, entropy and checker margin
   into a trainable controller adapter. Compare same-width adapter with
   all feedback channels zero. Backbone frozen. Scores are beliefs, not
   calibrated values; extra causal information can help or reinforce errors.
2. Replace deterministic F angle outputs with zero-mean residual Gaussians
   centered at those outputs. Fit q/r-conditioned second moments on TRAIN
   prefixes, shrink sparse pairs toward pooled covariance. Keep q discrete.
   This differs from continuous latent state centered at q clusters.

`feedback_distribution_pilot.py` implements a 3×2 factorial: frozen,
adapter without feedback, adapter with feedback × point/Gaussian writer.
New fitting uses10 TRAIN videos;3 TRAIN videos select adapter epochs.
Frozen original model has seen these TRAIN videos already. Evaluate DEV
tails only; no new test use. Initial pilot has NO checker or search in
all arms to isolate the mechanisms. Gaussian samples use separate RNG
so they do not shift event/q random draws by consumption alone.

Important limitations: teacher-forcing exposure mismatch; independent
residual noise lacks temporal coherence; fixed-center Gaussian cannot
correct bias; controls do not isolate confidence semantics from generic
extra causal history. Subsequent search integration must revisit bans:
with event-specific feedback, equal r reached by different e no longer
necessarily implies identical future state. Existing destination-only
rollback bans must not be reused blindly.

## Completed initial factorial results

TRAIN prefixes13312 states, including3072 adapter-selection states from
3 whole TRAIN videos.32DEV windows ×8particles ×3seeds,300steps.
All six arms use the same event/q uniform streams, with separate residual
noise streams. CPU NumPy inference/PyTorch adapter fitting, BLAS threads1.

| Arm | .5s RMSE | 1s RMSE | 3s RMSE | 3s energy score | 3s embedding coverage90 |
|---|---|---|---|---|---|
|Frozen point|.039119|.125447|.362147|.413651|.614583|
|Adapter point|.036334|.144057|.369020|.427281|.526042|
|Feedback point|.035906|.142693|.366136|.427187|.526042|
|Frozen Gaussian|.055925|.203597|.432564|.525566|.781250|
|Adapter Gaussian|.053342|.204748|.432373|.519530|.778646|
|Feedback Gaussian|.052457|.203438|.428120|.509541|.802083|

Values are arithmetic means over three seed metrics, not independent
confidence intervals. Feedback wins against no-feedback adapter by
1.18%/.95%/.78% at .5/1/3s, but does not beat frozen1/3s. At3s it improves
DEV videos9/6 and worsens3/8 versus adapter. Observed q NLL frozen .196425,
adapter .198049, feedback .193919: small teacher-forced classification
improvement does not establish long-rollout improvement.

Every arm has zero numeric failures under the pilot safety guard. This
does NOT mean checker-valid completion: checker is diagnostic only, and
frozen point violates it on10.77% of assessed transitions. Do not directly
compare these runs with checked-search completion or frozen official results.

Fixed-center conditional Gaussian90% ellipse coverage is91.93% on adapter
TRAIN holdout and94.38% on observed DEV transitions. These are conditional
on observed q/r; rollout embedding coverage is a different statistic. Wider
rollout coverage comes with worse energy score and point error. Thus simply
adding independent local residual samples is not currently a winning model.

Next candidates, not yet tested: train score/value feedback with deployed
multi-step continuations and free-running histories; model a persistent
latent residual/velocity uncertainty rather than independent angle jitter;
permit learned residual bias with an explicit fixed-center control; compare
against extra-history (non-score) feedback to isolate confidence semantics.
No new settings selected using the test set; frozen package untouched.

## Temporal residual follow-up

Standardized fit-TRAIN residual lag1 correlations: -.560613/-.529534.
The IID Gaussian assumption is therefore questionable on observed data.
Added post-hoc AR(1) standardized-noise control with those TRAIN-derived
coefficients, no DEV coefficient sweep. Same marginal noise scale, separate
noise stream. The conditional residual mean now depends on previous noise;
unlike the initial Gaussian, each conditional center is no longer exactly F.

| Arm | .5s RMSE | 1s RMSE | 3s RMSE | 3s energy | 3s coverage |
|---|---|---|---|---|---|
|Frozen AR Gaussian|.045655|.157958|.411238|.479479|.729167|
|Adapter AR Gaussian|.041866|.155644|.418404|.484893|.747396|
|Feedback AR Gaussian|.040785|.157491|.409984|.471252|.742188|

Temporal structure recovers some IID-noise loss but does not beat point
baseline. This supports addressing temporal structure, not claiming a
successful continuous model. Marginal observed-angle Gaussian NLL for
frozen/adapter/feedback is -9.924492/-9.924694/-9.924737 (rad-density units,
including latent r mixture). Minuscule teacher-forced differences again
do not establish free-run gains.15 tests pass, including rho0 exactly
matching independent-noise outputs.

Reproduce:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python feedback_distribution_pilot.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python feedback_distribution_pilot.py --temporal-only
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m unittest test_adaptive_search test_feedback_distribution -q
```

First command overwrites the initial report; second appends temporal arms.
JSON report is tracked; fitted NPZ models and trajectory arrays stay local
under existing binary-artifact ignore rules.
