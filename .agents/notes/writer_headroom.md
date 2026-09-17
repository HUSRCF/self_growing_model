# TRAIN candidate-center versus routing headroom — 2026-09-17

Previous turn completed horizon/budget controls; longer rollouts did not
uniformly win. This is a read-only diagnostic before altering F, not a
new inference policy, fit, DEV experiment or oracle deployment.

## Data and quantities

Only first halves of all13 TRAIN videos. Report10 prior fit videos and3
prior adapter-holdout videos separately; the original backbone has seen
all TRAIN videos.16 initial histories/video,64 teacher-forced steps each:
10240 fit and3072 hold rows. Also16 histories/video×3 action seeds at
free-run offsets0/24/100; drift0 repeats identical initial contexts.
No statistical independence of windows or repeated initial contexts claimed.

At each history read p(e),p(r|e), then enumerate F(q,r) centers. Since the
current writer has no event-specific state, marginal p(r)=sum_e p(e)p(r|e)
gives exact one-step candidate mixture weights. Truth is used only AFTER
candidate creation for scores. Free trajectories preserve original event
sampling, q sampling, F and numeric guard; snapshots match frozen rollout
histories exactly in tests. Teacher-forced chains append observed truth,
explicitly not a deployable rollout.

Embedding errors use four sin/cos coordinates. Report expected single-point
MSE, local mixture-mean MSE, candidate variance, and privileged min-r MSE.
Identity expected=mixture-mean-error+candidate-variance holds numerically.
The min-r floor applies ONLY to selecting one of these points: even their
weighted mean can beat it. It is NOT a bound on arbitrary distributions,
continuous writers, multi-step policies, or the full rollout ensemble.
Local mixture mean here conditions on ONE current history; it does not
average the diverged histories of a trajectory ensemble.

Additional masks: original checker on the current middle point, and
destination consistency (`state_from_history(history+candidate)==r`).
The latter is a diagnostic restriction, NOT an official hard rule. Current
middle validity does not establish validity of the newly written point or
its future. Restricted oracle errors/gaps use matched supported rows; do
not compare a subset's mean directly with the all-row unrestricted mean.
No numeric-invalid candidates or failed rollout snapshots were observed.

## Findings

| Context | Fit expected MSE | Fit oracle MSE | Hold expected MSE | Hold oracle MSE |
|---|---|---|---|---|
|Observed history|2.34382e-6|1.52296e-6|2.20715e-6|1.43307e-6|
|Free offset24|.00840524|.00836698|.00267906|.00264028|
|Free offset100|.20394666|.20370314|.19869646|.19844989|

On observed histories privileged routing removes about35% of single-point
MSE, leaving~65% at the nearest center. This is MSE, not35% RMSE or a
learnable performance promise. Median nearest-center angular residual
norm is .001664/.001616rad fit/hold. Near-perfect oracle selection cannot
remove all local numerical residual, but this does not establish that the
residual is predictable rather than measurement/noise/representation error.

Only25.79%/27.60% of those unrestricted best centers are destination-q
consistent. Requiring consistency cuts matched expected-MSE improvement
to2.21%/2.40%; consistent support exists for99.89%/99.87% of observed rows.
The official mixture check instead has support99.29%/99.02% and adds only
3.15e-9/7.79e-10 MSE to its matched unrestricted oracle. These are distinct
constraints: do not blame the mixture threshold for the q-label tradeoff.

At offset100, changing only the next destination removes~0.119%/0.124%
of aligned single-point MSE, with median residual .406/.376rad. This says
late one-step rerouting cannot repair an already-diverged history. It does
NOT say early routing decisions do not matter, nor that F alone caused
the accumulated trajectory mismatch. Dynamics can follow its own state
while being far from the original video's state.

54 tests pass, including mixture variance identity, oracle-not-ensemble
bound counterexample, no routing gain for identical centers, exact free
history parity, restricted-oracle ordering and read purity. All groups'
variance identities and paired mask ordering also verified in full audit.

## Next bounded experiment

Measure whether small continuous residual corrections are PREDICTABLE from
causal history on fit TRAIN and transfer to the three TRAIN holdout videos.
Keep the event/q/F-center lineage and fixed-center control; inspect q
contradictions as well as point/distribution error. Do not infer from the
oracle that adding independent noise helps (previous experiments refuted
that simple version), or silently treat teacher-forced gains as free-run
improvements. No parameter or default-runtime changes in this audit.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python audit_writer_headroom.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m unittest test_writer_headroom test_training_horizons test_ensemble_objective test_particle_frontier test_read_cache test_commit_probe test_stateless_adapter test_training_window_sampler test_closed_loop_policy test_feedback_distribution test_adaptive_search -q
```
