# Ordinary particles versus search compute — 2026-09-17

Previous turn was progress: exact hidden-aware read caching saved11.4% in
paired timing and~16% adapter-family full-run CPU. This turn compares a
system-level alternative rather than assuming faster search is worthwhile.

## Protocol and interpretation

Unchanged batched CheckedMachine at8/32/128/256 particles; original plus
all3 TRAIN-selected no-feedback adapters,3 rollout seeds each. Same32DEV
histories,300 returned points with terminal right boundary, unchanged
independent central checker validation. No training or TEST use. The
8-particle rerun is bitwise identical to prior checked outputs for all
models/seeds, including q and boundary, not just headline metrics.

Search references: read-cached viability probe and every-step full depth2,
both8particles, same strict endpoint protocol. Compare measured CPU, not
nominal particle counts or parallel wall time. Ordinary checker has
budget_factor10 and no two-step revision-distance constraint; search has
budget300 and revisionwindow2. Thus this is an OFFLINE compute/quality
frontier, not an equal-latency or identical-feasibility-policy experiment.
RNG streams are not nested when P changes. All optimizer seeds retained;
same four DEV videos reused, no independent replication/significance claim.

Inference CPU includes initialization, built-in diagnostics and independent
checker verification, excluding offline metrics and NPZ compression, as in
the previous study. Verification chunks four windows to bound memory.
Report rollout CPU separately in each raw JSON. The128/256 runs are not
automatically equal-budget to search; use actual costs to interpret them.

## Scoring details

Only reported horizons10/25/50/100/300 are passed into the original metrics
function; exact all-pairs energy is retained. This avoids computing an
O(P²) energy score for295 unused intermediate steps. Full-vs-selected
metrics agree to1e-14, including failure-conditioned scores in unit tests.
No random pair approximation. Chunked versus whole-check audits agree.

The original empirical energy V-statistic includes zero-distance self
pairs; its expectation changes with P even for the same generating law.
The summary additionally reports off-diagonal energy:
`U=(P*V-mean_distance_to_truth)/(P-1)`.
This removes self-pair bias. Calling it an unbiased population estimate
requires IID particles; that assumption is not proven for the shared
RNG/rollback implementation. It is a sensitivity check, not a blanket
calibration claim. Larger empirical quantile coverage can likewise reflect
more particles rather than a better learned uncertainty model.
The V-statistic is still the correct energy score for the actual finite
empirical forecast being delivered; it is not a bug. U asks a different
question about the generating law. A ranking reversal between V and U
must be reported rather than selecting whichever favors the new method.

42 tests pass. Summary independently verifies window/truth alignment,
prediction shape, failure monotonicity, and every reported RMSE against
saved trajectories, then derives per-model/family/video scores. It does
not refit or choose a best optimizer on DEV.

## Completed compute frontier

Adapter family means over all3 optimizer ×3 rollout seeds:

| Runtime | CPU s/run | .5s RMSE | 1s RMSE | 3s RMSE | Endpoint failures |
|---|---|---|---|---|---|
|Checked8|4.160|.033520|.142024|.395217|0/2304|
|Checked32|16.605|.033276|.141555|.388969|0/9216|
|Checked128|67.817|.033340|.137972|.384376|0/36864|
|Checked256|134.645|.033488|.141750|.384686|0/73728|
|Cached probe8|144.473|.034029|.138854|.387192|0/2304|
|Full depth2 search8|369.405|.033657|.121471|.389235|1/2304|

Checked128 costs53.1% less CPU than probe and its averaged three RMSEs
are lower (1s only~0.64% lower). This is NOT broad dominance: revision
semantics differ; at3s checked128 improves videos3/8 but worsens6/9,
and at1s improves6/9 but worsens3/8. The energy comparison below reverses
under diagonal correction.256 does not improve headline errors over128;
three non-nested RNG seeds do not establish a monotone trend or optimal P.
Do not set a new default from this reused DEV sweep.

Original-policy control remains important: checked128 CPU62.128s and
.033428/.137760/.383130 vs original probe CPU136.831 and
.035507/.130582/.388095. Search still improves original-policy1s error;
full original depth2 reaches .119322 at1s, CPU331.901. The adapter does
not beat original checked128 on average1/3s, so earlier low-P adapter
improvement is not a robust justification for deployment either.

## Distribution scoring reversal

For the adapter family at3s, checked128 has empirical energy .448139 versus
probe8 .453037 (lower is better), but off-diagonal energy is .446207 versus
.420567, reversing the order. Thus a CPU/RMSE comparison alone does not
establish superior generating-distribution quality. V scores the finite
forecast actually delivered; U removes its own-particle pair terms, with
the sampling assumptions and variance caveats above. Do not claim universal
dominance by choosing just one of these metrics.

This also makes a concrete next model experiment relevant: earlier on-policy
adapters optimize per-particle MSE, not a proper ensemble-distribution score.
Compare an ensemble-energy training objective at fixed TRAIN rollout budget,
with TRAIN-video holdout and all optimizer seeds, before more unchecked
Gaussian noise. A policy-gradient baseline must exclude the sampled
particle and ALL pair terms involving it; naively using the other particles'
coupled energy contributions as an independent baseline would be wrong.
This is a queued experiment, not implemented or validated by this study.

Reproduce:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python evaluate_checked_particles.py --particles 8 --model-seed 0
# Repeat P=32,128,256; model seeds0,1901,2718,3141.
python summarize_particle_frontier.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m unittest test_particle_frontier test_read_cache test_commit_probe test_stateless_adapter test_training_window_sampler test_closed_loop_policy test_feedback_distribution test_adaptive_search -q
```
