# Self-growing algorithm: durable memory

Last updated: 2026-09-17 UTC.

User preference: always respond in Chinese. The active scope and working contract are in `TASK.md`.

## Workspace inventory

| Version | Archive | SHA-256 | Review state |
|---|---|---|---|
| v17 | `self_growing_algorithm_v17_complete.zip` | `31f54bc732cf06698c1225ef4ca8ac268d7df011d359fd2a90b2ddc66dfe30ef` | Extracted to `v17_high_order_local_rules/`; read and summarized; ZIP CRC passes |
| v18 | `self_growing_algorithm_v18_complete.zip` | `a44610f5e556b6955b6971499c151a3e7b4f51b26de1ad2ec96a1e7c64a78712` | Extracted to `v18_event_dynamics/`; read and summarized |
| v20 | `self_growing_algorithm_v20_complete.zip` | `75272f3de7bf60a3fbfc44e993b1f49a994ef14f8f6aa530b7257abd26d58aa7` | ZIP had no enclosing directory; extracted to `v20_complete/`; read and summarized, including embedded v19 bridge |
| v20-rnn-mixture | `v20_rnn_mixture_complete.zip` | `8a9a8099bc9aff664d77437d79ffc0fe9a99e21864a39a978054c414004ce721` | Extracted to `v20_rnn_mixture/`; read and summarized; ZIP CRC passes |

There is no standalone v19 archive in the folder as of this update. The v20 package contains `v19/` and `v19_models/`; their role must be established from v20's own documentation rather than inferred from filenames.

## Current understanding through v17

The project predicts two double-pendulum angles with an event-driven finite-state outer machine. At each step it reads an event distribution from causal history and current register `q`, samples an event, lets that event select next register `r`, executes the selected local numerical rule `F(q,r)`, and only then commits the new history. The rule may have internal memory, but it must not replace or back-solve the outer event/register sequence.

v17 tested two ideas: preserving more high-order derivative information in the 8-state `q` partition, and replacing/adapting the local `F(q,r)` rule with MLP or HMM material. The durable result is asymmetric:

- q-pair-adapted MLP rules helped, especially as a 50/50 blend with the old polynomial rule.
- Increasing the role of high-order derivatives fixed a real representation-compression issue, but the high-order partition alone did not generalize on the reused test set.
- The tested local HMM implementation failed badly and developed numerical failures over long rollouts.
- Point prediction improved without solving state/history contradictions or uncertainty calibration.

The frozen v17 primary is `q_rms17_mlp_poly_blend0.5`; the best overall development control is `q_old_mlp_poly_blend0.5`. See `versions/v17.md` for evidence and caveats.

## Cross-version picture through v20

- v17: improved the local numerical writer. Pair-adapted MLP plus a 50/50 polynomial blend worked; high-order q alone did not generalize and HMM failed.
- v18: added explicit event time state and tested seven event mechanisms. Better causal history routing helped, but event memory itself had no stable independent test benefit.
- v19, preserved inside v20: produced the largest gain by carrying velocity explicitly, learning its initialization, and using an integral-style writer. Its strong continuous controller largely bypassed the event variable.
- v20: forced continuous state and path information back through a finite event bottleneck and repaired exposure mismatch. Events became operationally necessary, but q-switch classification weakened and prediction traded short-horizon quality for selected medium/long-horizon gains.

The central unresolved interface is now precise: jointly make event codes distinguishable, q transitions predictable, and frozen/local writers capable of realizing the selected q without increasing register/history contradiction.

The separate v20-rnn-mixture package replaces leaf routing with a trained 8-event GRU and adds rollback through a q-conditional velocity-mixture checker. The architecture and audit succeed, but the frozen GRU+mixture is not a predictive upgrade: 3 s embedding RMSE is 0.495692 versus 0.470046 for the old leaf+mixture baseline. Resetting GRU hidden each step reaches 0.438309 at 3 s but loses 1 s quality, q matching and efficiency; it is a diagnostic, not a frozen winner. See `versions/v20_rnn_mixture.md`.

Latest repository scan is recorded in `notes/repository_scan_2026-09-17.md`. The package's saved mechanism audit passes, and the current shell has now rerun all 16 mechanism tests successfully.

An isolated `adaptive_search_prototype.py` now tests AlphaGo-style variable-depth event-first beam lookahead without changing the frozen package. On the 32-window development run (2 particles, 50 steps), fixed depth 1/2 had 50-step embedding RMSE 0.04109/0.03514; adaptive min-depth 2, max-depth 3 averaged depth 2.29 and gave 0.03535 with zero failures. This is an exploratory greedy-beam result, not yet a fair replacement for the official stochastic baseline or a trained value-guided search.

The prototype was then optimized by batching candidate F/checker/q operations per node and deduplicating deterministic temperature-0 particles. The same 64-window test run dropped from 225.9 s to 31.4 s (~7.2×) with identical metrics; v20-rnn-mixture's 16 live mechanism tests still pass.

## Active queue

Latest particle-search study: parallel window execution verified exactly equal to serial. On expanded32 DEV windows,3seeds8particles, fixed2+rollback temperature1 gives RMSE .032421/.113978/.394924 at .5/1/3s, zero failure; original8 gives .033691/.144262/.398124. Search needs90–105s with8workers vs original8~5s single-worker. Original32 particles gives3s .389295, energy .458394, coverage .429688 in18–23s, beating search on long-horizon quality/cost but not1s. No broad winner; next prioritize sparse search and shared computation to retain short/medium benefit. See notes/particle_search_diversity.md. Test set untouched this round.

Latest common-window300 DEV experiment (16 windows): fixed2 .46592 RMSE with2/16 failures; fixed3 .426047 with0 failures,92.7s; adaptive2–3 .46471 with2 failures; simple distinct-q-path support guard unchanged. New opt-in OFFLINE rollback (default budget0) with fixed2 made only2 rewinds, completed16/16, RMSE .413622 in42.5s, but fixed3 remains better at .5/1s. Matched original8-particle3-seed baseline .346814 at3s remains stronger; not equal compute or particle semantics. 5 prototype tests pass. See notes/common_window_long_rollouts.md. Next: larger dev paired evaluation and uncertainty diversity, then policy-aligned value labels.

Latest optimization checkpoint: root middle checking is now enforced across replanning (observed initial middle excepted); 3 prototype tests pass. This exposes a fixed-depth2 dead-end in dev window30 step17; full widening cannot recover it. Fixed-depth3 with the same check avoids failure and reaches dev50 RMSE .029683 (32/32 completion), versus checked-depth2 .186045 with1/32 failure. This revises earlier "depth adds no benefit" interpretation: prior adaptive stopping can stop too early. Train-only paired continuation audit: 51/114 candidate pairs reverse ranking between greedy and checked-depth2 rollout; successful-only9/23 reverse. Details `notes/continuation_policy_audit.md`. Next: common-window longer-horizon viability-aware stopping, then aligned value labels. Research goal remains active.

Remote initialized: https://github.com/HUSRCF/self_growing_model, default branch master, research branch shuang; local tracks origin/shuang. GitHub connector provides writes; local CLI currently supports only public fetch (no HTTPS credentials / SSH auth). See notes/candidate_ranking.md.

Follow-up pairwise ranker: 3744 train-prefix candidate states at rollout drift0/8/24, train-video holdout selection cost improves5.7%, but independent dev-tail candidate cost worsens3.2% (.004593 vs .004451). Do not enable it in search. No test-set ranker tuning/evaluation. Code train_search_ranker.py and results/ranker JSON summaries; full details notes/candidate_ranking.md.

Critical follow-up diagnosis: prototype passes a history ending at the middle point to the checker's left_history argument, computing (right-middle)/(2dt), not the calibrated central velocity. Both scalar and batched prototype paths are affected. Existing prototype accuracy results do not establish whether adaptive depth works. Fix and rerun before judging constraints/value/search. Test failure 1.65625% was time-averaged; endpoint failure at 50 steps was 3.125% (2/64 windows).

This interface bug is now fixed and covered by two prototype-specific tests. Corrected dev fixed2/adaptive2–3 RMSE at 0.5 s: .035752/.035754, zero failures. Corrected reused-test 64-window scores: .092618/.092649, zero failures (previous 2/64 failures gone). Full candidate widening rescued only 4/4875 dead internal nodes on dev and did not change predictions; soft-check dev RMSE worsened to .039288. A small train-prefix-only value regressor failed whole-training-video heldout MSE baseline (correlation .089), so was not enabled. Details and next research priorities: `notes/corrected_search_experiments.md`.

1. If implementation work is requested, choose a target metric and lineage first: old q remains stronger at short horizons; high q v20 has its clearest benefit at 3 s.
2. Treat v19 as an embedded reference baseline, not as a missing-evidence gap; there is no standalone v19 archive here, but v20 includes its report, freeze and summary.
3. A sensible next research target is constrained multi-step co-training of event partition, transition controller and writer realizability, with contradiction and calibration included in selection rather than treated only as diagnostics.
