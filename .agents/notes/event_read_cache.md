# Exact event-read reuse — 2026-09-17

Previous turn was progress: full output/counter parity and paired timing
showed viability-only Node removal saves only about1.6%. The prior profile
identified event reads as another substantial cost; this experiment tests
reusing them without weakening the event→q→F mechanism.

## Contract

`AdaptiveBeam(read_cache_size=4096)` stores pure frozen-machine reads in a
per-searcher LRU. Key: machine object identity, complete history shape/dtype/
bytes, current integer q, and complete hidden shape/dtype/bytes. The key
retains the machine object itself, avoiding numeric object-id reuse. Neither
event identity nor path score enters this reader's inputs; current supported
adapters are manifest-verified no-feedback models. Unlike F caching, hidden
MUST be included. All candidate event/q probabilities are still sampled or
ranked in exactly the usual order; cache hits do not commit hidden state.

Cached pe/T/read_hidden are owned read-only copies, bounded to4096 entries.
Node construction copies committed hidden as before. Default capacity0
leaves reads uncached. `clear_read_cache()` must be called after in-place
model/config changes; no automatic online-training invalidation is claimed.
If changing base feature/F parameters, construct a fresh searcher to avoid
stale entries in either cache. This experiment never changes model weights.

## Verification protocol

39 tests pass: equal copied inputs hit; changed q/history/hidden/shape/dtype
miss; returned arrays cannot be accidentally written and do not alias model
output buffers; LRU refresh/eviction; explicit invalidation; replaced model
identity; disabled/invalid capacities; prior search/checker/learning tests.

Full evaluation: original and all3 TRAIN-selected adapters ×3 rollout RNG
seeds,32 common DEV windows,8particles,301 generated/300 scored, terminal
right point checked. `probe_read` adds only read LRU to `probe_fast`.
Compare every prediction/failure/q/boundary bitwise and unchanged checker
audit; all search counters except read-hit/miss instrumentation must agree.
Hits+misses must equal expansion calls. No fitting or TEST use.

Separate alternating timing uses windows0/8/16/24, original policy,
seed1729,two repeats; exact output/audit/search-counter parity required.
Existing timing fields `cache_cpu`/`fast_cpu` mean reference/candidate;
the JSON also explicitly identifies `probe_fast`/`probe_read`.

Reproduce:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python evaluate_adapted_checked.py --mode probe_read --model-seed 0 --workers 3
# Repeat with1901,2718,3141.
python summarize_commit_probe.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python benchmark_fast_probe.py --read-cache
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m unittest test_read_cache test_commit_probe test_stateless_adapter test_training_window_sampler test_closed_loop_policy test_feedback_distribution test_adaptive_search -q
```

## Scope of the optimization pathway

Full evaluation completed with exact prediction/failure/q/boundary parity
for all12 model/rollout combinations. Unchanged checker audits and original
search counters also match. Original0/768 and adapter0/2304 endpoint failures;
all claimed valid points pass the unchanged check, including final point.

| Policy | Fast probe CPU/run | + read cache CPU/run | Read hit rate |
|---|---|---|---|
|Original|157.524|136.831|44.19%|
|Adapter1901|—|144.414|46.17%|
|Adapter2718|—|144.529|46.48%|
|Adapter3141|—|144.475|45.69%|
|Adapter family|172.276|144.473|—|

Family reduction16.14%; original13.14%. These compare separate full runs.
The alternating paired original-policy control totals38.906→34.463 CPU s,
11.42% lower, all8 pairs favoring caching. Timing includes unchanged output
verification. Repeated DEV videos are not independent accuracy evidence.

Relative to naive uncached probe205.631 family CPU, cumulative implemented
reuse/fast-path cost144.473 is~29.7% lower. This now has roughly the CPU cost
of the earlier unprobed sparse family145.984 while retaining the probe's
observed0 rather than9/2304 failures. This is a descriptive tradeoff across
recorded runs, not a universal completion or real-time latency guarantee.
1s error remains .138854 vs full-depth2 .121471; caching does not erase that
search-quality tradeoff. No changes to default flags or frozen package.

Exact reuse can reduce search overhead; it cannot improve predictive error.
Do not promote search simply because it gets faster: the original batched
checked runner is still a much cheaper baseline. A useful next system-level
comparison is allocating comparable CPU to more ordinary checked particles,
including the same adapters, before investing further in search caches.
Model-quality questions remain deployed-continuation value feedback and
temporally coherent learned residual distributions, not independent jitter.
