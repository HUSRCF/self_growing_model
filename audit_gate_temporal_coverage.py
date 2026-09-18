"""History-only temporal shift audit on the ten fitting videos; no gate fitting."""
import hashlib
import json
from pathlib import Path
import numpy as np
from adaptive_search_prototype import AdaptiveBeam
from training_window_sampler import TrainingPrefixPool
from validate_soft_gates_new_windows import causal_features
from validate_full_soft_gates import weights
from feature_soft_kernel_gate import design
from v20_rnn_mixture.engine.common import SPLITS
from v20_rnn_mixture.engine.data import load_video


def region_starts(length, region, excluded=(), count=64, horizon=300):
    if region == 'prefix':
        candidates = np.arange(63, length//2-horizon)
    elif region == 'tail':
        # Entire history, not just target, belongs to the second half.
        candidates = np.arange(length//2+31, length-horizon)
    else:
        raise ValueError('Unknown region')
    candidates = candidates[~np.isin(candidates, list(excluded))]
    if len(candidates) < count:
        raise ValueError('Insufficient unique starts')
    return candidates[np.linspace(0, len(candidates)-1, count, dtype=int)]


def nearest_other_video(query, reference, video, reference_video):
    distance = np.sqrt(np.mean((query[:, None]-reference[None])**2, axis=-1))
    distance[np.asarray(video)[:, None] == np.asarray(reference_video)[None]] = np.inf
    result = distance.min(axis=1)
    if not np.isfinite(result).all():
        raise ValueError('Reference must contain another video')
    return result


def quantiles(x):
    return np.quantile(x, [0, .25, .5, .75, 1], axis=0).tolist()


def main():
    root = Path('adaptive_search_results')
    paths = [root/'full_soft_gate_model.json', Path('v20_rnn_mixture/models/frozen_dynamics.json'),
             Path('v20_rnn_mixture/models/gru_1901.npz')]
    hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    model = json.loads(paths[0].read_text())['model']
    engine = AdaptiveBeam()
    fit = TrainingPrefixPool(engine.base, steps=300).sample(481017, per_video=8)
    motion, raw = causal_features(engine, fit)
    mapping = model['mapping']
    np.testing.assert_array_equal(raw.mean(0), mapping['mean'])
    np.testing.assert_array_equal(np.maximum(raw.std(0), 1e-5), mapping['scale'])
    mean, scale = np.asarray(mapping['mean']), np.asarray(mapping['scale'])
    reference_z = np.clip((raw-mean)/scale, -3, 3)
    reference_x = design(mapping, raw)[:, 1:]
    rows = {}
    for region in ['prefix', 'tail']:
        histories, videos, starts = [], [], []
        for video in SPLITS['train'][:-3]:
            y = load_video(video)
            chosen = region_starts(len(y), region, fit['start'][fit['video'] == video])
            histories.extend(y[t-31:t+1] for t in chosen)
            videos.extend([video]*len(chosen)); starts.extend(chosen.tolist())
        w = dict(history=np.asarray(histories), video=np.asarray(videos), start=np.asarray(starts))
        m, r = causal_features(engine, w)
        z = (r-mean)/scale
        q, _ = engine.machine.initialize(w['history'])
        alpha = weights(model, m, r)['features']
        distances = dict(raw_clipped=nearest_other_video(np.clip(z, -3, 3), reference_z, videos, fit['video']),
                         projected=nearest_other_video(design(mapping, r)[:, 1:], reference_x, videos, fit['video']))
        def summarize(mask):
            return dict(count=int(mask.sum()), motion_quantiles=quantiles(m[mask]),
                        low_motion_fraction=float((m[mask] < model['kernel']['threshold']).mean()),
                        clipped_coordinate_fraction=float((np.abs(z[mask]) > 3).mean()),
                        clipped_window_fraction=float((np.abs(z[mask]) > 3).any(1).mean()),
                        q_counts=np.bincount(q[mask], minlength=8).tolist(),
                        alpha_mean=alpha[mask].mean(0).tolist(),
                        alpha_quantiles=quantiles(alpha[mask]),
                        other_video_nn_quantiles={k: quantiles(v[mask]) for k, v in distances.items()})
        rows[region] = dict(summary=summarize(np.ones(len(videos), bool)),
                            per_video={str(v): summarize(w['video'] == v) for v in np.unique(videos)},
                            video=videos, start=starts, motion=m.tolist(), alpha=alpha.tolist(),
                            nearest={k: v.tolist() for k, v in distances.items()})
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest() == h for p, h in hashes.items())
    report = dict(regions=rows, source_hashes=hashes,
                  fit_motion_quantiles=quantiles(motion),
                  note='TRAIN fitting10 only; 64 deterministic unique starts/video/region, excluding original80. '
                       'Tail histories wholly after midpoint. No target extraction, rollout, loss labels, fitting or hold/DEV/TEST. '
                       'NN to original80 excludes same video; raw distance after frozen clip and projected distance both descriptive. '
                       'Within-region windows overlap; no independent-sample inference. Frozen mapping/source hashes verified.')
    (root/'gate_temporal_coverage.json').write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v['summary'] for k, v in rows.items()}, indent=2))


if __name__ == '__main__':
    main()
