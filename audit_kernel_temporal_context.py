"""Descriptive causal-history motion versus frozen-kernel loss; no fitting."""
import json
from pathlib import Path
import numpy as np
from train_closed_loop_policy import prefix_windows
from v20_rnn_mixture.engine.data import tail_windows
from v20_rnn_mixture.engine.common import SPLITS


def main():
    root = Path('adaptive_search_results')
    out = {}
    for region, filename in [('prefix', 'energy_output_kernel_confirmation.json'),
                             ('tail', 'energy_kernel_temporal_shift.json')]:
        report = json.loads((root/filename).read_text())
        if region == 'prefix':
            w = prefix_windows(SPLITS['train'][-3:], steps=300, per_video=8)
        else:
            full = tail_windows('train', 300, 8)
            mask = np.isin(full['video'], SPLITS['train'][-3:])
            w = {key: value[mask] for key, value in full.items()}
        np.testing.assert_array_equal(w['video'], report['runs'][0]['video'])
        np.testing.assert_array_equal(w['start'], report['runs'][0]['start'])
        motion = np.sqrt(np.mean(np.diff(w['history'], axis=1)**2, axis=(1, 2)))
        delta = np.mean([np.array(r['results']['4097']['energy']['cost'])-
                         r['results']['4097']['zero'] for r in report['runs']], axis=0).mean(1)
        out[region] = dict(motion_rms_per_window=motion.tolist(), seed_mean_delta_per_window=delta.tolist(),
                           video=w['video'].tolist(), start=w['start'].tolist(),
                           motion_median=float(np.median(motion)),
                           correlation_log_motion_delta=float(np.corrcoef(np.log(motion+1e-12), delta)[0, 1]),
                           per_video_motion_median={str(v): float(np.median(motion[w['video'] == v])) for v in np.unique(w['video'])})
    out['note'] = ('Causal history31 increments RMS rad/sample only,all24windows eachregion; '
                   'seedmean loss not independent192observations. Descriptive association,not causal evidence '
                   'or fitted gate. No threshold/kernel fitting,DEV/TEST or promotion.')
    (root/'energy_kernel_temporal_context.json').write_text(json.dumps(out, indent=2))
    print(json.dumps({key: {k: v for k, v in row.items() if k in ['motion_median', 'correlation_log_motion_delta', 'per_video_motion_median']}
                      for key, row in out.items() if isinstance(row, dict)}, indent=2))


if __name__ == '__main__':
    main()
