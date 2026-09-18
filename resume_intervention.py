"""Continue actual pre-read states, without reinitializing recurrent memory."""
import copy
import numpy as np
from feedback_distribution_pilot import sample


def resume(s, state, steps=250, root=None, seed=None, branches=1):
    if not isinstance(steps, (int, np.integer)) or steps < 1:
        raise ValueError('steps must be positive integer')
    if not isinstance(branches, (int, np.integer)) or branches < 1:
        raise ValueError('branches must be positive integer')
    if root is not None:
        root = np.asarray(root)
        if (root.ndim > 1 or (root.ndim == 1 and root.shape != (len(state['q']),))
                or not np.issubdtype(root.dtype, np.integer) or (root < 0).any() or (root >= 8).any()):
            raise ValueError('root must be an integer in [0,8) or one per state')
    if seed is None and branches != 1:
        raise ValueError('Exact RNG replay requires branches=1')
    h, q, hidden, dead = [np.repeat(state[k], branches, axis=0)
                           for k in ['history', 'q', 'hidden', 'failed']]
    n = len(state['q'])
    rng = np.random.default_rng(seed)
    if seed is None:
        rng.bit_generator.state = copy.deepcopy(state['rng_state'])
    predictions, failures = [], []
    for t in range(steps):
        pe, tr, read = s.machine.read(h, q, {'hidden': hidden})
        e = sample(pe, rng.random(len(h)))
        r = sample(tr[np.arange(len(h)), e], rng.random(len(h)))
        if t == 0 and root is not None:
            r = np.full(len(h), root, dtype=int) if root.ndim == 0 else np.repeat(root, branches)
        y = s.base.execute_rule(h, q, r)
        dead |= (~np.isfinite(y)).any(1) | (np.abs(y-h[:, -1]) > np.pi).any(1)
        y[dead] = h[dead, -1]
        predictions.append(y.copy()); failures.append(dead.copy())
        h = np.concatenate([h[:, 1:], y[:, None]], 1)
        q, hidden = r, read['read_hidden']
    return (np.stack(predictions, 1).reshape(n, branches, steps, 2),
            np.stack(failures, 1).reshape(n, branches, steps))


def influence_cost(x, truth, reference, failed):
    """Energy-score first variation, fixed independent reference bank.

    x[N,B,D], truth[N,D], reference[N,R,D], failed[N,B].
    Not a finite-mixture score or a per-particle squared error.
    """
    x, truth, reference, failed = map(np.asarray, (x, truth, reference, failed))
    if (x.ndim != 3 or reference.ndim != 3 or truth.shape != (x.shape[0], x.shape[2])
            or reference.shape[0] != x.shape[0] or reference.shape[2] != x.shape[2]
            or reference.shape[1] < 1 or failed.shape != x.shape[:2]):
        raise ValueError('Mismatched candidate/truth/reference/failure dimensions')
    return (np.linalg.norm(x-truth[:, None], axis=-1)
            - np.linalg.norm(x[:, :, None]-reference[:, None], axis=-1).mean(-1)
            + 2*failed).mean(-1)
