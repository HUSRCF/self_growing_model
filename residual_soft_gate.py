"""Bounded causal correction to a frozen motion gate; train-only fallback."""
import numpy as np
from scipy.optimize import minimize


def predict(beta, x, base):
    return np.clip(base+.25*np.tanh(x@beta), 0, 1)


def objective(beta, x, base, linear, quadratic):
    t = np.tanh(x@beta)
    raw = base+.25*t
    a = np.clip(raw, 0, 1)
    d = (linear+2*quadratic*a)*((raw>0)&(raw<1))*.25*(1-t*t)
    return float(np.mean(linear*a+quadratic*a*a)+.001*np.sum(beta*beta)), x.T@d/len(x)+.002*beta


def fit_gate(x, base, linear, quadratic):
    initial = np.zeros(x.shape[1])
    baseline = float(np.mean(linear*base+quadratic*base*base))
    result = minimize(objective, initial, args=(x, base, linear, quadratic), jac=True,
                      method='L-BFGS-B', bounds=[(-2, 2)]*len(initial),
                      options=dict(maxiter=200, gtol=1e-9, ftol=1e-12))
    return dict(beta=result.x.tolist(), success=bool(result.success), message=str(result.message),
                use_residual=bool(result.success and result.fun<baseline), base_loss=baseline,
                candidate_loss=float(result.fun), iterations=int(result.nit))


def predict_gate(model, x, base):
    return predict(np.asarray(model['beta']), x, base) if model['use_residual'] else np.asarray(base).copy()


def block_starts(length, block, count=4, horizon=300):
    midpoint, boundary = length//2, 3*length//4
    if block == 'train':
        lo, hi = midpoint+31, boundary-horizon-1
    elif block == 'evaluation':
        lo, hi = boundary+31, length-horizon-1
    else:
        raise ValueError('Unknown temporal block')
    if hi-lo+1 < count:
        raise ValueError('Insufficient block length')
    return np.linspace(lo, hi, count, dtype=int)
