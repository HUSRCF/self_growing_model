"""Small clipped-affine gate with exact scalar control for finite-P quadratic U."""
import numpy as np
from scipy.optimize import minimize


def scalar_optimum(linear, quadratic):
    l, q = float(np.mean(linear)), float(np.mean(quadratic))
    candidates = [0., 1.]
    if q > 0:
        candidates.append(float(np.clip(-l/(2*q), 0, 1)))
    return min(candidates, key=lambda a: l*a+q*a*a)


def objective(beta, z, linear, quadratic, penalty=.001):
    raw = beta[0]+beta[1]*z
    alpha = np.clip(raw, 0, 1)
    active = (raw > 0)&(raw < 1)
    derivative = (linear+2*quadratic*alpha)*active
    value = np.mean(linear*alpha+quadratic*alpha**2)+penalty*beta[1]**2
    gradient = np.array([derivative.mean(), np.mean(derivative*z)+2*penalty*beta[1]])
    return float(value), gradient


def fit_gate(log_motion, linear, quadratic):
    x, l, q = [np.asarray(a, dtype=float) for a in [log_motion, linear, quadratic]]
    if x.ndim != 1 or x.shape!=l.shape or x.shape!=q.shape or not len(x) or not all(np.isfinite(a).all() for a in [x,l,q]):
        raise ValueError('Matching nonempty finite vectors required')
    mean, scale = float(x.mean()), max(float(x.std()), 1e-6)
    z = np.clip((x-mean)/scale, -3, 3)
    scalar = scalar_optimum(l, q); scalar_loss = float(np.mean(l*scalar+q*scalar**2))
    result = minimize(objective, [.5, 0.], args=(z,l,q), jac=True, method='L-BFGS-B',
                      bounds=[(-2.,3.),(-2.,2.)], options=dict(maxiter=200,gtol=1e-9,ftol=1e-12))
    use_affine = bool(result.success and result.fun < scalar_loss)
    return dict(mean=mean, scale=scale, scalar=scalar, scalar_loss=scalar_loss,
                beta=result.x.tolist(), candidate_loss=float(result.fun), use_affine=use_affine,
                success=bool(result.success), message=str(result.message), iterations=int(result.nit),
                note='alpha=clip(b0+b1*clip(z,-3,3),0,1),penalty .001*b1²;fit-only scalar fallback')


def predict_gate(model, log_motion):
    x = np.asarray(log_motion, dtype=float)
    if not np.isfinite(x).all():
        raise ValueError('Finite causal feature required')
    if not model['use_affine']:
        return np.full(x.shape, model['scalar'], dtype=float)
    z = np.clip((x-model['mean'])/model['scale'], -3, 3)
    return np.clip(model['beta'][0]+model['beta'][1]*z,0,1)
