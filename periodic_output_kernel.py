"""Zero-center circular output smoothing, separate from frozen dynamics."""
import numpy as np
from scipy.optimize import minimize
from scipy.special import i0e, i1e, logsumexp


def nll_gradient(log_kappa, centers, truth):
    kappa = np.exp(log_kappa)
    cosine = np.cos(truth[:, None]-centers)
    log_kernel = ((cosine-1)*kappa-np.log(i0e(kappa))-np.log(2*np.pi)).sum(-1)
    normalizer = logsumexp(log_kernel, axis=1)
    responsibilities = np.exp(log_kernel-normalizer[:, None])
    loss = float(np.mean(np.log(centers.shape[1])-normalizer))
    gradient = kappa*(i1e(kappa)/i0e(kappa)-(responsibilities[:, :, None]*cosine).sum(1).mean(0))
    return loss, gradient


def fit_kernel(centers, truth):
    """One predeclared optimization start; finite-particle mixture likelihood."""
    if centers.ndim != 3 or centers.shape[-1] != 2 or truth.shape != (len(centers), 2):
        raise ValueError('Expected centers[N,P,2], truth[N,2]')
    if not np.isfinite(centers).all() or not np.isfinite(truth).all():
        raise ValueError('Finite angles required')
    result = minimize(nll_gradient, np.log([10., 10.]), args=(centers, truth), jac=True,
                      method='L-BFGS-B', bounds=[(np.log(1e-4), np.log(1e6))]*2,
                      options=dict(maxiter=200, gtol=1e-7, ftol=1e-12))
    if not result.success:
        raise RuntimeError('Kernel optimization failed: '+str(result.message))
    return dict(kappa=np.exp(result.x).tolist(), nll=float(result.fun), gradient=result.jac.tolist(),
                iterations=int(result.nit), message=str(result.message),
                at_boundary=((result.x < np.log(1e-4)+1e-6)|(result.x > np.log(1e6)-1e-6)).tolist())


def convolve(points, kappa, seed):
    """One independent circular perturbation per base particle/horizon/angle."""
    if kappa is None:
        return points.copy()
    kappa = np.asarray(kappa, float)
    if points.ndim != 4 or points.shape[-2:] != (3, 2) or kappa.shape != (3, 2):
        raise ValueError('Expected points[N,P,3,2], kappa[3,2]')
    if not np.isfinite(kappa).all() or (kappa <= 0).any():
        raise ValueError('Finite positive concentrations required')
    rng = np.random.default_rng(seed); output = points.copy()
    for t in range(3):
        output[:, :, t] += rng.vonmises(0., kappa[t], size=points[:, :, t].shape)
    return output
