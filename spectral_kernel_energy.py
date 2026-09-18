"""Zero-anchored Fourier approximation of expected circular-kernel U-energy."""
from functools import lru_cache
import numpy as np
from scipy.special import ive
from ensemble_score_objective import energy_costs
from audit_energy_action_value import embedding


@lru_cache(maxsize=8)
def distance_spectrum(size):
    if size < 3 or size % 2 != 1:
        raise ValueError('Odd grid size >=3 required')
    angle = 2*np.pi*np.arange(size)/size
    distance = np.sqrt(np.maximum(4-2*np.cos(angle[:, None])-2*np.cos(angle[None]), 0))
    coeff = np.fft.fft2(distance)/size**2
    np.testing.assert_allclose(coeff.imag, 0, atol=1e-14)
    return np.rint(np.fft.fftfreq(size)*size).astype(int), coeff.real


def statistics(centers, truth, failed, size):
    n, p, d = centers.shape
    if d != 2 or p < 3 or truth.shape != (n, 2) or failed.shape != (n, p):
        raise ValueError('Expected centers[N,P>=3,2] and matching truth/failed')
    modes, coeff = distance_spectrum(size)
    a = np.exp(1j*centers[:, :, 0, None]*modes)
    b = np.exp(1j*centers[:, :, 1, None]*modes)
    characteristic = np.einsum('npi,npj->nij', a, b, optimize=True)/p
    target = np.exp(-1j*truth[:, 0, None]*modes)[:, :, None]*np.exp(-1j*truth[:, 1, None]*modes)[:, None, :]
    attraction = (characteristic*target).real
    # Distinct original particles only; self-pairs are NOT smoothed copies.
    pair = (p*np.abs(characteristic)**2-1)/(p-1)
    baseline = energy_costs(embedding(centers), embedding(truth), failed)[0]
    return dict(modes=modes, coeff=coeff, attraction=attraction, pair=pair,
                baseline=baseline, penalty=2*failed.mean(1))


def spectral_cost(stats, kernel):
    return np.sum(stats['coeff']*(kernel*stats['attraction']-.5*kernel**2*stats['pair']), axis=(1, 2))+stats['penalty']


def moment_factors(modes, log_kappa):
    kappa = np.exp(np.asarray(log_kappa))
    if kappa.shape != (2,) or not np.isfinite(kappa).all() or (kappa <= 0).any():
        raise ValueError('Two finite log concentrations required')
    order = np.abs(modes)
    values, derivatives = [], []
    for k in kappa:
        base = ive(0, k); ratio = ive(order, k)/base
        derivative = k*((ive(np.abs(order-1), k)+ive(order+1, k))/(2*base)-ratio*ive(1, k)/base)
        values.append(ratio); derivatives.append(derivative)
    return values, derivatives


def kernel_moments(modes, log_kappa):
    values, derivatives = moment_factors(modes, log_kappa)
    kernel = values[0][:, None]*values[1][None, :]
    gradient = np.stack([derivatives[0][:, None]*values[1][None], values[0][:, None]*derivatives[1][None]])
    return kernel, gradient


def value_gradient(stats, log_kappa=None):
    if log_kappa is None:
        return stats['baseline'].copy(), np.zeros((len(stats['baseline']), 2))
    kernel, derivative = kernel_moments(stats['modes'], log_kappa)
    # Anchor the no-noise endpoint to its exact original score, not its Fourier approximation.
    change = np.sum(stats['coeff']*((kernel-1)*stats['attraction']-.5*(kernel**2-1)*stats['pair']), axis=(1, 2))
    integrand = stats['coeff']*(stats['attraction']-kernel*stats['pair'])
    gradient = np.einsum('nij,kij->nk', integrand, derivative)
    return stats['baseline']+change, gradient


def blocked_value(centers, truth, failed, size, log_kappa, block=128, mixture=False):
    """One window, row-blocked full spectrum; no frequency truncation."""
    if centers.ndim != 2 or centers.shape[1] != 2 or len(centers) < 3:
        raise ValueError('Expected centers[P>=3,2]')
    if truth.shape != (2,) or failed.shape != (len(centers),) or block < 1:
        raise ValueError('Invalid truth/failed/block')
    baseline = float(energy_costs(embedding(centers[None]), embedding(truth[None]), failed[None])[0][0])
    if log_kappa is None:
        result = dict(cost=baseline, gradient=np.zeros(2), baseline=baseline)
        if mixture:
            result.update(linear=0., quadratic=0., exact_linear=0., exact_quadratic=0.)
        return result
    modes, coeff = distance_spectrum(size)
    values, derivatives = moment_factors(modes, log_kappa)
    a = np.exp(1j*centers[:, 0, None]*modes)
    b = np.exp(1j*centers[:, 1, None]*modes)
    target0, target1 = np.exp(-1j*truth[0]*modes), np.exp(-1j*truth[1]*modes)
    raw, point, gradient = 0., 0., np.zeros(2)
    linear, quadratic = 0., 0.
    point_attraction, point_pair = 0., 0.
    for start in range(0, size, block):
        sl = slice(start, min(start+block, size))
        characteristic = a[:, sl].T@b/len(centers)
        attraction = (characteristic*target0[sl, None]*target1[None]).real
        pair = (len(centers)*np.abs(characteristic)**2-1)/(len(centers)-1)
        kernel = values[0][sl, None]*values[1][None]
        raw += np.sum(coeff[sl]*(kernel*attraction-.5*kernel**2*pair))
        point += np.sum(coeff[sl]*(attraction-.5*pair))
        integrand = coeff[sl]*(attraction-kernel*pair)
        gradient[0] += np.sum(integrand*derivatives[0][sl, None]*values[1][None])
        gradient[1] += np.sum(integrand*values[0][sl, None]*derivatives[1][None])
        if mixture:
            change = kernel-1
            linear += np.sum(coeff[sl]*change*(attraction-pair))
            quadratic -= .5*np.sum(coeff[sl]*change**2*pair)
            point_attraction += np.sum(coeff[sl]*attraction)
            point_pair += np.sum(coeff[sl]*pair)
    penalty = 2*failed.mean()
    result = dict(cost=float(baseline+raw-point), raw_cost=float(raw+penalty),
                  baseline_error=float(point+penalty-baseline), gradient=gradient, baseline=baseline)
    if mixture:
        result.update(linear=float(linear), quadratic=float(quadratic))
        exact_attraction = float(np.linalg.norm(embedding(centers)-embedding(truth), axis=-1).mean())
        exact_pair = 2*(exact_attraction+penalty-baseline)
        attraction_error, pair_error = point_attraction-exact_attraction, point_pair-exact_pair
        result.update(exact_linear=float(linear+attraction_error-pair_error),
                      exact_quadratic=float(quadratic+.5*pair_error))
    return result
