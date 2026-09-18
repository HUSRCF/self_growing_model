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


def kernel_moments(modes, log_kappa):
    kappa = np.exp(np.asarray(log_kappa))
    if kappa.shape != (2,) or not np.isfinite(kappa).all() or (kappa <= 0).any():
        raise ValueError('Two finite log concentrations required')
    order = np.abs(modes)
    values, derivatives = [], []
    for k in kappa:
        base = ive(0, k); ratio = ive(order, k)/base
        derivative = k*((ive(np.abs(order-1), k)+ive(order+1, k))/(2*base)-ratio*ive(1, k)/base)
        values.append(ratio); derivatives.append(derivative)
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
