"""Independent per-particle point/kernel mixture: anchored U is quadratic in alpha."""
import numpy as np


def terms_from_statistics(stats, kernel):
    change = kernel-1
    return dict(baseline=stats['baseline'].copy(),
                linear=np.sum(stats['coeff']*change*(stats['attraction']-stats['pair']), axis=(1, 2)),
                quadratic=-.5*np.sum(stats['coeff']*change**2*stats['pair'], axis=(1, 2)))


def value_gradient(terms, alpha):
    alpha = np.asarray(alpha, dtype=float)
    if not np.isfinite(alpha).all() or ((alpha < 0)|(alpha > 1)).any():
        raise ValueError('Finite mixture probability in [0,1] required')
    linear, quadratic = np.asarray(terms['linear']), np.asarray(terms['quadratic'])
    return np.asarray(terms['baseline'])+alpha*linear+alpha**2*quadratic, linear+2*alpha*quadratic


def exact_point_terms(blocked_result):
    """Use exact unsmoothed distances; alpha=1 equals RAW smoothed integral.

    This is deliberately not the old finite-grid zero-anchored correction.
    Smoothed terms remain spectral approximations; concentration limits still need checks.
    """
    return dict(baseline=blocked_result['baseline'], linear=blocked_result['exact_linear'],
                quadratic=blocked_result['exact_quadratic'])


def interval_grid_difference(first, second):
    """Exact maximum difference between two quadratics/derivatives on [0,1].

    This bounds grid-to-grid change, NOT true integration error.
    Both quadratics must share the exact point baseline.
    """
    np.testing.assert_array_equal(first['baseline'], second['baseline'])
    linear = np.asarray(first['linear'])-np.asarray(second['linear'])
    quadratic = np.asarray(first['quadratic'])-np.asarray(second['quadratic'])
    vertex = np.divide(-linear, 2*quadratic, out=np.zeros_like(linear, dtype=float), where=quadratic!=0)
    interior = np.where((vertex>0)&(vertex<1), np.abs(vertex*linear+vertex**2*quadratic), 0.)
    return dict(cost=np.maximum(np.abs(linear+quadratic), interior),
                gradient=np.maximum(np.abs(linear), np.abs(linear+2*quadratic)))
