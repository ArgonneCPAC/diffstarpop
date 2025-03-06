"""
"""

import numpy as np
from jax import grad
from jax import jit as jjit
from jax import random as jran
from jax import vmap
from scipy.stats import random_correlation

from .. import utils as ut


def _enforce_is_cov(matrix):
    det = np.linalg.det(matrix)
    assert det.shape == ()
    assert det > 0
    covinv = np.linalg.inv(matrix)
    assert np.all(np.isfinite(covinv))
    assert np.all(np.isreal(covinv))
    assert np.allclose(matrix, matrix.T)
    evals, evecs = np.linalg.eigh(matrix)
    assert np.all(evals > 0)


def test_get_t50_p50_evaluates():
    nt = 50
    t_table = np.linspace(0.1, 13.7, nt)
    ngals = 150
    ran_key = jran.PRNGKey(0)
    histories = jran.uniform(ran_key, minval=0, maxval=10, shape=(ngals, nt))
    threshold = 0.5
    logmpeak = histories[:, -1]
    try:
        t50, p50 = ut.get_t50_p50(t_table, 10**histories, threshold, logmpeak)
        assert t50.shape == (ngals,)
        assert p50.shape == (ngals,)
    except ImportError:
        pass


def test_correlation_from_covariance():
    ntests = 100
    for __ in range(ntests):
        ndim = np.random.randint(2, 10)
        evals = np.sort(np.random.uniform(0, 100, ndim))
        evals = ndim * evals / evals.sum()
        corr_matrix = random_correlation.rvs(evals)
        cov_matrix = ut.covariance_from_correlation(corr_matrix, evals)
        S = np.sqrt(np.diag(cov_matrix))
        assert np.allclose(S, evals, rtol=1e-4)
        inferred_corr_matrix = ut.correlation_from_covariance(cov_matrix)
        assert np.allclose(corr_matrix, inferred_corr_matrix, rtol=1e-4)
        _enforce_is_cov(cov_matrix)


def test_smoothly_clipped_line():
    ran_key = jran.key(0)
    xarr = np.linspace(-10, 10, 200)
    x0 = 0.0

    _V = (0, None, None, None, None, None)
    smoothly_clipped_line_grad = jjit(
        vmap(grad(ut.smoothly_clipped_line, argnums=(2, 3)), in_axes=_V)
    )

    n_tests = 200
    for __ in range(n_tests):
        test_key, ran_key = jran.split(ran_key, 2)
        m_key, y_key = jran.split(test_key, 2)
        m = 10 ** jran.uniform(m_key, minval=-3, maxval=3, shape=())
        y_lo, y0, y_hi = np.sort(
            jran.uniform(y_key, minval=-100, maxval=100, shape=(3,))
        )

        # Enforce y is properly bounded
        y = ut.smoothly_clipped_line(xarr, x0, y0, m, y_lo, y_hi)
        assert np.all(np.isfinite(y))
        assert np.all(y > y_lo)
        assert np.all(y < y_hi)

        # Enforce gradients are finite and non-zero
        dy_dy0, dy_dm = smoothly_clipped_line_grad(xarr, x0, y0, m, y_lo, y_hi)
        assert np.all(np.isfinite(dy_dy0))
        assert np.all(np.isfinite(dy_dm))
        assert np.all(np.abs(dy_dy0) > 0)
        assert np.all(np.abs(m) > 0)
