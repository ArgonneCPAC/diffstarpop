"""
"""

import numpy as np
from jax import random as jran
from scipy.stats import random_correlation

from ..utils import (
    correlation_from_covariance,
    covariance_from_correlation,
    get_t50_p50,
)


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
    t50, p50 = get_t50_p50(t_table, 10**histories, threshold, logmpeak)
    assert t50.shape == (ngals,)
    assert p50.shape == (ngals,)


def test_correlation_from_covariance():
    ntests = 100
    for __ in range(ntests):
        ndim = np.random.randint(2, 10)
        evals = np.sort(np.random.uniform(0, 100, ndim))
        evals = ndim * evals / evals.sum()
        corr_matrix = random_correlation.rvs(evals)
        cov_matrix = covariance_from_correlation(corr_matrix, evals)
        S = np.sqrt(np.diag(cov_matrix))
        assert np.allclose(S, evals, rtol=1e-4)
        inferred_corr_matrix = correlation_from_covariance(cov_matrix)
        assert np.allclose(corr_matrix, inferred_corr_matrix, rtol=1e-4)
        _enforce_is_cov(cov_matrix)
