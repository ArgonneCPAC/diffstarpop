"""
"""
import numpy as np

from ..assembias_kernels import (
    DEFAULT_AB_MAINSEQ_PARAMS,
    DEFAULT_AB_QSEQ_PARAMS,
    _get_slopes_mainseq,
    _get_slopes_qseq,
    get_bounded_ab_mainseq_params,
    get_bounded_ab_qseq_params,
    get_unbounded_ab_mainseq_params,
    get_unbounded_ab_qseq_params,
)


def test_get_slopes_qseq():
    lgm = 12.0
    slopes_qseq = _get_slopes_qseq(DEFAULT_AB_QSEQ_PARAMS, lgm)
    assert np.all(np.isfinite(slopes_qseq))


def test_get_slopes_mainseq():
    lgm = 12.0
    slopes_mainseq = _get_slopes_mainseq(DEFAULT_AB_MAINSEQ_PARAMS, lgm)
    assert np.all(np.isfinite(slopes_mainseq))


def test_ab_qseq_params_u_params_inverts():
    ab_qseq_u_params = get_unbounded_ab_qseq_params(DEFAULT_AB_QSEQ_PARAMS)
    ab_qseq_params = get_bounded_ab_qseq_params(ab_qseq_u_params)
    assert np.allclose(DEFAULT_AB_QSEQ_PARAMS, ab_qseq_params)


def test_ab_mainseq_params_u_params_inverts():
    ab_mainseq_u_params = get_unbounded_ab_mainseq_params(DEFAULT_AB_MAINSEQ_PARAMS)
    ab_mainseq_params = get_bounded_ab_mainseq_params(ab_mainseq_u_params)
    assert np.allclose(DEFAULT_AB_MAINSEQ_PARAMS, ab_mainseq_params)
