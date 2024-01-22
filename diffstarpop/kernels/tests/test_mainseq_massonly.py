"""
"""
import numpy as np

from ..mainseq_massonly import (
    DEFAULT_SFH_PDF_MAINSEQ_PARAMS,
    _get_cov_mainseq,
    _get_mean_u_params_mainseq,
    get_bounded_mainseq_massonly_params,
    get_unbounded_mainseq_massonly_params,
)


def test_get_mean_u_params_mainseq_agrees_with_legacy():
    lgm = 12.0
    mu_ms = _get_mean_u_params_mainseq(DEFAULT_SFH_PDF_MAINSEQ_PARAMS, lgm)
    assert np.all(np.isfinite(mu_ms))


def test_get_cov_mainseq_agrees_with_legacy():
    lgm = 12.0
    cov_ms = _get_cov_mainseq(DEFAULT_SFH_PDF_MAINSEQ_PARAMS, lgm)
    assert np.all(np.isfinite(cov_ms))


def test_params_u_params():
    mainseq_massonly_u_params = get_unbounded_mainseq_massonly_params(
        DEFAULT_SFH_PDF_MAINSEQ_PARAMS
    )
    mainseq_massonly_params = get_bounded_mainseq_massonly_params(
        mainseq_massonly_u_params
    )
    assert np.allclose(DEFAULT_SFH_PDF_MAINSEQ_PARAMS, mainseq_massonly_params)
