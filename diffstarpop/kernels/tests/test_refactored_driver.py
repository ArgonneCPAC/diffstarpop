"""
"""
import numpy as np
from diffmah.defaults import DEFAULT_MAH_PARAMS

from ...defaults import DEFAULT_DIFFSTARPOP_PARAMS
from ..refactored_driver import (
    get_assembias_slopes_mainseq,
    get_assembias_slopes_qseq,
    main_sequence_mu_cov,
    quenched_sequence_mu_cov,
)


def test_quenched_sequence_mu_cov():
    mu, cov = quenched_sequence_mu_cov(
        DEFAULT_DIFFSTARPOP_PARAMS.sfh_pdf_quench_params, DEFAULT_MAH_PARAMS
    )


def test_main_sequence_mu_cov():
    mu, cov = main_sequence_mu_cov(
        DEFAULT_DIFFSTARPOP_PARAMS.sfh_pdf_mainseq_params, DEFAULT_MAH_PARAMS
    )


def test_get_assembias_slopes_mainseq():
    ab_slopes_ms = get_assembias_slopes_mainseq(
        DEFAULT_DIFFSTARPOP_PARAMS.assembias_mainseq_params, DEFAULT_MAH_PARAMS
    )
    assert np.all(np.isfinite(ab_slopes_ms))


def test_get_assembias_slopes_qseq():
    ab_slopes_q = get_assembias_slopes_qseq(
        DEFAULT_DIFFSTARPOP_PARAMS.assembias_quench_params, DEFAULT_MAH_PARAMS
    )
    assert np.all(np.isfinite(ab_slopes_q))
