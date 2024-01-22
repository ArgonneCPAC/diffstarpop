"""
"""
import numpy as np
from diffmah.defaults import DEFAULT_MAH_PARAMS
from jax import random as jran

from ...defaults import DEFAULT_DIFFSTARPOP_PARAMS
from ..assembias_kernels import DEFAULT_AB_MAINSEQ_PARAMS, DEFAULT_AB_QSEQ_PARAMS
from ..kernel_wrapper import (
    _diffstarpop_pdf_params,
    get_assembias_slopes_mainseq,
    get_assembias_slopes_qseq,
    main_sequence_mu_cov,
    mc_diffstar_u_params_singlegal_kernel,
    quenched_sequence_mu_cov,
)
from ..mainseq_massonly import DEFAULT_SFH_PDF_MAINSEQ_PARAMS
from ..qseq_massonly import DEFAULT_SFH_PDF_QUENCH_PARAMS
from ..satquenchpop_model import DEFAULT_SATQUENCHPOP_PARAMS


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


def test_diffstarpop_pdf_params():
    p50 = 0.7
    lgmu_infall = -1.5
    logmhost_infall = 14.0
    gyr_since_infall = 3.0
    mu_ms, cov_ms, mu_qs, cov_qs, frac_q = _diffstarpop_pdf_params(
        DEFAULT_SFH_PDF_MAINSEQ_PARAMS,
        DEFAULT_SFH_PDF_QUENCH_PARAMS,
        DEFAULT_AB_MAINSEQ_PARAMS,
        DEFAULT_AB_QSEQ_PARAMS,
        DEFAULT_SATQUENCHPOP_PARAMS,
        DEFAULT_MAH_PARAMS,
        p50,
        lgmu_infall,
        logmhost_infall,
        gyr_since_infall,
    )
    assert np.array(mu_ms).shape == (4,)
    assert np.array(mu_qs).shape == (8,)
    assert cov_ms.shape == (4, 4)
    assert cov_qs.shape == (8, 8)
    assert frac_q.shape == ()
    assert 0 <= frac_q <= 1


def test_mc_diffstar_u_params_singlegal_kernel():
    p50 = 0.7
    lgmu_infall = -1.5
    logmhost_infall = 14.0
    gyr_since_infall = 3.0
    ran_key = jran.PRNGKey(0)
    res = mc_diffstar_u_params_singlegal_kernel(
        DEFAULT_SFH_PDF_MAINSEQ_PARAMS,
        DEFAULT_SFH_PDF_QUENCH_PARAMS,
        DEFAULT_AB_MAINSEQ_PARAMS,
        DEFAULT_AB_QSEQ_PARAMS,
        DEFAULT_SATQUENCHPOP_PARAMS,
        DEFAULT_MAH_PARAMS,
        p50,
        lgmu_infall,
        logmhost_infall,
        gyr_since_infall,
        ran_key,
    )
    assert np.all(np.isfinite(res.u_ms_params))
    assert np.all(np.isfinite(res.u_q_params))
