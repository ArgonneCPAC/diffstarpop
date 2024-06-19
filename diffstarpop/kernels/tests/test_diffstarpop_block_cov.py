"""
"""

import numpy as np
from diffmah.defaults import DEFAULT_MAH_PARAMS
from jax import random as jran

from .. import diffstarpop_block_cov as dsp
from ..sfh_pdf_block_cov import SFH_PDF_QUENCH_PARAMS


def test_mc_diffstar_u_params_singlegal_kernel():
    ran_key = jran.key(0)
    args = (SFH_PDF_QUENCH_PARAMS, DEFAULT_MAH_PARAMS, ran_key)
    _res = dsp.mc_diffstar_u_params_singlegal_kernel(*args)
    u_params_ms, u_params_qseq, frac_quench, mc_is_quenched_sequence = _res
    assert len(u_params_ms.u_ms_params) == 5
    assert len(u_params_ms.u_q_params) == 4
    assert len(u_params_qseq.u_ms_params) == 5
    assert len(u_params_qseq.u_q_params) == 4
    for _u_p in u_params_ms:
        assert np.all(np.isfinite(_u_p))
    for _u_p in u_params_qseq:
        assert np.all(np.isfinite(_u_p))
    assert frac_quench.shape == ()
    assert mc_is_quenched_sequence.shape == ()


def test_diffstarpop_means_covs():
    means_covs = dsp._diffstarpop_means_covs(SFH_PDF_QUENCH_PARAMS, DEFAULT_MAH_PARAMS)
    qseq_means_covs = means_covs

    frac_quench = qseq_means_covs[0]
    assert np.all(frac_quench >= 0)
    assert np.all(frac_quench <= 1)
    mu_qseq_ms_block, cov_qseq_ms_block = qseq_means_covs[1:3]
    mu_qseq_q_block, cov_qseq_q_block = qseq_means_covs[3:]
    assert np.all(np.isfinite(mu_qseq_ms_block))
    assert np.all(np.isfinite(cov_qseq_ms_block))
    assert np.all(np.isfinite(mu_qseq_q_block))
    assert np.all(np.isfinite(cov_qseq_q_block))
