"""
"""

import numpy as np
from diffmah.defaults import DEFAULT_MAH_PARAMS
from jax import random as jran

from .. import diffstarpop_block_cov as dsp
from ..mainseq_massonly import DEFAULT_SFH_PDF_MAINSEQ_PARAMS
from ..qseq_massonly_block_cov import SFH_PDF_QUENCH_PARAMS


def test_mc_diffstar_u_params_singlegal_kernel():
    ran_key = jran.key(0)
    args = (
        DEFAULT_SFH_PDF_MAINSEQ_PARAMS,
        SFH_PDF_QUENCH_PARAMS,
        DEFAULT_MAH_PARAMS,
        ran_key,
    )
    _res = dsp.mc_diffstar_u_params_singlegal_kernel(*args)
    for _x in _res:
        assert np.all(np.isfinite(_x))
    u_params_ms, u_params_qseq, frac_quench, mc_is_quenched_sequence = _res
    assert len(u_params_ms) == 9
    assert len(u_params_qseq) == 9
    assert frac_quench.shape == ()
    assert mc_is_quenched_sequence.shape == ()


def test_diffstarpop_means_covs():
    means_covs = dsp._diffstarpop_means_covs(
        DEFAULT_SFH_PDF_MAINSEQ_PARAMS, SFH_PDF_QUENCH_PARAMS, DEFAULT_MAH_PARAMS
    )
    ms_means_covs, qseq_means_covs = means_covs
    mu_ms, cov_ms = ms_means_covs
    assert np.all(np.isfinite(mu_ms))
    assert np.all(np.isfinite(cov_ms))

    frac_quench = qseq_means_covs[0]
    assert np.all(frac_quench >= 0)
    assert np.all(frac_quench <= 1)
    mu_qseq_ms_block, cov_qseq_ms_block = qseq_means_covs[1:3]
    mu_qseq_q_block, cov_qseq_q_block = qseq_means_covs[3:]
    assert np.all(np.isfinite(mu_qseq_ms_block))
    assert np.all(np.isfinite(cov_qseq_ms_block))
    assert np.all(np.isfinite(mu_qseq_q_block))
    assert np.all(np.isfinite(cov_qseq_q_block))
