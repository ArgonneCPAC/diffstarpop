"""
"""

import numpy as np
from diffmah.defaults import DEFAULT_MAH_PARAMS
from jax import random as jran

from .. import diffstarpop_tpeak as dsp
from ..defaults_tpeak import DEFAULT_DIFFSTARPOP_PARAMS


def test_mc_diffstar_u_params_singlegal_kernel():
    ran_key = jran.key(0)
    t_peak = 12.0
    lgmu_infall = -1.0
    logmhost_infall = 13.0
    gyr_since_infall = 2.0
    args = (
        DEFAULT_DIFFSTARPOP_PARAMS,
        DEFAULT_MAH_PARAMS,
        t_peak,
        lgmu_infall,
        logmhost_infall,
        gyr_since_infall,
        ran_key,
    )
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
    lgmu_infall = -1.0
    logmhost_infall = 13.0
    gyr_since_infall = 2.0
    means_covs = dsp._diffstarpop_means_covs(
        DEFAULT_DIFFSTARPOP_PARAMS,
        DEFAULT_MAH_PARAMS,
        lgmu_infall,
        logmhost_infall,
        gyr_since_infall,
    )
    #
    frac_quench = means_covs[0]
    assert np.all(frac_quench >= 0)
    assert np.all(frac_quench <= 1)
    mu_mseq = means_covs[1]
    assert np.all(np.isfinite(mu_mseq))
    mu_qseq_ms_block, cov_qseq_ms_block = means_covs[2:4]
    mu_qseq_q_block, cov_qseq_q_block = means_covs[4:]
    assert np.all(np.isfinite(mu_qseq_ms_block))
    assert np.all(np.isfinite(cov_qseq_ms_block))
    assert np.all(np.isfinite(mu_qseq_q_block))
    assert np.all(np.isfinite(cov_qseq_q_block))
