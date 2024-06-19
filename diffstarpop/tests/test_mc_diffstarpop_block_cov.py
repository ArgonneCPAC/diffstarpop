"""
"""

import numpy as np
from diffmah.defaults import DEFAULT_MAH_PARAMS
from jax import random as jran

from ..kernels.defaults_block_cov import DEFAULT_DIFFSTARPOP_PARAMS
from ..mc_diffstarpop_block_cov import mc_diffstar_params_singlegal


def test_mc_diffstar_params_singlegal_evaluates():
    ran_key = jran.PRNGKey(0)
    args = (DEFAULT_DIFFSTARPOP_PARAMS, DEFAULT_MAH_PARAMS, ran_key)
    _res = mc_diffstar_params_singlegal(*args)
    params_ms, params_qseq, frac_q, mc_is_q = _res
    assert np.all(frac_q >= 0)
    assert np.all(frac_q <= 1)
    assert np.all(np.isfinite(params_ms.ms_params))
    assert np.all(np.isfinite(params_ms.q_params))
    assert np.all(np.isfinite(params_qseq.ms_params))
    assert np.all(np.isfinite(params_qseq.q_params))
    assert mc_is_q in (False, True)
