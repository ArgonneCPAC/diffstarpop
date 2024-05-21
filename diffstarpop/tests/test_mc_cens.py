"""
"""

import numpy as np
from diffmah.defaults import DEFAULT_MAH_PARAMS
from diffstar import DEFAULT_DIFFSTAR_PARAMS
from diffstar.defaults import T_TABLE_MIN, TODAY
from jax import random as jran

from ..defaults import DEFAULT_DIFFSTARPOP_PARAMS
from ..mc_cens import mc_diffstar_sfh_singlecen


def test_mc_diffstar_params_galpop_evaluates():
    ran_key = jran.PRNGKey(0)

    p50 = 0.25

    tarr = np.linspace(T_TABLE_MIN, TODAY)
    args = (DEFAULT_DIFFSTARPOP_PARAMS, DEFAULT_MAH_PARAMS, p50, ran_key, tarr)
    params_q, params_ms, sfh_q, sfh_ms, frac_q = mc_diffstar_sfh_singlecen(*args)
    assert len(params_q.ms_params) == len(DEFAULT_DIFFSTAR_PARAMS.ms_params)
    assert len(params_q.q_params) == len(DEFAULT_DIFFSTAR_PARAMS.q_params)
    for p in params_q.ms_params:
        assert p.shape == ()
    for p in params_q.q_params:
        assert p.shape == ()

    assert len(params_ms.ms_params) == len(DEFAULT_DIFFSTAR_PARAMS.ms_params)
    assert len(params_ms.q_params) == len(DEFAULT_DIFFSTAR_PARAMS.q_params)
    for p in params_ms.ms_params:
        assert p.shape == ()
    for p in params_ms.q_params:
        assert p.shape == ()

    assert frac_q.shape == ()
