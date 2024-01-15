"""
"""
import numpy as np
from diffmah.defaults import DEFAULT_MAH_PARAMS
from jax import random as jran

from ..defaults import DEFAULT_DIFFSTARPOP_PARAMS
from ..kernels.start_over import mc_diffstar_u_params_singlegal_kernel
from ..mc_diffstarpop import mc_diffstar_u_params_galpop, mc_diffstar_u_params_singlegal


def test_mc_diffstar_params_singlegal_consistent_with_mc_diffstar_u_params_singlegal():
    ran_key = jran.PRNGKey(0)

    p50 = 0.25
    args = DEFAULT_MAH_PARAMS, p50, ran_key
    u_params = mc_diffstar_u_params_singlegal_kernel(*args)

    args2 = (DEFAULT_DIFFSTARPOP_PARAMS, DEFAULT_MAH_PARAMS, p50, ran_key)
    u_params2 = mc_diffstar_u_params_singlegal(*args2)

    assert np.allclose(u_params[:4], u_params2.u_ms_params)
    assert np.allclose(u_params[4:], u_params2.u_q_params)


def test_mc_diffstar_params_galpop_evaluates():
    ran_key = jran.PRNGKey(0)

    p50 = 0.25

    ngals = 15
    zz = np.zeros(ngals)
    nmah = len(DEFAULT_MAH_PARAMS)
    mah_params_galpop = np.repeat(DEFAULT_MAH_PARAMS, ngals).reshape((ngals, nmah))
    args = (DEFAULT_DIFFSTARPOP_PARAMS, mah_params_galpop, p50 + zz, ran_key)
    u_params = mc_diffstar_u_params_galpop(*args)
    assert u_params.u_ms_params.shape == (ngals, 4)
    assert u_params.u_q_params.shape == (ngals, 4)
