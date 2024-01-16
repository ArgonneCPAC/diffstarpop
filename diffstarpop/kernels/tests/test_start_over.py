"""
"""
import numpy as np
from diffmah.defaults import DEFAULT_MAH_PARAMS
from jax import random as jran

from .. import start_over


def test_mc_diffstar_u_params_singlegal_kernel_evaluates():
    ran_key = jran.PRNGKey(0)

    p50 = 0.25
    args = DEFAULT_MAH_PARAMS, p50, ran_key
    u_params = start_over.mc_diffstar_u_params_singlegal_kernel(*args)
    assert len(u_params) == 9
    assert np.all(np.isfinite(u_params))
