""" """

import numpy as np
from jax import random as jran

from .. import DEFAULT_DIFFSTARPOP_PARAMS, mc_diffstar_params_galpop


def test_mc_diffstar_params_galpop():
    ran_key = jran.key(0)
    n_gals = 500
    logmp0 = np.zeros(n_gals) + 12.0
    upid = np.zeros(n_gals).astype(int) - 1
    lgmu_infall = np.zeros_like(logmp0)
    logmhost_infall = np.copy(logmp0)
    gyr_since_infall = np.zeros_like(logmp0)
    _res = mc_diffstar_params_galpop(
        DEFAULT_DIFFSTARPOP_PARAMS,
        logmp0,
        upid,
        lgmu_infall,
        logmhost_infall,
        gyr_since_infall,
        ran_key,
    )
