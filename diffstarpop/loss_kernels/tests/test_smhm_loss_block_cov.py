"""
"""

import numpy as np
from diffmah.diffmah_kernels import DEFAULT_MAH_PARAMS
from jax import random as jran

from ...kernels.defaults_block_cov import DEFAULT_DIFFSTARPOP_PARAMS
from ...sumstats.smdpl_smhm_targets import umachine_smhm_z0_allhalos
from .. import smhm_loss_block_cov as smhm_loss


def test_smhm_loss_is_finite():
    n_halos = 2_000
    ZZ = np.zeros(n_halos)
    t0 = 13.8
    lgt0 = np.log10(t0)
    t_peak = np.zeros(n_halos) + t0
    ran_key = jran.key(0)
    t_table = np.linspace(0.1, t0, 100)
    fb = 0.16

    logmh_bins, smhm0_target = umachine_smhm_z0_allhalos()
    sigma_sumstat = np.mean(np.diff(logmh_bins)) / 2.0

    lgm0_key, loss_key = jran.split(ran_key, 2)
    mah_params = DEFAULT_MAH_PARAMS._make([ZZ + x for x in DEFAULT_MAH_PARAMS])
    logm0 = jran.uniform(lgm0_key, minval=11.0, maxval=15.0, shape=(n_halos,))
    mah_params = mah_params._replace(logm0=logm0)

    loss_data = (
        mah_params,
        t_peak,
        loss_key,
        t_table,
        lgt0,
        fb,
        logmh_bins,
        smhm0_target,
        sigma_sumstat,
    )
    loss = smhm_loss._in_situ_smhm_loss_kern(DEFAULT_DIFFSTARPOP_PARAMS, loss_data)
    assert loss > 0
