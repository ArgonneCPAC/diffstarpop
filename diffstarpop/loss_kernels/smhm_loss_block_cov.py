"""
"""

from jax import jit as jjit
from jax import numpy as jnp

from ..mc_diffstarpop_block_cov import mc_diffstar_sfh_galpop
from ..sumstats.smhm import compute_smhm


@jjit
def _mse(pred, target):
    diff = pred - target
    return jnp.mean(diff**2)


@jjit
def _in_situ_smhm_loss_kern(diffstarpop_params, loss_data):
    mah_params, ran_key, tarr, lgt0, fb, logmh_bins, smhm_target = loss_data
    n_halos = mah_params.logm0.size
    lgmu_infall = jnp.zeros(n_halos)
    logmhost_infall = jnp.zeros(n_halos)
    gyr_since_infall = jnp.zeros(n_halos)

    _res = mc_diffstar_sfh_galpop(
        diffstarpop_params,
        mah_params,
        lgmu_infall,
        logmhost_infall,
        gyr_since_infall,
        ran_key,
        tarr,
        lgt0,
        fb,
    )
    diffstar_params_ms, diffstar_params_q, sfh_ms, sfh_q, frac_q, mc_is_q = _res

    mean_logsm0 = compute_smhm(logmh, logsm, sigma, logmh_bins)
