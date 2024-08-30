"""
"""

from diffmah.diffmah_kernels import mah_halopop
from diffstar.utils import cumulative_mstar_formed
from jax import jit as jjit
from jax import numpy as jnp
from jax import vmap

from ..mc_diffstarpop_block_cov import mc_diffstar_sfh_galpop
from ..sumstats.smhm import compute_smhm

_A = (None, 0)
cumulative_mstar_formed_halopop = jjit(vmap(cumulative_mstar_formed, in_axes=_A))


@jjit
def _mse(pred, target):
    diff = pred - target
    return jnp.mean(diff**2)


@jjit
def _in_situ_smhm_loss_kern(diffstarpop_params, loss_data):
    (
        mah_params,
        t_peak,
        ran_key,
        t_table,
        lgt0,
        fb,
        logmh_bins,
        smhm0_target,
        sigma_sumstat,
    ) = loss_data
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
        t_table,
        lgt0,
        fb,
    )
    diffstar_params_ms, diffstar_params_q, sfh_ms, sfh_q, frac_q, mc_is_q = _res
    smh_ms = cumulative_mstar_formed_halopop(t_table, sfh_ms)
    smh_q = cumulative_mstar_formed_halopop(t_table, sfh_q)

    logsm_z0 = jnp.log10(frac_q * smh_q[:, -1] + (1 - frac_q) * smh_ms[:, -1])

    __, log_mah_table = mah_halopop(mah_params, t_table, t_peak, lgt0)

    n_halos = logsm_z0.shape[0]
    sigma = jnp.zeros(n_halos) + sigma_sumstat

    logmh_z0 = log_mah_table[:, -1]
    smhm0_pred = compute_smhm(logmh_z0, logsm_z0, sigma, logmh_bins)
    return _mse(smhm0_pred, smhm0_target)
