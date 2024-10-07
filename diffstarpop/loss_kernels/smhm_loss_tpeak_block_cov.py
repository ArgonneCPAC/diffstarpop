"""
"""

from collections import OrderedDict, namedtuple
from diffmah.diffmah_kernels import mah_halopop
from diffstar.utils import cumulative_mstar_formed
import jax
from jax import jit as jjit
from jax import numpy as jnp
from jax import vmap
from jax import value_and_grad

from ..mc_diffstarpop_tpeak_block_cov import mc_diffstar_sfh_galpop
from ..sumstats.smhm import compute_smhm
from ..kernels.defaults_tpeak_block_cov import (
    get_bounded_diffstarpop_params,
    DEFAULT_DIFFSTARPOP_U_PARAMS
)
from .namedtuple_utils import (
    tuple_to_jax_array,
    array_to_tuple_new_diffstarpop,
)
from diffmah import DiffmahParams


N_TIMES = 20

_A = (None, 0)
cumulative_mstar_formed_halopop = jjit(vmap(cumulative_mstar_formed, in_axes=_A))

unbound_params_dict = OrderedDict(
    diffstarpop_u_params=DEFAULT_DIFFSTARPOP_U_PARAMS
)
UnboundParams = namedtuple(
    "UnboundParams", list(unbound_params_dict.keys()))

@jjit
def _mse(pred, target):
    diff = pred - target
    return jnp.mean(diff**2)


def _calculate_obs_smh_kern(
    tobs_target,
    sfh_ms,
    sfh_q,
):
    tarr = jnp.logspace(-1, jnp.log10(tobs_target), N_TIMES)
    smh_ms = cumulative_mstar_formed_halopop(tarr, sfh_ms)
    smh_q = cumulative_mstar_formed_halopop(tarr, sfh_q)
    smh_ms_tobs = smh_ms[:, -1]
    smh_q_tobs = smh_q[:, -1]
    return smh_ms_tobs, smh_q_tobs
    
calculate_obs_smh = jjit(vmap(_calculate_obs_smh_kern, in_axes=(0,0,0)))

@jjit
def _mse(pred, target):
    diff = pred - target
    return jnp.mean(diff**2)


@jjit
def mean_smhm_loss_kern(diffstarpop_params, loss_data):
    (
        mah_params,
        t_peak,
        lgmu_infall,
        logmhost_infall,
        gyr_since_infall,
        ran_key,
        t_table,
        mean_logsm_target
    ) = loss_data
    
    _res = mc_diffstar_sfh_galpop(
        diffstarpop_params,
        mah_params,
        t_peak,
        lgmu_infall,
        logmhost_infall,
        gyr_since_infall,
        ran_key,
        t_table,
    )
    diffstar_params_ms, diffstar_params_q, sfh_ms, sfh_q, frac_q, mc_is_q = _res
    smh_ms = cumulative_mstar_formed_halopop(t_table, sfh_ms)
    smh_q = cumulative_mstar_formed_halopop(t_table, sfh_q)

    # logsm = jnp.log10(frac_q * smh_q[:, -1] + (1 - frac_q) * smh_ms[:, -1])
    logsm = frac_q * jnp.log10(smh_q[:, -1]) + (1 - frac_q) * jnp.log10(smh_ms[:, -1])

    mean_logsm_pred = jnp.mean(logsm)

    return _mse(mean_logsm_pred, mean_logsm_target)


def _mc_diffstar_sfh_galpop_vmap_kern(
    diffstarpop_params,
    mah_params,
    t_peak,
    lgmu_infall,
    logmhost_infall,
    gyr_since_infall,
    ran_key,
    tobs_target,
):
    mah_params = DiffmahParams(*mah_params)
    tarr = jnp.logspace(-1, jnp.log10(tobs_target), N_TIMES)
    res = mc_diffstar_sfh_galpop(
        diffstarpop_params,
        mah_params,
        t_peak,
        lgmu_infall,
        logmhost_infall,
        gyr_since_infall,
        ran_key,
        tarr
    )
    return res


_U = (None, *[0]*7)
mc_diffstar_sfh_galpop_vmap = jjit(vmap(_mc_diffstar_sfh_galpop_vmap_kern, in_axes=_U))

@jjit
def mean_smhm_loss_kern_tobs(u_params, loss_data):
    (
        mah_params,
        t_peak,
        lgmu_infall,
        logmhost_infall,
        gyr_since_infall,
        ran_key,
        tobs_target,
        mean_logsm_target
    ) = loss_data

    diffstarpop_params = get_bounded_diffstarpop_params(u_params.diffstarpop_u_params)
    
    _res = mc_diffstar_sfh_galpop_vmap(
        diffstarpop_params,
        mah_params,
        t_peak,
        lgmu_infall,
        logmhost_infall,
        gyr_since_infall,
        ran_key,
        tobs_target,
    )
    diffstar_params_ms, diffstar_params_q, sfh_ms, sfh_q, frac_q, mc_is_q = _res

    smh_ms_tobs, smh_q_tobs = calculate_obs_smh(tobs_target, sfh_ms, sfh_q)

    # logsm = jnp.log10(frac_q * smh_q_tobs + (1 - frac_q) * smh_ms_tobs)
    logsm = frac_q * jnp.log10(smh_q_tobs) + (1 - frac_q) * jnp.log10(smh_ms_tobs)

    mean_logsm_pred = jnp.mean(logsm, axis=1)

    return _mse(mean_logsm_pred, mean_logsm_target)

mean_smhm_loss_kern_tobs_grad_kern = jjit(value_and_grad(mean_smhm_loss_kern_tobs, argnums=(0,)))


def mean_smhm_loss_kern_tobs_wrapper(flat_uparams, loss_data):
        
    namedtuple_uparams = array_to_tuple_new_diffstarpop(flat_uparams, UnboundParams)

    loss, grads = mean_smhm_loss_kern_tobs_grad_kern(namedtuple_uparams, loss_data)
    grads = tuple_to_jax_array(grads)

    return loss, grads

