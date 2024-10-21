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
from diffsky.diffndhist import tw_ndhist_weighted


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

def _calculate_obs_data_kern(
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
    
calculate_obs_data = jjit(vmap(_calculate_obs_data_kern, in_axes=(0,0,0)))

def _mc_diffstar_sfh_galpop_vmap_kern(
    diffstarpop_params,
    mah_params,
    logm0,
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
        logm0,
        t_peak,
        lgmu_infall,
        logmhost_infall,
        gyr_since_infall,
        ran_key,
        tarr
    )
    return res


_U = (None, *[0]*8)
mc_diffstar_sfh_galpop_vmap = jjit(vmap(_mc_diffstar_sfh_galpop_vmap_kern, in_axes=_U))

@jjit
def compute_diff_histograms_mstar_atmobs_z(
    logmstar_bins,
    log_smh_table,
    weight,
):

    n_halos = log_smh_table.shape[0]

    nddata = log_smh_table.reshape((-1, 1))

    sigma = jnp.mean(jnp.diff(logmstar_bins)) + jnp.zeros(n_halos)
    ndsig = sigma.reshape((-1, 1))

    ydata = log_smh_table.reshape((-1, 1))

    ndbins_lo = logmstar_bins[:-1].reshape((-1, 1))
    ndbins_hi = logmstar_bins[1:].reshape((-1, 1))

    wcounts = tw_ndhist_weighted(nddata, ndsig, weight, ndbins_lo, ndbins_hi)

    wcounts = wcounts / jnp.sum(wcounts)

    return wcounts

_A = (None, 0, 0)
compute_diff_histograms_mstar_atmobs_z_vmap = jjit(vmap(compute_diff_histograms_mstar_atmobs_z, in_axes=_A))

@jjit
def mstar_kern_tobs(u_params, loss_data):
    (
        mah_params,
        logm0,
        t_peak,
        lgmu_infall,
        logmhost_infall,
        gyr_since_infall,
        ran_key,
        tobs_target,
        logmstar_bins,
        target_mstar_pdf,
    ) = loss_data

    diffstarpop_params = get_bounded_diffstarpop_params(u_params.diffstarpop_u_params)
    
    _res = mc_diffstar_sfh_galpop_vmap(
        diffstarpop_params,
        mah_params,
        logm0,
        t_peak,
        lgmu_infall,
        logmhost_infall,
        gyr_since_infall,
        ran_key,
        tobs_target,
    )
    diffstar_params_ms, diffstar_params_q, sfh_ms, sfh_q, frac_q, mc_is_q = _res

    smh_ms_tobs, smh_q_tobs = calculate_obs_data(tobs_target, sfh_ms, sfh_q)
    sfh_ms_tobs, sfh_q_tobs = sfh_ms[:,:,-1], sfh_q[:,:,-1]

    log_sfh_ms = jnp.log10(sfh_ms_tobs)
    log_sfh_q = jnp.log10(sfh_q_tobs)
    log_smh_ms = jnp.log10(smh_ms_tobs)
    log_smh_q = jnp.log10(smh_q_tobs)

    log_ssfrh_ms = log_sfh_ms - log_smh_ms
    log_ssfrh_q = log_sfh_q - log_smh_q

    weight_q = jnp.ones_like(log_sfh_q) * frac_q
    weight_ms = jnp.ones_like(log_sfh_ms) * (1 - frac_q)

    log_smh = jnp.concatenate((log_smh_ms, log_smh_q), axis=1)
    weight_smh = jnp.concatenate((weight_ms, weight_q), axis=1)

    pred_mstar_pdf = compute_diff_histograms_mstar_atmobs_z_vmap(
        logmstar_bins,
        log_smh,
        weight_smh,
    )

    return pred_mstar_pdf 

@jjit
def loss_mstar_kern_tobs(u_params, loss_data):
    target_mstar_pdf = loss_data[-1]
    pred_mstar_pdf  = mstar_kern_tobs(u_params, loss_data)

    return _mse(pred_mstar_pdf, target_mstar_pdf) * 1000

loss_mstar_kern_tobs_grad_kern = jjit(value_and_grad(loss_mstar_kern_tobs, argnums=(0,)))


def loss_mstar_kern_tobs_grad_wrapper(flat_uparams, loss_data):
        
    namedtuple_uparams = array_to_tuple_new_diffstarpop(flat_uparams, UnboundParams)
    loss, grads = loss_mstar_kern_tobs_grad_kern(namedtuple_uparams, loss_data)
    grads = tuple_to_jax_array(grads)

    return loss, grads

def get_pred_mstar_data_wrapper(flat_uparams, loss_data):
        
    namedtuple_uparams = array_to_tuple_new_diffstarpop(flat_uparams, UnboundParams)
    pred_mstar_pdf  = mstar_kern_tobs(namedtuple_uparams, loss_data)

    return pred_mstar_pdf