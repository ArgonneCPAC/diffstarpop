"""
"""

from collections import OrderedDict, namedtuple

from diffsky.diffndhist import tw_ndhist_weighted
from diffstar.utils import cumulative_mstar_formed
from jax import jit as jjit
from jax import numpy as jnp
from jax import value_and_grad, vmap

from ..kernels.defaults_tpeak import (
    DEFAULT_DIFFSTARPOP_U_PARAMS,
    get_bounded_diffstarpop_params,
)
from ..mc_diffstarpop_tpeak import mc_diffstar_sfh_galpop
from .namedtuple_utils_tpeak import (
    array_to_tuple_new_diffstarpop_tpeak,
    tuple_to_jax_array,
)

N_TIMES = 20

_A = (None, 0)
cumulative_mstar_formed_halopop = jjit(vmap(cumulative_mstar_formed, in_axes=_A))

unbound_params_dict = OrderedDict(diffstarpop_u_params=DEFAULT_DIFFSTARPOP_U_PARAMS)
UnboundParams = namedtuple("UnboundParams", list(unbound_params_dict.keys()))


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


calculate_obs_data = jjit(vmap(_calculate_obs_data_kern, in_axes=(0, 0, 0)))


def _mc_diffstar_sfh_galpop_vmap_kern(
    diffstarpop_params,
    mah_params,
    logmp0,
    lgmu_infall,
    logmhost_infall,
    gyr_since_infall,
    ran_key,
    tobs_target,
):
    tarr = jnp.logspace(-1, jnp.log10(tobs_target), N_TIMES)
    res = mc_diffstar_sfh_galpop(
        diffstarpop_params,
        mah_params,
        logmp0,
        lgmu_infall,
        logmhost_infall,
        gyr_since_infall,
        ran_key,
        tarr,
    )
    return res


_U = (None, *[0] * 7)
mc_diffstar_sfh_galpop_vmap = jjit(vmap(_mc_diffstar_sfh_galpop_vmap_kern, in_axes=_U))


# =====================================
# Functions for P(Mstar | Mobs, zobs)
# =====================================


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

    ndbins_lo = logmstar_bins[:-1].reshape((-1, 1))
    ndbins_hi = logmstar_bins[1:].reshape((-1, 1))

    wcounts = tw_ndhist_weighted(nddata, ndsig, weight, ndbins_lo, ndbins_hi)

    wcounts = wcounts / jnp.sum(wcounts)

    return wcounts


_A = (None, 0, 0)
compute_diff_histograms_mstar_atmobs_z_vmap = jjit(
    vmap(compute_diff_histograms_mstar_atmobs_z, in_axes=_A)
)


@jjit
def mstar_kern_tobs(u_params, loss_data):
    (
        mah_params,
        logmp0,
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
        logmp0,
        lgmu_infall,
        logmhost_infall,
        gyr_since_infall,
        ran_key,
        tobs_target,
    )
    diffstar_params_ms, diffstar_params_q, sfh_ms, sfh_q, frac_q, mc_is_q = _res

    smh_ms_tobs, smh_q_tobs = calculate_obs_data(tobs_target, sfh_ms, sfh_q)
    sfh_ms_tobs, sfh_q_tobs = sfh_ms[:, :, -1], sfh_q[:, :, -1]

    log_sfh_ms = jnp.log10(sfh_ms_tobs)
    log_sfh_q = jnp.log10(sfh_q_tobs)
    log_smh_ms = jnp.log10(smh_ms_tobs)
    log_smh_q = jnp.log10(smh_q_tobs)

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
    pred_mstar_pdf = mstar_kern_tobs(u_params, loss_data)

    return _mse(pred_mstar_pdf, target_mstar_pdf) * 1000


loss_mstar_kern_tobs_grad_kern = jjit(
    value_and_grad(loss_mstar_kern_tobs, argnums=(0,))
)


def loss_mstar_kern_tobs_grad_wrapper(flat_uparams, loss_data):

    namedtuple_uparams = array_to_tuple_new_diffstarpop_tpeak(
        flat_uparams, UnboundParams
    )
    loss, grads = loss_mstar_kern_tobs_grad_kern(namedtuple_uparams, loss_data)
    grads = tuple_to_jax_array(grads)

    return loss, grads


def get_pred_mstar_data_wrapper(flat_uparams, loss_data):

    namedtuple_uparams = array_to_tuple_new_diffstarpop_tpeak(
        flat_uparams, UnboundParams
    )
    pred_mstar_pdf = mstar_kern_tobs(namedtuple_uparams, loss_data)

    return pred_mstar_pdf


# =================================================
# Functions for P(sSFR | Mstar, zobs) for centrals
# =================================================


def compute_diff_histograms_mstar_ssfr_atz(
    log_smh_table,
    log_ssfr_table,
    weight,
    ndbins_lo,
    ndbins_hi,
    logmstar_bins,
    logssfr_bins,
):
    n_halos = log_smh_table.shape[0]

    sigma_mstar = jnp.mean(jnp.diff(logmstar_bins)) + jnp.zeros(n_halos)
    sigma_ssfr = jnp.mean(jnp.diff(logssfr_bins)) + jnp.zeros(n_halos)

    ndsig = jnp.array([sigma_mstar, sigma_ssfr]).T
    nddata = jnp.array([log_smh_table, log_ssfr_table]).T

    wcounts = tw_ndhist_weighted(nddata, ndsig, weight, ndbins_lo, ndbins_hi)

    wcounts = wcounts / jnp.sum(wcounts)

    return wcounts


_A = (0, 0, 0, None, None, None, None)
compute_diff_histograms_mstar_ssfr_atmobs_z_vmap = jjit(
    vmap(compute_diff_histograms_mstar_ssfr_atz, in_axes=_A)
)


@jjit
def mstar_ssfr_kern_tobs(u_params, loss_data):
    (
        mah_params,
        logmp0,
        lgmu_infall,
        logmhost_infall,
        gyr_since_infall,
        ran_key,
        tobs_target,
        ndbins_lo,
        ndbins_hi,
        logmstar_bins,
        logssfr_bins,
        nmhalo_pdf,
        target_mstar_ids,
        target_data,
    ) = loss_data

    diffstarpop_params = get_bounded_diffstarpop_params(u_params.diffstarpop_u_params)

    _res = mc_diffstar_sfh_galpop_vmap(
        diffstarpop_params,
        mah_params,
        logmp0,
        lgmu_infall,
        logmhost_infall,
        gyr_since_infall,
        ran_key,
        tobs_target,
    )
    diffstar_params_ms, diffstar_params_q, sfh_ms, sfh_q, frac_q, mc_is_q = _res

    smh_ms_tobs, smh_q_tobs = calculate_obs_data(tobs_target, sfh_ms, sfh_q)
    sfh_ms_tobs, sfh_q_tobs = sfh_ms[:, :, -1], sfh_q[:, :, -1]

    log_sfh_ms = jnp.log10(sfh_ms_tobs)
    log_sfh_q = jnp.log10(sfh_q_tobs)
    log_smh_ms = jnp.log10(smh_ms_tobs)
    log_smh_q = jnp.log10(smh_q_tobs)

    log_ssfrh_ms = log_sfh_ms - log_smh_ms
    log_ssfrh_q = log_sfh_q - log_smh_q

    log_ssfrh_ms = jnp.clip(log_ssfrh_ms, -12.0, None)
    log_ssfrh_q = jnp.clip(log_ssfrh_q, -12.0, None)

    weight_q = jnp.ones_like(log_sfh_q) * frac_q
    weight_ms = jnp.ones_like(log_sfh_ms) * (1 - frac_q)

    log_smh = jnp.concatenate((log_smh_ms, log_smh_q), axis=1)
    log_ssfrh = jnp.concatenate((log_ssfrh_ms, log_ssfrh_q), axis=1)
    weight = jnp.concatenate((weight_ms, weight_q), axis=1)

    pred_mstar_ssfr_pdf = compute_diff_histograms_mstar_ssfr_atmobs_z_vmap(
        log_smh,
        log_ssfrh,
        weight,
        ndbins_lo,
        ndbins_hi,
        logmstar_bins,
        logssfr_bins,
    )

    nms, nsf = len(logmstar_bins) - 1, len(logssfr_bins) - 1

    pdfs_z0 = pred_mstar_ssfr_pdf[0:11].reshape((11, nms, nsf))
    pdfs_z1 = pred_mstar_ssfr_pdf[11:22].reshape((11, nms, nsf))
    pdfs_z2 = pred_mstar_ssfr_pdf[22:33].reshape((11, nms, nsf))
    pdfs_z3 = pred_mstar_ssfr_pdf[33:43].reshape((10, nms, nsf))
    pdfs_z4 = pred_mstar_ssfr_pdf[43:52].reshape((9, nms, nsf))
    """
    pdfs_z0 = pred_mstar_ssfr_pdf[0:8].reshape((8, nms, nsf))
    pdfs_z1 = pred_mstar_ssfr_pdf[8:16].reshape((8, nms, nsf))
    pdfs_z2 = pred_mstar_ssfr_pdf[16:24].reshape((8, nms, nsf))
    pdfs_z3 = pred_mstar_ssfr_pdf[24:31].reshape((7, nms, nsf))
    pdfs_z4 = pred_mstar_ssfr_pdf[31:38].reshape((7, nms, nsf))
    """
    pdfs_z0 = jnp.einsum("mab,m->ab", pdfs_z0, nmhalo_pdf[0])
    pdfs_z1 = jnp.einsum("mab,m->ab", pdfs_z1, nmhalo_pdf[1])
    pdfs_z2 = jnp.einsum("mab,m->ab", pdfs_z2, nmhalo_pdf[2])
    pdfs_z3 = jnp.einsum("mab,m->ab", pdfs_z3, nmhalo_pdf[3, :-1])
    pdfs_z4 = jnp.einsum("mab,m->ab", pdfs_z4, nmhalo_pdf[4, :-2])

    pdfs_z0 = pdfs_z0[target_mstar_ids]
    pdfs_z1 = pdfs_z1[target_mstar_ids]
    pdfs_z2 = pdfs_z2[target_mstar_ids]
    pdfs_z3 = pdfs_z3[target_mstar_ids]
    pdfs_z4 = pdfs_z4[target_mstar_ids]

    pdfs_z0 = pdfs_z0 / jnp.sum(pdfs_z0, axis=1)[:, None]
    pdfs_z1 = pdfs_z1 / jnp.sum(pdfs_z1, axis=1)[:, None]
    pdfs_z2 = pdfs_z2 / jnp.sum(pdfs_z2, axis=1)[:, None]
    pdfs_z3 = pdfs_z3 / jnp.sum(pdfs_z3, axis=1)[:, None]
    pdfs_z4 = pdfs_z4 / jnp.sum(pdfs_z4, axis=1)[:, None]

    pred_data = jnp.array(
        [
            pdfs_z0,
            pdfs_z1,
            pdfs_z2,
            pdfs_z3,
            pdfs_z4,
        ]
    )

    return pred_data


@jjit
def loss_mstar_ssfr_kern_tobs(u_params, loss_data):
    target_data = loss_data[-1]

    pred_data = mstar_ssfr_kern_tobs(u_params, loss_data)

    return _mse(pred_data, target_data) * 1000


loss_mstar_ssfr_kern_tobs_grad_kern = jjit(
    value_and_grad(loss_mstar_ssfr_kern_tobs, argnums=(0,))
)


def loss_mstar_ssfr_kern_tobs_grad_wrapper(flat_uparams, loss_data):

    namedtuple_uparams = array_to_tuple_new_diffstarpop_tpeak(
        flat_uparams, UnboundParams
    )
    loss, grads = loss_mstar_ssfr_kern_tobs_grad_kern(namedtuple_uparams, loss_data)
    grads = tuple_to_jax_array(grads)

    return loss, grads


def get_pred_mstar_ssfr_data_wrapper(flat_uparams, loss_data):

    namedtuple_uparams = array_to_tuple_new_diffstarpop_tpeak(
        flat_uparams, UnboundParams
    )
    pred_mstar_pdf = mstar_ssfr_kern_tobs(namedtuple_uparams, loss_data)

    return pred_mstar_pdf


# =================================================
# Functions for P(sSFR | Mstar, zobs) for satellites
# =================================================


@jjit
def mstar_ssfr_sat_kern_tobs(u_params, loss_data):
    (
        mah_params,
        logmp0,
        lgmu_infall,
        logmhost_infall,
        gyr_since_infall,
        ran_key,
        tobs_target,
        ndbins_lo,
        ndbins_hi,
        logmstar_bins,
        logssfr_bins,
        nmhalo_pdf,
        target_mstar_ids,
        target_data,
    ) = loss_data

    diffstarpop_params = get_bounded_diffstarpop_params(u_params.diffstarpop_u_params)

    _res = mc_diffstar_sfh_galpop_vmap(
        diffstarpop_params,
        mah_params,
        logmp0,
        lgmu_infall,
        logmhost_infall,
        gyr_since_infall,
        ran_key,
        tobs_target,
    )
    diffstar_params_ms, diffstar_params_q, sfh_ms, sfh_q, frac_q, mc_is_q = _res

    smh_ms_tobs, smh_q_tobs = calculate_obs_data(tobs_target, sfh_ms, sfh_q)
    sfh_ms_tobs, sfh_q_tobs = sfh_ms[:, :, -1], sfh_q[:, :, -1]

    log_sfh_ms = jnp.log10(sfh_ms_tobs)
    log_sfh_q = jnp.log10(sfh_q_tobs)
    log_smh_ms = jnp.log10(smh_ms_tobs)
    log_smh_q = jnp.log10(smh_q_tobs)

    log_ssfrh_ms = log_sfh_ms - log_smh_ms
    log_ssfrh_q = log_sfh_q - log_smh_q

    log_ssfrh_ms = jnp.clip(log_ssfrh_ms, -12.0, None)
    log_ssfrh_q = jnp.clip(log_ssfrh_q, -12.0, None)

    weight_q = jnp.ones_like(log_sfh_q) * frac_q
    weight_ms = jnp.ones_like(log_sfh_ms) * (1 - frac_q)

    log_smh = jnp.concatenate((log_smh_ms, log_smh_q), axis=1)
    log_ssfrh = jnp.concatenate((log_ssfrh_ms, log_ssfrh_q), axis=1)
    weight = jnp.concatenate((weight_ms, weight_q), axis=1)

    pred_mstar_ssfr_pdf = compute_diff_histograms_mstar_ssfr_atmobs_z_vmap(
        log_smh,
        log_ssfrh,
        weight,
        ndbins_lo,
        ndbins_hi,
        logmstar_bins,
        logssfr_bins,
    )

    nms, nsf = len(logmstar_bins) - 1, len(logssfr_bins) - 1

    pdfs_z0 = pred_mstar_ssfr_pdf[0:11].reshape((11, nms, nsf))
    pdfs_z1 = pred_mstar_ssfr_pdf[11:21].reshape((10, nms, nsf))
    pdfs_z2 = pred_mstar_ssfr_pdf[21:31].reshape((10, nms, nsf))
    pdfs_z3 = pred_mstar_ssfr_pdf[31:40].reshape((9, nms, nsf))
    pdfs_z4 = pred_mstar_ssfr_pdf[40:48].reshape((8, nms, nsf))
    """
    pdfs_z0 = pred_mstar_ssfr_pdf[0:8].reshape((8, nms, nsf))
    pdfs_z1 = pred_mstar_ssfr_pdf[8:16].reshape((8, nms, nsf))
    pdfs_z2 = pred_mstar_ssfr_pdf[16:24].reshape((8, nms, nsf))
    pdfs_z3 = pred_mstar_ssfr_pdf[24:31].reshape((7, nms, nsf))
    pdfs_z4 = pred_mstar_ssfr_pdf[31:38].reshape((7, nms, nsf))
    """
    pdfs_z0 = jnp.einsum("mab,m->ab", pdfs_z0, nmhalo_pdf[0])
    pdfs_z1 = jnp.einsum("mab,m->ab", pdfs_z1, nmhalo_pdf[1, :-1])
    pdfs_z2 = jnp.einsum("mab,m->ab", pdfs_z2, nmhalo_pdf[2, :-1])
    pdfs_z3 = jnp.einsum("mab,m->ab", pdfs_z3, nmhalo_pdf[3, :-2])
    pdfs_z4 = jnp.einsum("mab,m->ab", pdfs_z4, nmhalo_pdf[4, :-3])

    pdfs_z0 = pdfs_z0[target_mstar_ids]
    pdfs_z1 = pdfs_z1[target_mstar_ids]
    pdfs_z2 = pdfs_z2[target_mstar_ids]
    pdfs_z3 = pdfs_z3[target_mstar_ids]
    pdfs_z4 = pdfs_z4[target_mstar_ids]

    pdfs_z0 = pdfs_z0 / jnp.sum(pdfs_z0, axis=1)[:, None]
    pdfs_z1 = pdfs_z1 / jnp.sum(pdfs_z1, axis=1)[:, None]
    pdfs_z2 = pdfs_z2 / jnp.sum(pdfs_z2, axis=1)[:, None]
    pdfs_z3 = pdfs_z3 / jnp.sum(pdfs_z3, axis=1)[:, None]
    pdfs_z4 = pdfs_z4 / jnp.sum(pdfs_z4, axis=1)[:, None]

    pred_data = jnp.array(
        [
            pdfs_z0,
            pdfs_z1,
            pdfs_z2,
            pdfs_z3,
            pdfs_z4,
        ]
    )

    return pred_data


@jjit
def loss_mstar_ssfr_sat_kern_tobs(u_params, loss_data):
    target_data = loss_data[-1]

    pred_data = mstar_ssfr_sat_kern_tobs(u_params, loss_data)

    return _mse(pred_data, target_data) * 1000


loss_mstar_ssfr_sat_kern_tobs_grad_kern = jjit(
    value_and_grad(loss_mstar_ssfr_sat_kern_tobs, argnums=(0,))
)


def loss_mstar_ssfr_sat_kern_tobs_grad_wrapper(flat_uparams, loss_data):

    namedtuple_uparams = array_to_tuple_new_diffstarpop_tpeak(
        flat_uparams, UnboundParams
    )
    loss, grads = loss_mstar_ssfr_sat_kern_tobs_grad_kern(namedtuple_uparams, loss_data)
    grads = tuple_to_jax_array(grads)

    return loss, grads


def get_pred_mstar_ssfr_sat_data_wrapper(flat_uparams, loss_data):

    namedtuple_uparams = array_to_tuple_new_diffstarpop_tpeak(
        flat_uparams, UnboundParams
    )
    pred_mstar_pdf = mstar_ssfr_sat_kern_tobs(namedtuple_uparams, loss_data)

    return pred_mstar_pdf


# =================================================
# Loss functions that combine multiple PDFs
# =================================================


@jjit
def loss_combined_kern(u_params, loss_data_mstar, loss_data_ssfr_cen):
    loss_mstar_ssfr_val_cen = loss_mstar_ssfr_kern_tobs(u_params, loss_data_ssfr_cen)
    loss_mstar_val = loss_mstar_kern_tobs(u_params, loss_data_mstar)

    return loss_mstar_ssfr_val_cen + loss_mstar_val


loss_combined_grad_kern = jjit(value_and_grad(loss_combined_kern, argnums=(0,)))


def loss_combined_wrapper(flat_uparams, loss_data_mstar, loss_data_ssfr_cen):

    namedtuple_uparams = array_to_tuple_new_diffstarpop_tpeak(
        flat_uparams, UnboundParams
    )
    loss, grads = loss_combined_grad_kern(
        namedtuple_uparams, loss_data_mstar, loss_data_ssfr_cen
    )
    grads = tuple_to_jax_array(grads)

    return loss, grads


@jjit
def loss_combined_3loss_kern(
    u_params, loss_data_mstar, loss_data_ssfr_cen, loss_data_ssfr_sat
):
    loss_mstar_ssfr_val_cen = loss_mstar_ssfr_kern_tobs(u_params, loss_data_ssfr_cen)
    loss_mstar_ssfr_val_sat = loss_mstar_ssfr_sat_kern_tobs(
        u_params, loss_data_ssfr_sat
    )
    loss_mstar_val = loss_mstar_kern_tobs(u_params, loss_data_mstar)

    return loss_mstar_ssfr_val_cen + loss_mstar_val + loss_mstar_ssfr_val_sat


loss_combined_3loss_grad_kern = jjit(
    value_and_grad(loss_combined_3loss_kern, argnums=(0,))
)


def loss_combined_3loss_wrapper(
    flat_uparams, loss_data_mstar, loss_data_ssfr_cen, loss_data_ssfr_sat
):

    namedtuple_uparams = array_to_tuple_new_diffstarpop_tpeak(
        flat_uparams, UnboundParams
    )
    loss, grads = loss_combined_3loss_grad_kern(
        namedtuple_uparams, loss_data_mstar, loss_data_ssfr_cen, loss_data_ssfr_sat
    )
    grads = tuple_to_jax_array(grads)

    return loss, grads
