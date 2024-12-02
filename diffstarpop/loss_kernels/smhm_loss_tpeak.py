"""
"""

from collections import OrderedDict, namedtuple

import h5py
import numpy as np
from diffmah.diffmah_kernels import DiffmahParams, mah_halopop
from diffstar.defaults import LGT0
from diffstar.utils import cumulative_mstar_formed
from jax import jit as jjit
from jax import numpy as jnp
from jax import random as jran
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


calculate_obs_smh = jjit(vmap(_calculate_obs_smh_kern, in_axes=(0, 0, 0)))


@jjit
def _mse(pred, target):
    diff = pred - target
    return jnp.mean(diff**2)


@jjit
def mean_smhm_loss_kern(diffstarpop_params, loss_data):
    (
        mah_params,
        logm0,
        lgmu_infall,
        logmhost_infall,
        gyr_since_infall,
        ran_key,
        t_table,
        mean_logsm_target,
    ) = loss_data

    _res = mc_diffstar_sfh_galpop(
        diffstarpop_params,
        mah_params,
        logm0,
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
    logm0,
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
        lgmu_infall,
        logmhost_infall,
        gyr_since_infall,
        ran_key,
        tarr,
    )
    return res


_U = (None, *[0] * 7)
mc_diffstar_sfh_galpop_vmap = jjit(vmap(_mc_diffstar_sfh_galpop_vmap_kern, in_axes=_U))


@jjit
def mean_smhm_kern_tobs(u_params, loss_data):
    (
        mah_params,
        logm0,
        lgmu_infall,
        logmhost_infall,
        gyr_since_infall,
        ran_key,
        tobs_target,
        mean_logsm_target,
    ) = loss_data

    diffstarpop_params = get_bounded_diffstarpop_params(u_params.diffstarpop_u_params)

    _res = mc_diffstar_sfh_galpop_vmap(
        diffstarpop_params,
        mah_params,
        logm0,
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

    return mean_logsm_pred


@jjit
def mean_smhm_loss_kern_tobs(u_params, loss_data):
    mean_logsm_target = loss_data[-1]
    mean_logsm_pred = mean_smhm_kern_tobs(u_params, loss_data)

    return _mse(mean_logsm_pred, mean_logsm_target)


mean_smhm_loss_kern_tobs_grad_kern = jjit(
    value_and_grad(mean_smhm_loss_kern_tobs, argnums=(0,))
)


def mean_smhm_loss_kern_tobs_wrapper(flat_uparams, loss_data):

    namedtuple_uparams = array_to_tuple_new_diffstarpop_tpeak(
        flat_uparams, UnboundParams
    )

    loss, grads = mean_smhm_loss_kern_tobs_grad_kern(namedtuple_uparams, loss_data)
    grads = tuple_to_jax_array(grads)

    return loss, grads


def get_loss_data(indir, nhalos):
    # Load SMHM data ---------------------------------------------
    print("Loading SMHM data...")

    with h5py.File(indir + "smdpl_smhm.h5", "r") as hdf:
        redshift_targets = hdf["redshift_targets"][:]
        # smhm_diff = hdf["smhm_diff"][:]
        smhm = hdf["smhm"][:]
        logmh_bins = hdf["logmh_bins"][:]
        age_targets = hdf["age_targets"][:]
        """
            hdfout["counts_diff"] = wcounts
            hdfout["hist_diff"] = whist
            hdfout["counts"] = counts
            hdfout["hist"] = hist
            hdfout["smhm_diff"] = whist / wcounts
            hdfout["smhm"] = hist / counts
            hdfout["logmh_bins"] = smhm_utils.LOGMH_BINS
            hdfout["subvol_used"] = subvol_used
        """

    logmh_binsc = 0.5 * (logmh_bins[1:] + logmh_bins[:-1])

    with h5py.File(indir + "smdpl_smhm_samples_haloes.h5", "r") as hdf:
        logmh_id = hdf["logmh_id"][:]
        # logmh_val = hdf["logmh_id"][:]
        mah_params_samp = hdf["mah_params_samp"][:]
        # ms_params_samp = hdf["ms_params_samp"][:]
        # q_params_samp = hdf["q_params_samp"][:]
        t_peak_samp = hdf["t_peak_samp"][:]
        tobs_id = hdf["tobs_id"][:]
        # tobs_val = hdf["tobs_val"][:]
        # redshift_val = hdf["redshift_val"][:]

    mah_params_samp = np.concatenate((mah_params_samp, t_peak_samp[None, :]), axis=0)

    # Create loss_data ---------------------------------------------
    print("Creating loss data...")

    ran_key = jran.PRNGKey(np.random.randint(2**32))

    lgmu_infall = -1.0
    logmhost_infall = 13.0
    gyr_since_infall = 2.0

    mah_params_data = []
    lomg0_data = []
    lgmu_infall_data = []
    logmhost_infall_data = []
    gyr_since_infall_data = []
    t_obs_targets = []
    smhm_targets = []

    tarr_logm0 = np.logspace(-1, LGT0, 50)

    for i in range(len(age_targets)):
        t_target = age_targets[i]

        for j in range(len(logmh_binsc)):
            sel = (tobs_id == i) & (logmh_id == j)

            if sel.sum() < nhalos:
                continue
            arange_sel = np.arange(len(tobs_id))[sel]
            arange_sel = np.random.choice(arange_sel, nhalos, replace=False)
            mah_params_data.append(mah_params_samp[:, arange_sel])
            lgmu_infall_data.append(np.ones(len(arange_sel)) * lgmu_infall)
            logmhost_infall_data.append(np.ones(len(arange_sel)) * logmhost_infall)
            gyr_since_infall_data.append(np.ones(len(arange_sel)) * gyr_since_infall)
            t_obs_targets.append(t_target)
            smhm_targets.append(smhm[i, j])
            mah_pars_ntuple = DiffmahParams(*mah_params_samp[:, arange_sel])
            dmhdt_fit, log_mah_fit = mah_halopop(mah_pars_ntuple, tarr_logm0, LGT0)
            lomg0_data.append(log_mah_fit[:, -1])

    mah_params_data = np.array(mah_params_data)
    lomg0_data = np.array(lomg0_data)
    lgmu_infall_data = np.array(lgmu_infall_data)
    logmhost_infall_data = np.array(logmhost_infall_data)
    gyr_since_infall_data = np.array(gyr_since_infall_data)
    t_obs_targets = np.array(t_obs_targets)
    smhm_targets = np.array(smhm_targets)

    ran_key_data = jran.split(ran_key, len(smhm_targets))
    loss_data = (
        mah_params_data,
        lomg0_data,
        lgmu_infall_data,
        logmhost_infall_data,
        gyr_since_infall_data,
        ran_key_data,
        t_obs_targets,
        smhm_targets,
    )

    plot_data = (
        age_targets,
        logmh_binsc,
        tobs_id,
        logmh_id,
        tarr_logm0,
        lgmu_infall,
        logmhost_infall,
        gyr_since_infall,
        ran_key,
        redshift_targets,
        smhm,
        mah_params_samp,
    )

    return loss_data, plot_data
