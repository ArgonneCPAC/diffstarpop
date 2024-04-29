"""
"""

import numpy as np
from dsps.constants import SFR_MIN
from dsps.utils import _jax_get_dt_array, cumulative_mstar_formed
from jax import jit as jjit
from jax import numpy as jnp

TODAY = 13.75


@jjit
def sm_sfr_history_diffstar_scan(tarr, lgt, mah_params, sfr_ms_params, q_params):
    sfr_params = [*sfr_ms_params[0:3], UH, sfr_ms_params[3]]
    ms_sfr = sfh_scan_tobs_kern(tarr, mah_params, sfr_params)
    qfrac = quenching_function(lgt, *q_params)
    sfr = qfrac * ms_sfr
    sfr = jnp.clip(sfr, SFR_MIN, None)
    mstar = cumulative_mstar_formed(tarr, sfr)
    return mstar, sfr


def calculate_SMDPL_sumstats(
    t_table, logm0_binmids, logm0_bin_widths, mah_params, u_fit_params, p50
):
    logmpeak = mah_params[:, 1]

    lgt = np.log10(t_table)

    fstar_tdelay = 1.0
    dt = _jax_get_dt_array(t_table)

    stats = []
    for i in range(len(logm0_binmids)):
        print(
            "Calculating m0=[%.2f, %.2f]"
            % (
                logm0_binmids[i] - logm0_bin_widths[i],
                logm0_binmids[i] + logm0_bin_widths[i],
            )
        )
        sel = (logmpeak > logm0_binmids[i] - logm0_bin_widths[i]) & (
            logmpeak < logm0_binmids[i] + logm0_bin_widths[i]
        )
        print("Nhalos:", sel.sum())
        _res = sm_sfr_history_diffstar_scan_XsfhXmah_vmap(
            t_table,
            lgt,
            dt,
            mah_params[sel][:, [1, 2, 4, 5]],
            u_fit_params[sel][:, [0, 1, 2, 4]].copy(),
            u_fit_params[sel][:, [5, 6, 7, 8]].copy(),
            index_select,
            index_high,
            fstar_tdelay,
        )

        (
            mstar_histories,
            sfr_histories,
            fstar_histories,
        ) = _res

        ssfr = sfr_histories / mstar_histories
        weights_quench_bin = jnp.where(ssfr > 1e-11, 1.0, 0.0)

        _stats = calculate_sumstats_bin(
            mstar_histories, sfr_histories, p50[sel], weights_quench_bin
        )
        stats.append(_stats)

    print("Reshaping results")

    new_stats = []
    nres = len(_stats)
    for j in range(nres):
        _new_stats = []
        for i in range(len(logm0_binmids)):
            _new_stats.append(stats[i][j])
        new_stats.append(np.array(_new_stats))

    new_stats = np.array(new_stats)
    return new_stats


def get_loss_p50_data(path="/lcrc/project/halotools/alarcon/data/"):
    # path = "/Users/alarcon/Documents/diffmah_data/SMDPL/"

    # Pre-aggregated halo and galaxy fits from all 576 SMDPL volumes.
    # Below Mh<12 only a subset of halos are included in these files.
    # The selection makes the halo mass function is constant at Mh<12 (for speedup).
    mah_params_arr = np.load(path + "mah_params_arr_576_small.npy")
    u_fit_params_arr = np.load(path + "u_fit_params_arr_576_small.npy")
    # fit_params_arr = np.load(path+"fit_params_arr_576_small.npy")
    p50_arr = np.load(path + "p50_arr_576_small.npy")
    logmpeak = mah_params_arr[:, 1]

    t_table = np.linspace(1.0, TODAY, 20)
    # Define some mass bins for predictions
    # logm0_binmids = np.linspace(11.5, 13.5, 5)
    logm0_binmids = np.linspace(11.0, 14.0, 7)

    logm0_bin_widths = np.ones_like(logm0_binmids) * 0.1

    # Calculate the Target summary statistics from all haloes, conditioned on halo mass bins.
    MC_res_target = calculate_SMDPL_sumstats(
        t_table,
        logm0_binmids,
        logm0_bin_widths,
        mah_params_arr,
        u_fit_params_arr,
        p50_arr,
    )

    MC_res_target = np.array(MC_res_target)

    # Select the subset of Nhalos per halo mass bin that will be used in the gradient descent of the loss function.
    Nhalos = 3000
    halo_data_MC = []
    p50 = []
    for i in range(len(logm0_binmids)):
        _sel = (logmpeak > logm0_binmids[i] - logm0_bin_widths[i]) & (
            logmpeak < logm0_binmids[i] + logm0_bin_widths[i]
        )
        print(_sel.sum())
        replace = True if _sel.sum() < Nhalos else False
        sel = np.random.choice(np.arange(len(p50_arr))[_sel], Nhalos, replace=replace)
        halo_data_MC.append(mah_params_arr[sel])
        p50.append(p50_arr[sel])
    halo_data_MC = np.concatenate(halo_data_MC, axis=0)[:, np.array([1, 2, 4, 5])]
    p50 = np.concatenate(p50, axis=0)

    loss_data = (
        t_table,
        logm0_binmids,
        halo_data_MC.reshape(len(logm0_binmids), Nhalos, 4),
        p50.reshape(len(logm0_binmids), Nhalos),
        MC_res_target,
    )
    return loss_data


def calculate_sumstats_bin(mstar_histories, sfr_histories, p50, weights_MS):
    weights_Q = 1.0 - weights_MS

    # Clip weights. When all weights in a time
    # step are 0, Nans will occur in gradients.
    eps = 1e-10
    weights_Q = jnp.clip(weights_Q, eps, None)
    weights_MS = jnp.clip(weights_MS, eps, None)

    weights_early = jnp.where(p50 < 0.5, 1.0, 0.0)
    weights_late = 1.0 - weights_early
    weights_early = jnp.clip(weights_early, eps, None)
    weights_late = jnp.clip(weights_late, eps, None)

    mstar_histories = jnp.where(mstar_histories > 0.0, jnp.log10(mstar_histories), 0.0)
    sfr_histories = jnp.where(sfr_histories > 0.0, jnp.log10(sfr_histories), 0.0)
    # fstar_histories = jnp.where(fstar_histories > 0.0, jnp.log10(fstar_histories), 0.0)

    mean_sm = jnp.average(mstar_histories, axis=0)
    mean_sfr_MS = jnp.average(sfr_histories, weights=weights_MS, axis=0)
    mean_sfr_Q = jnp.average(sfr_histories, weights=weights_Q, axis=0)

    mean_sm_early = jnp.average(mstar_histories, weights=weights_early, axis=0)
    mean_sm_late = jnp.average(mstar_histories, weights=weights_late, axis=0)

    variance_sm = jnp.average(
        (mstar_histories - mean_sm[None, :]) ** 2,
        axis=0,
    )

    variance_sfr_MS = jnp.average(
        (sfr_histories - mean_sfr_MS[None, :]) ** 2,
        weights=weights_MS,
        axis=0,
    )
    variance_sfr_Q = jnp.average(
        (sfr_histories - mean_sfr_Q[None, :]) ** 2,
        weights=weights_Q,
        axis=0,
    )
    variance_sm_early = jnp.average(
        (mstar_histories - mean_sm[None, :]) ** 2,
        weights=weights_early,
        axis=0,
    )
    variance_sm_late = jnp.average(
        (mstar_histories - mean_sm[None, :]) ** 2,
        weights=weights_late,
        axis=0,
    )

    NHALO_MS = jnp.sum(weights_MS, axis=0)
    NHALO_Q = jnp.sum(weights_Q, axis=0)
    quench_frac = NHALO_Q / (NHALO_Q + NHALO_MS)

    mean_sfr_Q = jnp.where(quench_frac == 0.0, 0.0, mean_sfr_Q)
    variance_sfr_Q = jnp.where(quench_frac == 0.0, 0.0, variance_sfr_Q)
    mean_sfr_MS = jnp.where(quench_frac == 1.0, 0.0, mean_sfr_MS)
    variance_sfr_MS = jnp.where(quench_frac == 1.0, 0.0, variance_sfr_MS)

    NHALO_MS_early = jnp.sum(weights_MS * weights_early[:, None], axis=0)
    NHALO_Q_early = jnp.sum(weights_Q * weights_early[:, None], axis=0)
    quench_frac_early = NHALO_Q_early / (NHALO_Q_early + NHALO_MS_early)

    NHALO_MS_late = jnp.sum(weights_MS * weights_late[:, None], axis=0)
    NHALO_Q_late = jnp.sum(weights_Q * weights_late[:, None], axis=0)
    quench_frac_late = NHALO_Q_late / (NHALO_Q_late + NHALO_MS_late)

    _out = (
        mean_sm,
        variance_sm,
        mean_sfr_MS,
        mean_sfr_Q,
        variance_sfr_MS,
        variance_sfr_Q,
        quench_frac,
        mean_sm_early,
        mean_sm_late,
        variance_sm_early,
        variance_sm_late,
        quench_frac_early,
        quench_frac_late,
    )
    return _out
