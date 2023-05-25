import numpy as np

from jax import numpy as jnp
from jax import jit as jjit
from jax import vmap
from jax import grad
from jax import random as jran
import time

from jax.example_libraries import optimizers as jax_opt
from diffsky.diffndhist import tw_ndhist, tw_ndhist_weighted

from diffstarpop.monte_carlo_diff_halo_population import (
    sm_sfr_history_diffstar_scan_XsfhXmah_vmap,
    sm_sfr_history_diffstar_scan_MS_XsfhXmah_vmap,
    _jax_get_dt_array,
    calc_hist_mstar_ssfr,
)

from diffstarpop.fit_pop_helpers import (
    loss_hists_vmap,
    loss_hists_vmap_deriv,
    loss_hists_vmap_deriv_np,
)

from diffstar.constants import TODAY
from diffstar.stars import fstar_tools
import astropy.units as u
from astropy.cosmology import Planck13, z_at_value
from diffstarpop.json_utils import (
    load_params,
    write_params_json,
    print_all_default_dicts,
)


t_table = np.linspace(1.0, TODAY, 20)
z_table = np.array([z_at_value(Planck13.age, x * u.Gyr, zmin=-1) for x in t_table])

t_sel_hists = np.argmin(
    np.subtract.outer(z_table, np.arange(0, 2.1, 0.5)) ** 2, axis=0
)[::-1]

print(t_table)
print(z_table)
lgt = np.log10(t_table)

# Define some mass bins for predictions
logm0_binmids = np.linspace(11.0, 14.5, 8)
logm0_bin_widths = np.ones_like(logm0_binmids) * 0.1

# Define some useful quantities and masks for later
fstar_tdelay = 1.0
index_select, index_high = fstar_tools(t_table, fstar_tdelay=fstar_tdelay)

path = "/lcrc/project/halotools/alarcon/data/"

mah_params_arr = np.load(path + "mah_params_arr_SMDPL_576_small.npy")
u_fit_params_arr = np.load(path + "u_fit_params_arr_SMDPL_576_small.npy")
fit_params_arr = np.load(path + "fit_params_arr_SMDPL_576_small.npy")
p50_arr = np.load(path + "p50_arr_SMDPL_576_small.npy")

logmpeak = mah_params_arr[:, 1]


bins_mstar = jnp.linspace(7, 12, 25)
bins_ssfr = jnp.linspace(-13, -8.5, 25)
delta_bins_mstar = float(jnp.diff(bins_mstar)[0])
delta_bins_ssfr = float(jnp.diff(bins_ssfr)[0])

bins_LO = []
bins_HI = []
for i in range(len(bins_mstar) - 1):
    for j in range(len(bins_ssfr) - 1):
        bins_LO.append([bins_mstar[i], bins_ssfr[j]])
        bins_HI.append([bins_mstar[i + 1], bins_ssfr[j + 1]])
bins_LO = np.array(bins_LO)
bins_HI = np.array(bins_HI)


def calculate_SMDPL_sumstats(
    t_table,
    logm0_binmids,
    logm0_bin_widths,
    mah_params,
    fit_params,
    p50,
    bins_LO,
    bins_HI,
):
    logmpeak = mah_params[:, 1]

    lgt = np.log10(t_table)

    fstar_tdelay = 1.0
    index_select, index_high = fstar_tools(t_table, fstar_tdelay=fstar_tdelay)
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
            fit_params[sel][:, [0, 1, 2, 4]].copy(),
            fit_params[sel][:, [5, 6, 7, 8]].copy(),
            index_select,
            index_high,
            fstar_tdelay,
        )

        (mstar_histories, sfr_histories, fstar_histories,) = _res

        ssfr = sfr_histories / mstar_histories
        weights_quench_bin = jnp.where(ssfr > 1e-11, 1.0, 0.0)

        n_histories = len(mstar_histories)
        ndsig = np.ones((n_histories, 2))
        ndsig[:, 0] *= delta_bins_mstar
        ndsig[:, 1] *= delta_bins_ssfr

        _stats = calculate_sumstats_bin(
            mstar_histories,
            sfr_histories,
            p50[sel],
            weights_quench_bin,
            ndsig,
            bins_LO,
            bins_HI,
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

    return new_stats


def calculate_sumstats_bin(
    mstar_histories, sfr_histories, p50, weights_MS, ndsig, bins_LO, bins_HI
):

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

    ssfr = sfr_histories / mstar_histories

    mstar_histories = jnp.where(mstar_histories > 0.0, jnp.log10(mstar_histories), 0.0)
    sfr_histories = jnp.where(sfr_histories > 0.0, jnp.log10(sfr_histories), 0.0)

    ssfr = jnp.where(ssfr > 0.0, jnp.log10(ssfr), -50.0)

    mean_sm = jnp.average(mstar_histories, axis=0)
    mean_sfr_MS = jnp.average(sfr_histories, weights=weights_MS, axis=0)
    mean_sfr_Q = jnp.average(sfr_histories, weights=weights_Q, axis=0)

    mean_sm_early = jnp.average(mstar_histories, weights=weights_early, axis=0)
    mean_sm_late = jnp.average(mstar_histories, weights=weights_late, axis=0)

    variance_sm = jnp.average((mstar_histories - mean_sm[None, :]) ** 2, axis=0,)

    variance_sfr_MS = jnp.average(
        (sfr_histories - mean_sfr_MS[None, :]) ** 2, weights=weights_MS, axis=0,
    )
    variance_sfr_Q = jnp.average(
        (sfr_histories - mean_sfr_Q[None, :]) ** 2, weights=weights_Q, axis=0,
    )
    variance_sm_early = jnp.average(
        (mstar_histories - mean_sm[None, :]) ** 2, weights=weights_early, axis=0,
    )
    variance_sm_late = jnp.average(
        (mstar_histories - mean_sm[None, :]) ** 2, weights=weights_late, axis=0,
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

    weight = jnp.ones(len(sfr_histories))
    counts = calc_hist_mstar_ssfr(
        mstar_histories.T[t_sel_hists],
        ssfr.T[t_sel_hists],
        ndsig,
        weight,
        bins_LO,
        bins_HI,
    )

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
        counts,
    )
    return _out


MC_res_target = calculate_SMDPL_sumstats(
    t_table,
    logm0_binmids,
    logm0_bin_widths,
    mah_params_arr,
    u_fit_params_arr,
    p50_arr,
    bins_LO,
    bins_HI,
)

Nhalos = 3000
halo_data_MC = []
p50 = []
for i in range(len(logm0_binmids)):
    _sel = logmpeak > logm0_binmids[i] - logm0_bin_widths[i]
    _sel &= logmpeak < logm0_binmids[i] + logm0_bin_widths[i]
    replace = True if _sel.sum() < Nhalos else False
    sel = np.random.choice(np.arange(len(p50_arr))[_sel], Nhalos, replace=replace)
    halo_data_MC.append(mah_params_arr[sel])
    p50.append(p50_arr[sel])
halo_data_MC = np.concatenate(halo_data_MC, axis=0)[:, np.array([1, 2, 4, 5])]
p50 = np.concatenate(p50, axis=0)


path_json = "../diffstarpop/bestfit_diffstarpop_params_UM_hists_v4.json"
outputs = load_params(path_json)

n_histories = int(1e3)
# n_histories = int(1e4)
n_step = int(1e4)
step_size = 0.01


ndsig = np.ones((2 * n_histories, 2))
ndsig[:, 0] *= delta_bins_mstar
ndsig[:, 1] *= delta_bins_ssfr

_loss_data = (
    t_table,
    logm0_binmids,
    halo_data_MC.reshape(len(logm0_binmids), Nhalos, 4),
    p50.reshape(len(logm0_binmids), Nhalos),
    index_select,
    index_high,
    fstar_tdelay,
    ndsig,
    bins_LO,
    bins_HI,
    t_sel_hists,
    MC_res_target,
)
"""
init_guess = np.concatenate((
    dict_to_array(DEFAULT_SFH_PDF_QUENCH_PARAMS),
    dict_to_array(DEFAULT_SFH_PDF_MAINSEQ_PARAMS),
    dict_to_array(DEFAULT_R_QUENCH_PARAMS),
    dict_to_array(DEFAULT_R_MAINSEQ_PARAMS),
))
"""
init_guess = outputs[0].copy()
params_init = init_guess.copy()
ran_key = jran.PRNGKey(np.random.randint(2 ** 32))

loss_arr = np.zeros(n_step).astype("f4") + np.inf

opt_init, opt_update, get_params = jax_opt.adam(step_size)
opt_state = opt_init(params_init)

n_params = len(params_init)
params_arr = np.zeros((n_step, n_params)).astype("f4")

n_mah = 100
sampled_mahs_inds_arr = np.zeros((n_step, n_histories))
no_nan_grads_arr = np.zeros(n_step)
for istep in range(n_step):
    t0 = time.time()
    ran_key, subkey = jran.split(ran_key, 2)

    p = np.array(get_params(opt_state))

    loss = loss_hists_vmap(p, _loss_data, n_histories, ran_key)
    grads = loss_hists_vmap_deriv(p, _loss_data, n_histories, ran_key)

    no_nan_params = np.all(np.isfinite(p))
    no_nan_loss = np.isfinite(loss)
    no_nan_grads = np.all(np.isfinite(grads))
    if ~no_nan_params | ~no_nan_loss | ~no_nan_grads:
        # break
        if istep > 0:
            indx_best = np.nanargmin(loss_arr[:istep])
            best_fit_params = params_arr[indx_best]
            best_fit_loss = loss_arr[indx_best]
        else:
            best_fit_params = np.copy(p)
            best_fit_loss = 999.99
    else:
        params_arr[istep, :] = p
        loss_arr[istep] = loss
        opt_state = opt_update(istep, grads, opt_state)

    keys = jran.split(ran_key, 5)

    sampled_mahs_inds = jran.choice(keys[0], n_mah, shape=(n_histories,), replace=True)
    sampled_mahs_inds_arr[istep] = sampled_mahs_inds
    no_nan_grads_arr[istep] = ~no_nan_grads
    t1 = time.time()
    print(istep, loss, t1 - t0, no_nan_grads)
    # if ~no_nan_grads:
    #    break


indx_best = np.nanargmin(loss_arr)
best_fit_params = params_arr[indx_best]
loss = loss_arr[indx_best]
