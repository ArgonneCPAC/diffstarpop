import numpy as np
from jax import numpy as jnp, jit as jjit, vmap, random as jran, grad, lax
from jax.scipy.stats import multivariate_normal as jnorm
from collections import OrderedDict


# from diffmah.individual_halo_assembly import _calc_halo_history
# from diffmah.individual_halo_assembly import calc_halo_history


from diffstar.stars import (
    DEFAULT_SFR_PARAMS as DEFAULT_SFR_PARAMS_DICT,
    _get_unbounded_sfr_params,
)
from diffstar.stars import fstar_tools
from diffstar.constants import TODAY

from diffstar.utils import _jax_get_dt_array
from diffstar.quenching import DEFAULT_Q_PARAMS

from diffstarpop.star_wrappers import (
    sm_sfr_history_diffstar_scan_XsfhXmah_vmap,
)

from .pdf_quenched import DEFAULT_SFH_PDF_QUENCH_PARAMS
from .pdf_mainseq import DEFAULT_SFH_PDF_MAINSEQ_PARAMS
from .pdf_model_assembly_bias_shifts import (
    DEFAULT_R_QUENCH_PARAMS,
    DEFAULT_R_MAINSEQ_PARAMS,
)

from .monte_carlo_diff_halo_population import (
    sumstats_sfh_MIX_vmap,
    sumstats_sfh_MIX_p50_vmap,
    sumstats_sfh_with_hists,
    sumstats_sfh_with_hists_vmap,
    sumstats_sfh_with_hists_scan,
)
from functools import partial


DEFAULT_UNBOUND_SFR_PARAMS = _get_unbounded_sfr_params(
    *tuple(DEFAULT_SFR_PARAMS_DICT.values())
)
DEFAULT_UNBOUND_SFR_PARAMS_DICT = OrderedDict(
    zip(DEFAULT_SFR_PARAMS_DICT.keys(), DEFAULT_UNBOUND_SFR_PARAMS)
)

UH = DEFAULT_UNBOUND_SFR_PARAMS_DICT["indx_hi"]

SFH_PDF_Q_KEYS = list(DEFAULT_SFH_PDF_QUENCH_PARAMS.keys())
SFH_PDF_Q_VALUES = list(DEFAULT_SFH_PDF_QUENCH_PARAMS.values())

SFH_PDF_MS_KEYS = list(DEFAULT_SFH_PDF_MAINSEQ_PARAMS.keys())
SFH_PDF_MS_VALUES = list(DEFAULT_SFH_PDF_MAINSEQ_PARAMS.values())
# SFH_PDF_DIAG_KEYS = SFH_PDF_Q_KEYS[2 : 2 + 9 * 4]


N_PDF_Q = len(DEFAULT_SFH_PDF_QUENCH_PARAMS)
N_PDF_MS = len(DEFAULT_SFH_PDF_MAINSEQ_PARAMS)
N_R_Q = len(DEFAULT_R_QUENCH_PARAMS)
N_R_MS = len(DEFAULT_R_MAINSEQ_PARAMS)


@partial(jjit, static_argnames=["n_histories"])
def loss(params, loss_data, n_histories):
    (
        t_table,
        logm0_binmids,
        halo_data,
        p50,
        ran_key,
        index_select,
        index_high,
        fstar_tdelay,
        target_data,
        # target_data_weights,
    ) = loss_data
    (
        mean_sm_target,
        variance_sm_target,
        mean_fstar_MS_target,
        mean_fstar_Q_target,
        variance_fstar_MS_target,
        variance_fstar_Q_target,
        quench_frac_target,
    ) = target_data
    """
    (
        mean_sm_weights,
        variance_sm_weights,
        mean_fstar_MS_weights,
        mean_fstar_Q_weights,
        variance_fstar_MS_weights,
        variance_fstar_Q_weights,
        quench_frac_weights,
    ) = target_data_weights
    """
    _npar = 0
    pdf_q_params = params[_npar : _npar + N_PDF_Q]
    _npar += N_PDF_Q
    pdf_ms_params = params[_npar : _npar + N_PDF_MS]
    _npar += N_PDF_MS
    r_q_params = params[_npar : _npar + N_R_Q]
    _npar += N_R_Q
    r_ms_params = params[_npar : _npar + N_R_MS]

    _res = sumstats_sfh_MIX_vmap(
        t_table,
        logm0_binmids,
        halo_data,
        p50,
        n_histories,
        ran_key,
        index_select,
        index_high,
        fstar_tdelay,
        pdf_q_params,
        pdf_ms_params,
        r_q_params,
        r_ms_params,
    )
    (
        mean_sm,
        variance_sm,
        mean_fstar_MS,
        mean_fstar_Q,
        variance_fstar_MS,
        variance_fstar_Q,
        quench_frac,
    ) = _res

    loss = jnp.mean((mean_sm - mean_sm_target) ** 2)
    loss += jnp.mean((variance_sm - variance_sm_target) ** 2)

    loss += jnp.mean((mean_fstar_MS - mean_fstar_MS_target) ** 2)
    loss += jnp.mean((mean_fstar_Q - mean_fstar_Q_target) ** 2)
    loss += jnp.mean(
        (jnp.sqrt(variance_fstar_MS) - jnp.sqrt(variance_fstar_MS_target)) ** 2
    )
    loss += jnp.mean(
        (jnp.sqrt(variance_fstar_Q) - jnp.sqrt(variance_fstar_Q_target)) ** 2
    )
    loss += jnp.mean((quench_frac - quench_frac_target) ** 2)

    """
    loss = jnp.mean(mean_sm_weights * (mean_sm - mean_sm_target) ** 2)
    loss += jnp.mean(variance_sm_weights * (variance_sm - variance_sm_target) ** 2)

    loss += jnp.mean(
        mean_fstar_MS_weights * (mean_fstar_MS - mean_fstar_MS_target) ** 2
    )
    loss += jnp.mean(mean_fstar_Q_weights * (mean_fstar_Q - mean_fstar_Q_target) ** 2)
    loss += jnp.mean(
        variance_fstar_MS_weights * (variance_fstar_MS - variance_fstar_MS_target) ** 2
    )
    loss += jnp.mean(
        variance_fstar_Q_weights * (variance_fstar_Q - variance_fstar_Q_target) ** 2
    )
    loss += jnp.mean(quench_frac_weights * (quench_frac - quench_frac_target) ** 2)
    """

    return loss


loss_deriv = jjit(grad(loss, argnums=(0)), static_argnames=["n_histories"])


def loss_deriv_np(params, data, n_histories):
    return np.array(loss_deriv(params, data, n_histories)).astype(float)


@jjit
def mse(pred, targ):
    return jnp.mean((pred - targ) ** 2)


@jjit
def mse_var(pred, targ):
    return jnp.mean((jnp.log10(jnp.sqrt(pred)) - jnp.log10(jnp.sqrt(targ))) ** 2)


@jjit
def msew(pred, targ, weight):
    return jnp.mean(weight * (pred - targ) ** 2)


@jjit
def msew_var(pred, targ, weight):
    return jnp.mean(
        weight * (jnp.log10(jnp.sqrt(pred)) - jnp.log10(jnp.sqrt(targ))) ** 2
    )


@partial(jjit, static_argnames=["n_histories"])
def loss_p50(params, loss_data, n_histories, ran_key):
    (
        t_table,
        logm0_binmids,
        halo_data,
        p50,
        index_select,
        index_high,
        sfr_tdelay,
        target_data,
        # target_data_weights,
    ) = loss_data
    (
        mean_sm_target,
        variance_sm_target,
        mean_sfr_MS_target,
        mean_sfr_Q_target,
        variance_sfr_MS_target,
        variance_sfr_Q_target,
        quench_frac_target,
        mean_sm_early_target,
        mean_sm_late_target,
        variance_sm_early_target,
        variance_sm_late_target,
        quench_frac_early_target,
        quench_frac_late_target,
    ) = target_data

    _npar = 0
    pdf_q_params = params[_npar : _npar + N_PDF_Q]
    _npar += N_PDF_Q
    pdf_ms_params = params[_npar : _npar + N_PDF_MS]
    _npar += N_PDF_MS
    r_q_params = params[_npar : _npar + N_R_Q]
    _npar += N_R_Q
    r_ms_params = params[_npar : _npar + N_R_MS]

    _res = sumstats_sfh_MIX_p50_vmap(
        t_table,
        logm0_binmids,
        halo_data,
        p50,
        n_histories,
        ran_key,
        index_select,
        index_high,
        sfr_tdelay,
        pdf_q_params,
        pdf_ms_params,
        r_q_params,
        r_ms_params,
    )
    (
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
    ) = _res

    wQ = jnp.where(quench_frac_target > 0.01, 1.0, 0.0)
    wSF = jnp.where(quench_frac_target < 0.99, 1.0, 0.0)

    loss = mse(mean_sm, mean_sm_target)
    loss += mse_var(variance_sm, variance_sm_target)
    loss += msew(mean_sfr_MS, mean_sfr_MS_target, wSF)
    loss += msew(mean_sfr_Q, mean_sfr_Q_target, wQ)
    loss += msew_var(variance_sfr_MS, variance_sfr_MS_target, wSF)
    loss += msew_var(variance_sfr_Q, variance_sfr_Q_target, wQ)
    loss += mse(quench_frac, quench_frac_target)
    loss += mse(mean_sm_early, mean_sm_early_target)
    loss += mse(mean_sm_late, mean_sm_late_target)
    loss += mse_var(variance_sm_early, variance_sm_early_target)
    loss += mse_var(variance_sm_late, variance_sm_late_target)
    loss += mse(quench_frac_early, quench_frac_early_target)
    loss += mse(quench_frac_late, quench_frac_late_target)

    return loss


loss_p50_deriv = jjit(grad(loss_p50, argnums=(0)), static_argnames=["n_histories"])


def loss_p50_deriv_np(params, data, n_histories):
    return np.array(loss_p50_deriv(params, data, n_histories)).astype(float)


@partial(jjit, static_argnames=["n_histories"])
def loss_hists(params, loss_data, n_histories, ran_key):
    (
        t_table,
        logm0_binmids,
        halo_data,
        p50,
        index_select,
        index_high,
        sfr_tdelay,
        target_data,
        # target_data_weights,
    ) = loss_data
    (
        mean_sm_target,
        variance_sm_target,
        mean_sfr_MS_target,
        mean_sfr_Q_target,
        variance_sfr_MS_target,
        variance_sfr_Q_target,
        quench_frac_target,
        mean_sm_early_target,
        mean_sm_late_target,
        variance_sm_early_target,
        variance_sm_late_target,
        quench_frac_early_target,
        quench_frac_late_target,
        counts_target,
    ) = target_data

    _npar = 0
    pdf_q_params = params[_npar : _npar + N_PDF_Q]
    _npar += N_PDF_Q
    pdf_ms_params = params[_npar : _npar + N_PDF_MS]
    _npar += N_PDF_MS
    r_q_params = params[_npar : _npar + N_R_Q]
    _npar += N_R_Q
    r_ms_params = params[_npar : _npar + N_R_MS]

    _res = sumstats_sfh_with_hists_vmap(
        t_table,
        logm0_binmids,
        halo_data,
        p50,
        n_histories,
        ran_key,
        index_select,
        index_high,
        sfr_tdelay,
        pdf_q_params,
        pdf_ms_params,
        r_q_params,
        r_ms_params,
    )
    (
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
    ) = _res

    wQ = jnp.where(quench_frac_target > 0.01, 1.0, 0.0)
    wSF = jnp.where(quench_frac_target < 0.99, 1.0, 0.0)

    loss = mse(mean_sm, mean_sm_target)
    loss += mse_var(variance_sm, variance_sm_target)
    loss += msew(mean_sfr_MS, mean_sfr_MS_target, wSF)
    loss += msew(mean_sfr_Q, mean_sfr_Q_target, wQ)
    loss += msew_var(variance_sfr_MS, variance_sfr_MS_target, wSF)
    loss += msew_var(variance_sfr_Q, variance_sfr_Q_target, wQ)
    loss += mse(quench_frac, quench_frac_target)
    loss += mse(mean_sm_early, mean_sm_early_target)
    loss += mse(mean_sm_late, mean_sm_late_target)
    loss += mse_var(variance_sm_early, variance_sm_early_target)
    loss += mse_var(variance_sm_late, variance_sm_late_target)
    loss += mse(quench_frac_early, quench_frac_early_target)
    loss += mse(quench_frac_late, quench_frac_late_target)

    loss += mse(counts, counts_target)

    return loss


loss_hists_deriv = jjit(grad(loss_hists, argnums=(0)), static_argnames=["n_histories"])


def loss_hists_deriv_np(params, data, n_histories):
    return np.array(loss_hists_deriv(params, data, n_histories)).astype(float)


@jjit
def mse_arch(pred, targ):
    k = 10**5.0
    pred_arch = jnp.arcsinh(pred / 2 * k)
    targ_arch = jnp.arcsinh(targ / 2 * k)
    return jnp.mean((pred_arch - targ_arch) ** 2)


@partial(jjit, static_argnames=["n_histories"])
def loss_hists_vmap(params, loss_data, n_histories, ran_key):
    (
        t_table,
        logm0_binmids,
        halo_data,
        p50,
        index_select,
        index_high,
        sfr_tdelay,
        ndsig,
        bins_LO,
        bins_HI,
        t_sel_hists,
        target_data,
        # target_data_weights,
    ) = loss_data
    (
        mean_sm_target,
        variance_sm_target,
        mean_sfr_MS_target,
        mean_sfr_Q_target,
        variance_sfr_MS_target,
        variance_sfr_Q_target,
        quench_frac_target,
        mean_sm_early_target,
        mean_sm_late_target,
        variance_sm_early_target,
        variance_sm_late_target,
        quench_frac_early_target,
        quench_frac_late_target,
        counts_target,
    ) = target_data

    _npar = 0
    pdf_q_params = params[_npar : _npar + N_PDF_Q]
    _npar += N_PDF_Q
    pdf_ms_params = params[_npar : _npar + N_PDF_MS]
    _npar += N_PDF_MS
    r_q_params = params[_npar : _npar + N_R_Q]
    _npar += N_R_Q
    r_ms_params = params[_npar : _npar + N_R_MS]

    _res = sumstats_sfh_with_hists_vmap(
        t_table,
        logm0_binmids,
        halo_data,
        p50,
        n_histories,
        ran_key,
        index_select,
        index_high,
        sfr_tdelay,
        ndsig,
        bins_LO,
        bins_HI,
        t_sel_hists,
        pdf_q_params,
        pdf_ms_params,
        r_q_params,
        r_ms_params,
    )
    (
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
    ) = _res

    wQ = jnp.where(quench_frac_target > 0.01, 1.0, 0.0)
    wSF = jnp.where(quench_frac_target < 0.99, 1.0, 0.0)

    loss = mse(mean_sm, mean_sm_target)
    loss += mse(mean_sm_early, mean_sm_early_target)
    loss += mse(mean_sm_late, mean_sm_late_target)
    """
    loss += mse_var(variance_sm, variance_sm_target)
    loss += msew(mean_sfr_MS, mean_sfr_MS_target, wSF)
    loss += msew(mean_sfr_Q, mean_sfr_Q_target, wQ)
    loss += msew_var(variance_sfr_MS, variance_sfr_MS_target, wSF)
    loss += msew_var(variance_sfr_Q, variance_sfr_Q_target, wQ)
    loss += mse(quench_frac, quench_frac_target)

    loss += mse_var(variance_sm_early, variance_sm_early_target)
    loss += mse_var(variance_sm_late, variance_sm_late_target)
    loss += mse(quench_frac_early, quench_frac_early_target)
    loss += mse(quench_frac_late, quench_frac_late_target)
    """
    loss += 1e4 * mse(counts, counts_target)
    # loss += (1.0 / 20.0) * mse_arch(counts, counts_target)

    return loss


@partial(jjit, static_argnames=["n_histories"])
def loss_hists_scan(params, loss_data, n_histories, ran_key):
    (
        t_table,
        logm0_binmids,
        halo_data,
        p50,
        index_select,
        index_high,
        sfr_tdelay,
        ndsig,
        bins_LO,
        bins_HI,
        t_sel_hists,
        target_data,
        # target_data_weights,
    ) = loss_data
    (
        mean_sm_target,
        variance_sm_target,
        mean_sfr_MS_target,
        mean_sfr_Q_target,
        variance_sfr_MS_target,
        variance_sfr_Q_target,
        quench_frac_target,
        mean_sm_early_target,
        mean_sm_late_target,
        variance_sm_early_target,
        variance_sm_late_target,
        quench_frac_early_target,
        quench_frac_late_target,
        counts_target,
    ) = target_data

    _npar = 0
    pdf_q_params = params[_npar : _npar + N_PDF_Q]
    _npar += N_PDF_Q
    pdf_ms_params = params[_npar : _npar + N_PDF_MS]
    _npar += N_PDF_MS
    r_q_params = params[_npar : _npar + N_R_Q]
    _npar += N_R_Q
    r_ms_params = params[_npar : _npar + N_R_MS]

    _res = sumstats_sfh_with_hists_scan(
        t_table,
        logm0_binmids,
        halo_data,
        p50,
        n_histories,
        ran_key,
        index_select,
        index_high,
        sfr_tdelay,
        ndsig,
        bins_LO,
        bins_HI,
        t_sel_hists,
        pdf_q_params,
        pdf_ms_params,
        r_q_params,
        r_ms_params,
    )
    (
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
    ) = _res

    wQ = jnp.where(quench_frac_target > 0.01, 1.0, 0.0)
    wSF = jnp.where(quench_frac_target < 0.99, 1.0, 0.0)

    loss = mse(mean_sm, mean_sm_target)
    loss += mse(mean_sm_early, mean_sm_early_target)
    loss += mse(mean_sm_late, mean_sm_late_target)
    """
    loss += mse_var(variance_sm, variance_sm_target)
    loss += msew(mean_sfr_MS, mean_sfr_MS_target, wSF)
    loss += msew(mean_sfr_Q, mean_sfr_Q_target, wQ)
    loss += msew_var(variance_sfr_MS, variance_sfr_MS_target, wSF)
    loss += msew_var(variance_sfr_Q, variance_sfr_Q_target, wQ)
    loss += mse(quench_frac, quench_frac_target)

    loss += mse_var(variance_sm_early, variance_sm_early_target)
    loss += mse_var(variance_sm_late, variance_sm_late_target)
    loss += mse(quench_frac_early, quench_frac_early_target)
    loss += mse(quench_frac_late, quench_frac_late_target)
    """
    loss += 1e4 * mse(counts, counts_target)
    # loss += (1.0 / 20.0) * mse_arch(counts, counts_target)

    return loss


loss_hists_vmap_deriv = jjit(
    grad(loss_hists_vmap, argnums=(0)), static_argnames=["n_histories"]
)
loss_hists_scan_deriv = jjit(
    grad(loss_hists_scan, argnums=(0)), static_argnames=["n_histories"]
)


def loss_hists_vmap_deriv_np(params, data, n_histories, ran_key):
    return np.array(loss_hists_vmap_deriv(params, data, n_histories, ran_key)).astype(
        float
    )


def loss_hists_scan_deriv_np(params, data, n_histories, ran_key):
    return np.array(loss_hists_scan_deriv(params, data, n_histories, ran_key)).astype(
        float
    )


def get_loss_p50_data():
    # path = "/Users/alarcon/Documents/diffmah_data/SMDPL/"
    path = "/lcrc/project/halotools/alarcon/data/"

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

    fstar_tdelay = 1.0
    index_select, index_high = fstar_tools(t_table, fstar_tdelay=fstar_tdelay)

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
        index_select,
        index_high,
        fstar_tdelay,
        MC_res_target,
    )
    return loss_data


def calculate_SMDPL_sumstats(
    t_table,
    logm0_binmids,
    logm0_bin_widths,
    mah_params,
    u_fit_params,
    p50,
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
