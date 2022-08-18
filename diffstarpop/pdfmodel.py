"""
"""
import numpy as np
from jax import numpy as jnp, jit as jjit, vmap, random as jran
from jax.scipy.stats import multivariate_normal as jnorm
from collections import OrderedDict


# from diffmah.individual_halo_assembly import _calc_halo_history
# from diffmah.individual_halo_assembly import calc_halo_history


from diffstar.stars import (
    DEFAULT_SFR_PARAMS as DEFAULT_SFR_PARAMS_DICT,
    _get_unbounded_sfr_params,
)

from diffstar.utils import _jax_get_dt_array
from diffstar.quenching import DEFAULT_Q_PARAMS

from .pdf_diffmah import get_diffmah_grid
from .latin_hypercube import latin_hypercube_from_cov
from .star_wrappers import compute_histories_on_grids_Q, compute_histories_on_grids_MS
from .pdf_quenched import get_smah_means_and_covs_quench, DEFAULT_SFH_PDF_QUENCH_PARAMS
from .pdf_mainseq import get_smah_means_and_covs_mainseq, DEFAULT_SFH_PDF_MAINSEQ_PARAMS


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

N_MS_PARAMS = len(DEFAULT_UNBOUND_SFR_PARAMS)
N_Q_PARAMS = len(DEFAULT_Q_PARAMS)


@jjit
def _multivariate_normal_pdf_kernel(sfh_grid_params, mu, cov):
    return jnorm.pdf(sfh_grid_params, mu, cov)


_get_pdf_weights_kern = jjit(vmap(_multivariate_normal_pdf_kernel, in_axes=(0, 0, 0)))


@jjit
def compute_target_sumstats_from_histories(
    weights_pdf, weights_quench_bin, mstar_histories, sfr_histories, fstar_histories
):
    """
    Parameters
    ----------
    weights : ndarray of shape (n_m0, n_sfh_grid)
        Description.
    weights_quench_bin : ndarray of shape (n_m0, n_sfh_grid, n_per_m0, n_t)
        Description.
    mstar_histories : ndarray of shape (n_m0, n_sfh_grid, n_per_m0, n_t)
        Description.
    sfr_histories : ndarray of shape (n_m0, n_sfh_grid, n_per_m0, n_t)
        Description.
    fstar_histories : ndarray of shape (n_m0, n_sfh_grid, n_per_m0, n_t)
        Description.

    Returns
    -------

    """
    NHALO = mstar_histories.shape[2]
    w_sum = 1.0 / jnp.sum(weights_pdf, axis=1)
    mstar_histories = jnp.log10(mstar_histories)
    # sfr_histories = jnp.log10(sfr_histories)
    # fstar_histories = jnp.log10(fstar_histories)

    fstar_MS = fstar_histories * weights_quench_bin
    fstar_Q = fstar_histories * (1.0 - weights_quench_bin)
    fstar_MS = jnp.where(fstar_MS > 0.0, jnp.log10(fstar_MS), 0.0)
    fstar_Q = jnp.where(fstar_Q > 0.0, jnp.log10(fstar_Q), 0.0)

    NHALO_MS = jnp.einsum("ab,abcd->ad", weights_pdf, weights_quench_bin)
    NHALO_Q = jnp.einsum("ab,abcd->ad", weights_pdf, 1.0 - weights_quench_bin)
    quench_frac = NHALO_Q / (NHALO_Q + NHALO_MS)

    NHALO_MS = jnp.where(NHALO_MS > 0.0, NHALO_MS, 1.0)
    NHALO_Q = jnp.where(NHALO_Q > 0.0, NHALO_Q, 1.0)

    mean_sm = jnp.einsum("ab,a,abcd->ad", weights_pdf, w_sum, mstar_histories) / NHALO
    # mean_sfr = jnp.einsum("ab,abcd->ad", weights_pdf, sfr_histories)
    # mean_fstar = jnp.einsum("ab,a,abcd->ad", weights_pdf, w_sum, fstar_histories) / NHALO
    mean_fstar_MS = jnp.einsum("ab,abcd->ad", weights_pdf, fstar_MS) / NHALO_MS
    mean_fstar_Q = jnp.einsum("ab,abcd->ad", weights_pdf, fstar_Q) / NHALO_Q

    delta_sm = (mstar_histories - mean_sm[:, None, None, :]) ** 2
    # delta_sfr = (sfr_histories - mean_sfr) ** 2
    # delta_fstar = (fstar_histories - mean_fstar[:, None, None, :]) ** 2
    delta_fstar_MS = (fstar_MS - mean_fstar_MS[:, None, None, :]) ** 2
    delta_fstar_Q = (fstar_Q - mean_fstar_Q[:, None, None, :]) ** 2

    # delta_fstar_MS = jnp.where(fstar_MS == 0.0, 0.0, delta_fstar_MS)
    # delta_fstar_Q = jnp.where(fstar_Q == 0.0, 0.0, delta_fstar_Q)

    delta_fstar_MS = delta_fstar_MS * weights_quench_bin
    delta_fstar_Q = delta_fstar_Q * (1.0 - weights_quench_bin)

    variance_sm = jnp.einsum("ab,a,abcd->ad", weights_pdf, w_sum, delta_sm) / NHALO
    # variance_sfr = jnp.sum(delta_sfr * w, axis=(0, 1, 2, 3, 4))
    # variance_fstar = jnp.einsum("ab,a,abcd->ad", weights_pdf, w_sum, delta_fstar) / NHALO
    variance_fstar_MS = (
        jnp.einsum("ab,abcd->ad", weights_pdf, delta_fstar_MS) / NHALO_MS
    )
    variance_fstar_Q = jnp.einsum("ab,abcd->ad", weights_pdf, delta_fstar_Q) / NHALO_Q

    _out = (
        mean_sm,
        variance_sm,
        mean_fstar_MS,
        mean_fstar_Q,
        variance_fstar_MS,
        variance_fstar_Q,
        quench_frac,
    )
    return _out


def get_binned_means_and_covs_Q(pdf_model_pars, logm0_bins):
    """Calculate the mean and covariance in SFH parameter space for each input M0.

    Parameters
    ----------
    pdf_model_pars : ndarray of shape (n_sfh_params, )
        Description.
    logm0_bins : ndarray of shape (n_m0_bins, )
        Description.
    Returns
    -------
    means : ndarray of shape (n_m0_bins, n_sfh_params)
        Description.
    covs : ndarray of shape (n_m0_bins, n_sfh_params, n_sfh_params)
        Description.
    """
    subset_dict = OrderedDict(
        [(key, val) for (key, val) in zip(SFH_PDF_Q_KEYS, pdf_model_pars)]
    )

    pdf_model_params_dict = DEFAULT_SFH_PDF_QUENCH_PARAMS.copy()
    pdf_model_params_dict.update(subset_dict)

    _res = get_smah_means_and_covs_quench(logm0_bins, **pdf_model_params_dict)
    frac_quench, means_quench, covs_quench = _res
    return frac_quench, means_quench, covs_quench


def get_sfh_param_grid_Q(pdf_model_pars, logm0_bins, sfh_lh_sig, n_sfh_param_grid):
    fracs_logm0_bins, means_logm0_bins, covs_logm0_bins = get_binned_means_and_covs_Q(
        pdf_model_pars, logm0_bins
    )
    sfh_param_grid = [
        latin_hypercube_from_cov(mu, cov, sfh_lh_sig, n_sfh_param_grid)
        for mu, cov in zip(means_logm0_bins, covs_logm0_bins)
    ]
    return fracs_logm0_bins, np.array(sfh_param_grid)


def get_param_grids_Q(
    tarr,
    n_halos_per_bin,
    jran_key,
    logm0_binmids,
    logm0_bin_widths,
    logm0_halos,
    mah_tauc_halos,
    mah_early_halos,
    mah_late_halos,
    pdf_model_pars,
    sfh_lh_sig,
    n_sfh_param_grid,
    t0,
):
    dmhdt_grid, log_mah_grid = get_diffmah_grid(
        tarr,
        n_halos_per_bin,
        jran_key,
        logm0_binmids,
        logm0_bin_widths,
        logm0_halos,
        mah_tauc_halos,
        mah_early_halos,
        mah_late_halos,
        t0,
    )[:2]

    n_m0_bins, n_times = logm0_binmids.size, tarr.size
    _s = (n_m0_bins, n_halos_per_bin, n_times)
    log_mah_grids = np.array(log_mah_grid).reshape(_s)
    dmhdt_grids = np.array(dmhdt_grid).reshape(_s)

    fracs_logm0, sfh_param_grids = get_sfh_param_grid_Q(
        pdf_model_pars, logm0_binmids, sfh_lh_sig, n_sfh_param_grid
    )
    return dmhdt_grids, log_mah_grids, sfh_param_grids, fracs_logm0


def _get_default_pdf_SFH_prediction_quench(
    sfh_lh_sig,
    t_table,
    n_sfh_param_grid,
    logm0_binmids,
    logm0_bin_widths,
    n_halos_per_bin,
    halo_data,
    fstar_tdelay,
    pdf_model_params=SFH_PDF_Q_VALUES,
):

    index_high = np.searchsorted(t_table, t_table - fstar_tdelay)
    _mask = t_table > fstar_tdelay + fstar_tdelay / 2.0
    index_select = np.arange(len(t_table))[_mask]
    index_high = index_high[_mask]

    jran_key = jran.PRNGKey(0)
    t0 = t_table[-1]
    lgt_table = jnp.log10(t_table)
    dt_table = _jax_get_dt_array(t_table)
    logm0_halos, mah_tauc_halos, mah_early_halos, mah_late_halos = halo_data

    dmhdt_grids, log_mah_grids, sfh_param_grids, fracs_logm0 = get_param_grids_Q(
        t_table,
        n_halos_per_bin,
        jran_key,
        logm0_binmids,
        logm0_bin_widths,
        logm0_halos,
        mah_tauc_halos,
        mah_early_halos,
        mah_late_halos,
        pdf_model_params,
        sfh_lh_sig,
        n_sfh_param_grid,
        t0,
    )

    fracs, means, covs = get_binned_means_and_covs_Q(pdf_model_params, logm0_binmids)
    _res = compute_histories_on_grids_Q(
        lgt_table,
        dt_table,
        index_select,
        index_high,
        fstar_tdelay,
        dmhdt_grids,
        log_mah_grids,
        sfh_param_grids,
    )
    mstar_histories, sfr_histories, fstar_histories = _res

    sfstar_histories = fstar_histories / mstar_histories[:, :, :, index_select]

    weights_quench_bin = jnp.where(sfstar_histories > 1e-11, 1.0, 0.0)

    weights = _get_pdf_weights_kern(sfh_param_grids, means, covs)

    return compute_target_sumstats_from_histories(
        weights, weights_quench_bin, mstar_histories, sfr_histories, fstar_histories
    )


def get_binned_means_and_covs_MS(pdf_model_pars, logm0_bins):
    """Calculate the mean and covariance in SFH parameter space for each input M0.

    Parameters
    ----------
    pdf_model_pars : ndarray of shape (n_sfh_params, )
        Description.
    logm0_bins : ndarray of shape (n_m0_bins, )
        Description.
    Returns
    -------
    means : ndarray of shape (n_m0_bins, n_sfh_params)
        Description.
    covs : ndarray of shape (n_m0_bins, n_sfh_params, n_sfh_params)
        Description.
    """
    subset_dict = OrderedDict(
        [(key, val) for (key, val) in zip(SFH_PDF_MS_KEYS, pdf_model_pars)]
    )

    pdf_model_params_dict = DEFAULT_SFH_PDF_MAINSEQ_PARAMS.copy()
    pdf_model_params_dict.update(subset_dict)

    _res = get_smah_means_and_covs_mainseq(logm0_bins, **pdf_model_params_dict)
    means_mainseq, covs_mainseq = _res
    return means_mainseq, covs_mainseq


def get_sfh_param_grid_MS(pdf_model_pars, logm0_bins, sfh_lh_sig, n_sfh_param_grid):
    means_logm0_bins, covs_logm0_bins = get_binned_means_and_covs_MS(
        pdf_model_pars, logm0_bins
    )
    sfh_param_grid = [
        latin_hypercube_from_cov(mu, cov, sfh_lh_sig, n_sfh_param_grid)
        for mu, cov in zip(means_logm0_bins, covs_logm0_bins)
    ]
    return np.array(sfh_param_grid)


def get_param_grids_MS(
    tarr,
    n_halos_per_bin,
    jran_key,
    logm0_binmids,
    logm0_bin_widths,
    logm0_halos,
    mah_tauc_halos,
    mah_early_halos,
    mah_late_halos,
    pdf_model_pars,
    sfh_lh_sig,
    n_sfh_param_grid,
    t0,
):
    dmhdt_grid, log_mah_grid = get_diffmah_grid(
        tarr,
        n_halos_per_bin,
        jran_key,
        logm0_binmids,
        logm0_bin_widths,
        logm0_halos,
        mah_tauc_halos,
        mah_early_halos,
        mah_late_halos,
        t0,
    )[:2]

    n_m0_bins, n_times = logm0_binmids.size, tarr.size
    _s = (n_m0_bins, n_halos_per_bin, n_times)
    log_mah_grids = np.array(log_mah_grid).reshape(_s)
    dmhdt_grids = np.array(dmhdt_grid).reshape(_s)

    sfh_param_grids = get_sfh_param_grid_MS(
        pdf_model_pars, logm0_binmids, sfh_lh_sig, n_sfh_param_grid
    )
    return dmhdt_grids, log_mah_grids, sfh_param_grids


def _get_default_pdf_SFH_prediction_mainseq(
    sfh_lh_sig,
    t_table,
    n_sfh_param_grid,
    logm0_binmids,
    logm0_bin_widths,
    n_halos_per_bin,
    halo_data,
    fstar_tdelay,
    pdf_model_params=SFH_PDF_MS_VALUES,
):

    index_high = np.searchsorted(t_table, t_table - fstar_tdelay)
    _mask = t_table > fstar_tdelay + fstar_tdelay / 2.0
    index_select = np.arange(len(t_table))[_mask]
    index_high = index_high[_mask]

    jran_key = jran.PRNGKey(0)
    t0 = t_table[-1]
    lgt_table = jnp.log10(t_table)
    dt_table = _jax_get_dt_array(t_table)
    logm0_halos, mah_tauc_halos, mah_early_halos, mah_late_halos = halo_data

    dmhdt_grids, log_mah_grids, sfh_param_grids = get_param_grids_MS(
        t_table,
        n_halos_per_bin,
        jran_key,
        logm0_binmids,
        logm0_bin_widths,
        logm0_halos,
        mah_tauc_halos,
        mah_early_halos,
        mah_late_halos,
        pdf_model_params,
        sfh_lh_sig,
        n_sfh_param_grid,
        t0,
    )

    means, covs = get_binned_means_and_covs_MS(pdf_model_params, logm0_binmids)
    _res = compute_histories_on_grids_MS(
        lgt_table,
        dt_table,
        index_select,
        index_high,
        fstar_tdelay,
        dmhdt_grids,
        log_mah_grids,
        sfh_param_grids,
    )
    mstar_histories, sfr_histories, fstar_histories = _res

    sfstar_histories = fstar_histories / mstar_histories[:, :, :, index_select]

    weights_mainseq_bin = jnp.where(sfstar_histories > 1e-11, 1.0, 0.0)

    weights = _get_pdf_weights_kern(sfh_param_grids, means, covs)

    return compute_target_sumstats_from_histories(
        weights, weights_mainseq_bin, mstar_histories, sfr_histories, fstar_histories
    )


def get_param_grids(
    tarr,
    n_halos_per_bin,
    jran_key,
    logm0_binmids,
    logm0_bin_widths,
    logm0_halos,
    mah_tauc_halos,
    mah_early_halos,
    mah_late_halos,
    pdf_model_pars_MS,
    pdf_model_pars_Q,
    sfh_lh_sig,
    n_sfh_param_grid,
    t0,
):
    dmhdt_grid, log_mah_grid = get_diffmah_grid(
        tarr,
        n_halos_per_bin,
        jran_key,
        logm0_binmids,
        logm0_bin_widths,
        logm0_halos,
        mah_tauc_halos,
        mah_early_halos,
        mah_late_halos,
        t0,
    )[:2]

    n_m0_bins, n_times = logm0_binmids.size, tarr.size
    _s = (n_m0_bins, n_halos_per_bin, n_times)
    log_mah_grids = np.array(log_mah_grid).reshape(_s)
    dmhdt_grids = np.array(dmhdt_grid).reshape(_s)

    sfh_param_grids_MS = get_sfh_param_grid_MS(
        pdf_model_pars_MS, logm0_binmids, sfh_lh_sig, n_sfh_param_grid
    )
    fracs_logm0, sfh_param_grids_Q = get_sfh_param_grid_Q(
        pdf_model_pars_Q, logm0_binmids, sfh_lh_sig, n_sfh_param_grid
    )
    return (
        dmhdt_grids,
        log_mah_grids,
        sfh_param_grids_MS,
        sfh_param_grids_Q,
        fracs_logm0,
    )


def get_default_pdf_SFH_prediction(
    sfh_lh_sig,
    t_table,
    n_sfh_param_grid,
    logm0_binmids,
    logm0_bin_widths,
    n_halos_per_bin,
    halo_data,
    fstar_tdelay,
    pdf_model_params_MS=SFH_PDF_MS_VALUES,
    pdf_model_params_Q=SFH_PDF_Q_VALUES,
):

    index_high = np.searchsorted(t_table, t_table - fstar_tdelay)
    _mask = t_table > fstar_tdelay + fstar_tdelay / 2.0
    index_select = np.arange(len(t_table))[_mask]
    index_high = index_high[_mask]

    jran_key = jran.PRNGKey(0)
    t0 = t_table[-1]
    lgt_table = jnp.log10(t_table)
    dt_table = _jax_get_dt_array(t_table)
    logm0_halos, mah_tauc_halos, mah_early_halos, mah_late_halos = halo_data

    _res = get_param_grids(
        t_table,
        n_halos_per_bin,
        jran_key,
        logm0_binmids,
        logm0_bin_widths,
        logm0_halos,
        mah_tauc_halos,
        mah_early_halos,
        mah_late_halos,
        pdf_model_params_MS,
        pdf_model_params_Q,
        sfh_lh_sig,
        n_sfh_param_grid,
        t0,
    )
    (
        dmhdt_grids,
        log_mah_grids,
        sfh_param_grids_MS,
        sfh_param_grids_Q,
        fracs_logm0,
    ) = _res

    # Main sequence histories and weights
    means_MS, covs_MS = get_binned_means_and_covs_MS(pdf_model_params_MS, logm0_binmids)
    _res = compute_histories_on_grids_MS(
        lgt_table,
        dt_table,
        index_select,
        index_high,
        fstar_tdelay,
        dmhdt_grids,
        log_mah_grids,
        sfh_param_grids_MS,
    )
    mstar_histories_MS, sfr_histories_MS, fstar_histories_MS = _res
    sfstar_histories_MS = fstar_histories_MS / mstar_histories_MS[:, :, :, index_select]

    weights_MS_bin = jnp.where(sfstar_histories_MS > 1e-11, 1.0, 0.0)
    weights_MS = _get_pdf_weights_kern(sfh_param_grids_MS, means_MS, covs_MS)

    # Quenched histories and weights
    fracs_Q, means_Q, covs_Q = get_binned_means_and_covs_Q(
        pdf_model_params_Q, logm0_binmids
    )
    _res = compute_histories_on_grids_Q(
        lgt_table,
        dt_table,
        index_select,
        index_high,
        fstar_tdelay,
        dmhdt_grids,
        log_mah_grids,
        sfh_param_grids_Q,
    )
    mstar_histories_Q, sfr_histories_Q, fstar_histories_Q = _res
    sfstar_histories_Q = fstar_histories_Q / mstar_histories_Q[:, :, :, index_select]
    weights_Q_bin = jnp.where(sfstar_histories_Q > 1e-11, 1.0, 0.0)
    weights_Q = _get_pdf_weights_kern(sfh_param_grids_Q, means_Q, covs_Q)

    mstar_histories = jnp.concatenate((mstar_histories_MS, mstar_histories_Q), axis=1)
    sfr_histories = jnp.concatenate((sfr_histories_MS, sfr_histories_Q), axis=1)
    fstar_histories = jnp.concatenate((fstar_histories_MS, fstar_histories_Q), axis=1)

    weights_MS = jnp.einsum("ij,i->ij", weights_MS, (1.0 - fracs_Q))
    weights_Q = jnp.einsum("ij,i->ij", weights_Q, fracs_Q)

    weights = jnp.concatenate((weights_MS, weights_Q), axis=1)
    weights_bin = jnp.concatenate((weights_MS_bin, weights_Q_bin), axis=1)

    return compute_target_sumstats_from_histories(
        weights, weights_bin, mstar_histories, sfr_histories, fstar_histories
    )
