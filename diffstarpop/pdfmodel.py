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

from .pdf_diffmah import (
    get_diffmah_grid,
    get_binned_halo_sample,
    get_binned_halo_sample_p50,
)
from .latin_hypercube import latin_hypercube_from_cov
from .star_wrappers import (
    compute_histories_on_grids_diffstar_vmap_Xsfh_vmap_Xmah_vmap,
    compute_histories_on_grids_diffstar_vmap_Xsfh_scan_Xmah_scan,
    compute_histories_on_grids_diffstar_scan_Xsfh_scan_Xmah_scan,
    compute_histories_on_grids_diffstar_scan_Xsfh_vmap_Xmah_vmap,
    compute_histories_on_grids_MS_diffstar_vmap_Xsfh_vmap_Xmah_vmap,
    compute_histories_on_grids_MS_diffstar_vmap_Xsfh_scan_Xmah_scan,
    compute_histories_on_grids_MS_diffstar_scan_Xsfh_scan_Xmah_scan,
    compute_histories_on_grids_MS_diffstar_scan_Xsfh_vmap_Xmah_vmap,
)
from .pdf_quenched import get_smah_means_and_covs_quench, DEFAULT_SFH_PDF_QUENCH_PARAMS
from .pdf_mainseq import get_smah_means_and_covs_mainseq, DEFAULT_SFH_PDF_MAINSEQ_PARAMS
from .pdf_model_assembly_bias_shifts import (
    DEFAULT_R_QUENCH_PARAMS,
    DEFAULT_R_MAINSEQ_PARAMS,
    _get_slopes_quench,
    _get_slopes_mainseq,
    _get_shift_to_PDF_mean,
)

from scipy.stats import qmc
from scipy.stats import norm
from jax.scipy import stats as stats_jax


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

_kern1 = jjit(vmap(_multivariate_normal_pdf_kernel, in_axes=(0, None, None)))
_kern2 = jjit(vmap(_kern1, in_axes=(0, 0, None)))
_get_pdf_weights_kern_withR = jjit(vmap(_kern2, in_axes=(0, 0, 0)))


@jjit
def compute_target_sumstats_from_histories(
    weights_pdf, weights_quench_bin, mstar_histories, sfr_histories, fstar_histories
):
    """
    Compute differentiable summary statistics from pdf-weighting the histories
    computed in a latin hypercube grid (or other uniform volume schemes).

    Parameters
    ----------
    weights_pdf : ndarray of shape (n_m0, n_sfh_grid)
        PDF weight of the latin hypercube grid points.
    weights_quench_bin : ndarray of shape (n_m0, n_sfh_grid, n_per_m0, n_t)
        Weight array indicating when galaxy history is quenched.
            0: sSFR(t) < 1e-11
            1: sSFR(t) > 1e-11
    mstar_histories : ndarray of shape (n_m0, n_sfh_grid, n_per_m0, n_t)
        SMH at each latin hypercube grid point.
    sfr_histories : ndarray of shape (n_m0, n_sfh_grid, n_per_m0, n_t)
        SFH at each latin hypercube grid point.
    fstar_histories : ndarray of shape (n_m0, n_sfh_grid, n_per_m0, n_t)
        Fstar history at each latin hypercube grid point.

    Returns
    -------
    mean_sm : ndarray of shape (n_m0, n_t)
        Average log10 Stellar Mass.
    variance_sm : ndarray of shape (n_m0, n_t)
        Variance of log10 Stellar Mass.
    mean_fstar_MS : ndarray of shape (n_m0, n_t)
        Average fstar (average SFH within some timescale) for main sequence galaxies.
    mean_fstar_Q : ndarray of shape (n_m0, n_t)
        Average fstar (average SFH within some timescale) for quenched galaxies.
    variance_fstar_MS : ndarray of shape (n_m0, n_t)
        Variance of fstar (average SFH within some timescale) for MS galaxies.
    variance_fstar_Q : ndarray of shape (n_m0, n_t)
        Variance of fstar (average SFH within some timescale) for Q galaxies.
    quench_frac : ndarray of shape (n_m0, n_t)
        Fraction of quenched galaxies.
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


@jjit
def compute_target_sumstats_from_histories_diffhaloweights(
    weights_pdf, weights_quench_bin, mstar_histories, sfr_histories, fstar_histories
):
    """
    Compute differentiable summary statistics from pdf-weighting the histories
    computed in a latin hypercube grid (or other uniform volume schemes).

    Parameters
    ----------
    weights : ndarray of shape (n_m0, n_sfh_grid, n_per_m0)
        PDF weight of the latin hypercube grid points.
    weights_quench_bin : ndarray of shape (n_m0, n_sfh_grid, n_per_m0, n_t)
        Weight array indicating when galaxy history is quenched.
            0: sSFR(t) < 1e-11
            1: sSFR(t) > 1e-11
    mstar_histories : ndarray of shape (n_m0, n_sfh_grid, n_per_m0, n_t)
        SMH at each latin hypercube grid point.
    sfr_histories : ndarray of shape (n_m0, n_sfh_grid, n_per_m0, n_t)
        SFH at each latin hypercube grid point.
    fstar_histories : ndarray of shape (n_m0, n_sfh_grid, n_per_m0, n_t)
        Fstar history at each latin hypercube grid point.

    Returns
    -------
    mean_sm : ndarray of shape (n_m0, n_t)
        Average log10 Stellar Mass.
    variance_sm : ndarray of shape (n_m0, n_t)
        Variance of log10 Stellar Mass.
    mean_fstar_MS : ndarray of shape (n_m0, n_t)
        Average fstar (average SFH within some timescale) for main sequence galaxies.
    mean_fstar_Q : ndarray of shape (n_m0, n_t)
        Average fstar (average SFH within some timescale) for quenched galaxies.
    variance_fstar_MS : ndarray of shape (n_m0, n_t)
        Variance of fstar (average SFH within some timescale) for MS galaxies.
    variance_fstar_Q : ndarray of shape (n_m0, n_t)
        Variance of fstar (average SFH within some timescale) for Q galaxies.
    quench_frac : ndarray of shape (n_m0, n_t)
        Fraction of quenched galaxies.

    Notes
    -------
    Case where each halo has a different PDF weight.

    This is necessary if we do not approximate all halos in one bin to have
    the same M0

    It is also necessary if the PDF depends on the p50 of halos.
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

    w_sum_MS = jnp.einsum("abc,abcd->acd", weights_pdf, weights_quench_bin)
    w_sum_Q = jnp.einsum("abc,abcd->acd", weights_pdf, 1.0 - weights_quench_bin)

    NHALO_MS = jnp.sum(w_sum_MS, axis=1)
    NHALO_Q = jnp.sum(w_sum_Q, axis=1)

    quench_frac = NHALO_Q / (NHALO_Q + NHALO_MS)

    w_sum_MS = jnp.where(w_sum_MS > 0.0, 1.0 / w_sum_MS, 1.0)
    w_sum_Q = jnp.where(w_sum_Q > 0.0, 1.0 / w_sum_Q, 1.0)

    mean_sm = jnp.einsum("abc,ac,abcd->ad", weights_pdf, w_sum, mstar_histories) / NHALO
    # mean_sfr = jnp.einsum("abc,abcd->ad", weights_pdf, sfr_histories)
    # mean_fstar = jnp.einsum("abc,ac,abcd->ad", weights_pdf, w_sum, fstar_histories) / NHALO
    mean_fstar_MS = (
        jnp.einsum("abc,acd,abcd->ad", weights_pdf, w_sum_MS, fstar_MS) / NHALO
    )
    mean_fstar_Q = jnp.einsum("abc,acd,abcd->ad", weights_pdf, w_sum_Q, fstar_Q) / NHALO

    delta_sm = (mstar_histories - mean_sm[:, None, None, :]) ** 2
    # delta_sfr = (sfr_histories - mean_sfr) ** 2
    # delta_fstar = (fstar_histories - mean_fstar[:, None, None, :]) ** 2
    delta_fstar_MS = (fstar_MS - mean_fstar_MS[:, None, None, :]) ** 2
    delta_fstar_Q = (fstar_Q - mean_fstar_Q[:, None, None, :]) ** 2

    # delta_fstar_MS = jnp.where(fstar_MS == 0.0, 0.0, delta_fstar_MS)
    # delta_fstar_Q = jnp.where(fstar_Q == 0.0, 0.0, delta_fstar_Q)

    delta_fstar_MS = delta_fstar_MS * weights_quench_bin
    delta_fstar_Q = delta_fstar_Q * (1.0 - weights_quench_bin)

    variance_sm = jnp.einsum("abc,ac,abcd->ad", weights_pdf, w_sum, delta_sm) / NHALO
    # variance_sfr = jnp.sum(delta_sfr * w, axis=(0, 1, 2, 3, 4))
    # variance_fstar = jnp.einsum("abc,ac,abcd->ad", weights_pdf, w_sum, delta_fstar) / NHALO
    variance_fstar_MS = (
        jnp.einsum("abc,acd,abcd->ad", weights_pdf, w_sum_MS, delta_fstar_MS) / NHALO
    )
    variance_fstar_Q = (
        jnp.einsum("abc,acd,abcd->ad", weights_pdf, w_sum_Q, delta_fstar_Q) / NHALO
    )

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


def get_binned_means_and_covs_Q(pdf_model_pars_dict, logm0_bins):
    """Calculate the mean and covariance in SFH parameter space for each input M0.

    Parameters
    ----------
    pdf_model_pars_dict : dict
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

    pdf_model_params_dict = DEFAULT_SFH_PDF_QUENCH_PARAMS.copy()
    pdf_model_params_dict.update(pdf_model_pars_dict)

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


def get_default_pdf_SFH_prediction_Q_diffstar_vmap_Xsfh_scan_Xmah_scan(
    t_table,
    sfh_lh_sig,
    n_sfh_param_grid,
    logm0_binmids,
    logm0_bin_widths,
    n_halos_per_bin,
    halo_data,
    fstar_tdelay,
    pdf_model_params=DEFAULT_SFH_PDF_QUENCH_PARAMS,
):
    """
    Compute Diffstarpop summary statistic predictions for a population of
    quenched galaxies.

    Parameters
    ----------
    t_table : ndarray of shape (n_t, )
        Cosmic time array in Gyr.
    sfh_lh_sig : float
        Number of sigma used to define the latin hypercube box length.
    n_sfh_param_grid : int
        Number of sample points in latin hypercube box.
    logm0_binmids : ndarray of shape (n_m0, )
        Midpoint of the logarithmic halo mass bins
    logm0_bin_widths : ndarray of shape (n_m0, )
        Logarithmic width of the halo mass bin
    n_halos_per_bin : int
        Number of halos to be randomly sub-selected in each halo mass bin.
    halo_data : ndarray of shape (4, n_halos)
        Array containing the following Diffmah parameters
        (logm0, tauc, early, late):
            logmp : float
                Base-10 log of present-day peak halo mass in units of Msun assuming h=1
            tauc : float or ndarray of shape (n_halos, )
                Transition time between the fast- and slow-accretion regimes in Gyr
            early : float or ndarray of shape (n_halos, )
                Early-time power-law index in the scaling relation M(t)~t^a
            late : float or ndarray of shape (n_halos, )
                Late-time power-law index in the scaling relation M(t)~t^a
    fstar_tdelay: float
        Time interval in Gyr for fstar definition.
        fstar = (mstar(t) - mstar(t-fstar_tdelay)) / fstar_tdelay[Gyr]
    pdf_model_params : dict
        Dictionary containing the Diffstarpop parameters for the quenched population.
        Default is DEFAULT_SFH_PDF_QUENCH_PARAMS.

    Returns
    -------
    mean_sm : ndarray of shape (n_m0, n_t)
        Average log10 Stellar Mass.
    variance_sm : ndarray of shape (n_m0, n_t)
        Variance of log10 Stellar Mass.
    mean_fstar_MS : ndarray of shape (n_m0, n_t)
        Average fstar (average SFH within some timescale) for main sequence galaxies.
    mean_fstar_Q : ndarray of shape (n_m0, n_t)
        Average fstar (average SFH within some timescale) for quenched galaxies.
    variance_fstar_MS : ndarray of shape (n_m0, n_t)
        Variance of fstar (average SFH within some timescale) for MS galaxies.
    variance_fstar_Q : ndarray of shape (n_m0, n_t)
        Variance of fstar (average SFH within some timescale) for Q galaxies.
    quench_frac : ndarray of shape (n_m0, n_t)
        Fraction of quenched galaxies.

    Notes
    -----
    PDF is a unimodal Gaussian distribution on unbound Diffstar parameters.

    Details of kernel implementation:
        Diffstar tobs and tcons loops: vmap
        Diffstarpop Xmah loop: scan
        Diffstarpop Xsfh loop: scan
    """

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
    _res = compute_histories_on_grids_diffstar_vmap_Xsfh_scan_Xmah_scan(
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


def get_default_pdf_SFH_prediction_Q_diffstar_vmap_Xsfh_vmap_Xmah_vmap(
    t_table,
    sfh_lh_sig,
    n_sfh_param_grid,
    logm0_binmids,
    logm0_bin_widths,
    n_halos_per_bin,
    halo_data,
    fstar_tdelay,
    pdf_model_params=DEFAULT_SFH_PDF_QUENCH_PARAMS,
):
    """
    Compute Diffstarpop summary statistic predictions for a population of
    quenched galaxies.

    Parameters
    ----------
    t_table : ndarray of shape (n_t, )
        Cosmic time array in Gyr.
    sfh_lh_sig : float
        Number of sigma used to define the latin hypercube box length.
    n_sfh_param_grid : int
        Number of sample points in latin hypercube box.
    logm0_binmids : ndarray of shape (n_m0, )
        Midpoint of the logarithmic halo mass bins
    logm0_bin_widths : ndarray of shape (n_m0, )
        Logarithmic width of the halo mass bin
    n_halos_per_bin : int
        Number of halos to be randomly sub-selected in each halo mass bin.
    halo_data : ndarray of shape (4, n_halos)
        Array containing the following Diffmah parameters
        (logm0, tauc, early, late):
            logmp : float
                Base-10 log of present-day peak halo mass in units of Msun assuming h=1
            tauc : float or ndarray of shape (n_halos, )
                Transition time between the fast- and slow-accretion regimes in Gyr
            early : float or ndarray of shape (n_halos, )
                Early-time power-law index in the scaling relation M(t)~t^a
            late : float or ndarray of shape (n_halos, )
                Late-time power-law index in the scaling relation M(t)~t^a
    fstar_tdelay: float
        Time interval in Gyr for fstar definition.
        fstar = (mstar(t) - mstar(t-fstar_tdelay)) / fstar_tdelay[Gyr]
    pdf_model_params : dict
        Dictionary containing the Diffstarpop parameters for the quenched population.
        Default is DEFAULT_SFH_PDF_QUENCH_PARAMS.

    Returns
    -------
    mean_sm : ndarray of shape (n_m0, n_t)
        Average log10 Stellar Mass.
    variance_sm : ndarray of shape (n_m0, n_t)
        Variance of log10 Stellar Mass.
    mean_fstar_MS : ndarray of shape (n_m0, n_t)
        Average fstar (average SFH within some timescale) for main sequence galaxies.
    mean_fstar_Q : ndarray of shape (n_m0, n_t)
        Average fstar (average SFH within some timescale) for quenched galaxies.
    variance_fstar_MS : ndarray of shape (n_m0, n_t)
        Variance of fstar (average SFH within some timescale) for MS galaxies.
    variance_fstar_Q : ndarray of shape (n_m0, n_t)
        Variance of fstar (average SFH within some timescale) for Q galaxies.
    quench_frac : ndarray of shape (n_m0, n_t)
        Fraction of quenched galaxies.

    Notes
    -----
    PDF is a unimodal Gaussian distribution on unbound Diffstar parameters.

    Details of kernel implementation:
        Diffstar tobs and tcons loops: vmap
        Diffstarpop Xmah loop: vmap
        Diffstarpop Xsfh loop: vmap
    """

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
    _res = compute_histories_on_grids_diffstar_vmap_Xsfh_vmap_Xmah_vmap(
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


def get_binned_means_and_covs_MS(pdf_model_pars_dict, logm0_bins):
    """Calculate the mean and covariance in SFH parameter space for each input M0.

    Parameters
    ----------
    pdf_model_pars_dict : ndarray of shape (n_sfh_params, )
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

    pdf_model_params_dict = DEFAULT_SFH_PDF_MAINSEQ_PARAMS.copy()
    pdf_model_params_dict.update(pdf_model_pars_dict)

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


def get_default_pdf_SFH_prediction_MS_diffstar_vmap_Xsfh_scan_Xmah_scan(
    t_table,
    sfh_lh_sig,
    n_sfh_param_grid,
    logm0_binmids,
    logm0_bin_widths,
    n_halos_per_bin,
    halo_data,
    fstar_tdelay,
    pdf_model_params=DEFAULT_SFH_PDF_MAINSEQ_PARAMS,
):
    """
    Compute Diffstarpop summary statistic predictions for a population of
    main sequence galaxies.

    Parameters
    ----------
    t_table : ndarray of shape (n_t, )
        Cosmic time array in Gyr.
    sfh_lh_sig : float
        Number of sigma used to define the latin hypercube box length.
    n_sfh_param_grid : int
        Number of sample points in latin hypercube box.
    logm0_binmids : ndarray of shape (n_m0, )
        Midpoint of the logarithmic halo mass bins
    logm0_bin_widths : ndarray of shape (n_m0, )
        Logarithmic width of the halo mass bin
    n_halos_per_bin : int
        Number of halos to be randomly sub-selected in each halo mass bin.
    halo_data : ndarray of shape (4, n_halos)
        Array containing the following Diffmah parameters
        (logm0, tauc, early, late):
            logmp : float
                Base-10 log of present-day peak halo mass in units of Msun assuming h=1
            tauc : float or ndarray of shape (n_halos, )
                Transition time between the fast- and slow-accretion regimes in Gyr
            early : float or ndarray of shape (n_halos, )
                Early-time power-law index in the scaling relation M(t)~t^a
            late : float or ndarray of shape (n_halos, )
                Late-time power-law index in the scaling relation M(t)~t^a
    fstar_tdelay: float
        Time interval in Gyr for fstar definition.
        fstar = (mstar(t) - mstar(t-fstar_tdelay)) / fstar_tdelay[Gyr]
    pdf_model_params : dict
        Dictionary containing the Diffstarpop parameters for the main sequence population.
        Default is DEFAULT_SFH_PDF_MAINSEQ_PARAMS.

    Returns
    -------
    mean_sm : ndarray of shape (n_m0, n_t)
        Average log10 Stellar Mass.
    variance_sm : ndarray of shape (n_m0, n_t)
        Variance of log10 Stellar Mass.
    mean_fstar_MS : ndarray of shape (n_m0, n_t)
        Average fstar (average SFH within some timescale) for main sequence galaxies.
    mean_fstar_Q : ndarray of shape (n_m0, n_t)
        Average fstar (average SFH within some timescale) for quenched galaxies.
    variance_fstar_MS : ndarray of shape (n_m0, n_t)
        Variance of fstar (average SFH within some timescale) for MS galaxies.
    variance_fstar_Q : ndarray of shape (n_m0, n_t)
        Variance of fstar (average SFH within some timescale) for Q galaxies.
    quench_frac : ndarray of shape (n_m0, n_t)
        Fraction of quenched galaxies.

    Notes
    -----
    PDF is a unimodal Gaussian distribution on the four unbound main sequence
    Diffstar parameters.

    Details of kernel implementation:
        Diffstar tobs and tcons loops: vmap
        Diffstarpop Xmah loop: scan
        Diffstarpop Xsfh loop: scan
    """
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
    _res = compute_histories_on_grids_MS_diffstar_vmap_Xsfh_scan_Xmah_scan(
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


def get_default_pdf_SFH_prediction_MS_diffstar_vmap_Xsfh_vmap_Xmah_vmap(
    t_table,
    sfh_lh_sig,
    n_sfh_param_grid,
    logm0_binmids,
    logm0_bin_widths,
    n_halos_per_bin,
    halo_data,
    fstar_tdelay,
    pdf_model_params=DEFAULT_SFH_PDF_MAINSEQ_PARAMS,
):
    """
    Compute Diffstarpop summary statistic predictions for a population of
    main sequence galaxies.

    Parameters
    ----------
    t_table : ndarray of shape (n_t, )
        Cosmic time array in Gyr.
    sfh_lh_sig : float
        Number of sigma used to define the latin hypercube box length.
    n_sfh_param_grid : int
        Number of sample points in latin hypercube box.
    logm0_binmids : ndarray of shape (n_m0, )
        Midpoint of the logarithmic halo mass bins
    logm0_bin_widths : ndarray of shape (n_m0, )
        Logarithmic width of the halo mass bin
    n_halos_per_bin : int
        Number of halos to be randomly sub-selected in each halo mass bin.
    halo_data : ndarray of shape (4, n_halos)
        Array containing the following Diffmah parameters
        (logm0, tauc, early, late):
            logmp : float
                Base-10 log of present-day peak halo mass in units of Msun assuming h=1
            tauc : float or ndarray of shape (n_halos, )
                Transition time between the fast- and slow-accretion regimes in Gyr
            early : float or ndarray of shape (n_halos, )
                Early-time power-law index in the scaling relation M(t)~t^a
            late : float or ndarray of shape (n_halos, )
                Late-time power-law index in the scaling relation M(t)~t^a
    fstar_tdelay: float
        Time interval in Gyr for fstar definition.
        fstar = (mstar(t) - mstar(t-fstar_tdelay)) / fstar_tdelay[Gyr]
    pdf_model_params : dict
        Dictionary containing the Diffstarpop parameters for the main sequence population.
        Default is DEFAULT_SFH_PDF_MAINSEQ_PARAMS.

    Returns
    -------
    mean_sm : ndarray of shape (n_m0, n_t)
        Average log10 Stellar Mass.
    variance_sm : ndarray of shape (n_m0, n_t)
        Variance of log10 Stellar Mass.
    mean_fstar_MS : ndarray of shape (n_m0, n_t)
        Average fstar (average SFH within some timescale) for main sequence galaxies.
    mean_fstar_Q : ndarray of shape (n_m0, n_t)
        Average fstar (average SFH within some timescale) for quenched galaxies.
    variance_fstar_MS : ndarray of shape (n_m0, n_t)
        Variance of fstar (average SFH within some timescale) for MS galaxies.
    variance_fstar_Q : ndarray of shape (n_m0, n_t)
        Variance of fstar (average SFH within some timescale) for Q galaxies.
    quench_frac : ndarray of shape (n_m0, n_t)
        Fraction of quenched galaxies.

    Notes
    -----
    PDF is a unimodal Gaussian distribution on the four unbound main sequence
    Diffstar parameters.

    Details of kernel implementation:
        Diffstar tobs and tcons loops: vmap
        Diffstarpop Xmah loop: vmap
        Diffstarpop Xsfh loop: vmap
    """
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
    _res = compute_histories_on_grids_MS_diffstar_vmap_Xsfh_vmap_Xmah_vmap(
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


def get_default_pdf_SFH_prediction_MIX_diffstar_vmap_Xsfh_scan_Xmah_scan(
    t_table,
    sfh_lh_sig,
    n_sfh_param_grid,
    logm0_binmids,
    logm0_bin_widths,
    n_halos_per_bin,
    halo_data,
    fstar_tdelay,
    pdf_model_params_MS=DEFAULT_SFH_PDF_MAINSEQ_PARAMS,
    pdf_model_params_Q=DEFAULT_SFH_PDF_QUENCH_PARAMS,
):
    """
    Compute Diffstarpop summary statistic predictions for a mixed population of
    both main sequence and quenched galaxies.

    Parameters
    ----------
    t_table : ndarray of shape (n_t, )
        Cosmic time array in Gyr.
    sfh_lh_sig : float
        Number of sigma used to define the latin hypercube box length.
    n_sfh_param_grid : int
        Number of sample points in latin hypercube box.
    logm0_binmids : ndarray of shape (n_m0, )
        Midpoint of the logarithmic halo mass bins
    logm0_bin_widths : ndarray of shape (n_m0, )
        Logarithmic width of the halo mass bin
    n_halos_per_bin : int
        Number of halos to be randomly sub-selected in each halo mass bin.
    halo_data : ndarray of shape (4, n_halos)
        Array containing the following Diffmah parameters
        (logm0, tauc, early, late):
            logmp : float
                Base-10 log of present-day peak halo mass in units of Msun assuming h=1
            tauc : float or ndarray of shape (n_halos, )
                Transition time between the fast- and slow-accretion regimes in Gyr
            early : float or ndarray of shape (n_halos, )
                Early-time power-law index in the scaling relation M(t)~t^a
            late : float or ndarray of shape (n_halos, )
                Late-time power-law index in the scaling relation M(t)~t^a
    fstar_tdelay: float
        Time interval in Gyr for fstar definition.
        fstar = (mstar(t) - mstar(t-fstar_tdelay)) / fstar_tdelay[Gyr]
    pdf_model_params_MS : dict
        Dictionary containing the Diffstarpop parameters for the main sequence population.
        Default is DEFAULT_SFH_PDF_MAINSEQ_PARAMS.
    pdf_model_params_Q : dict
        Dictionary containing the Diffstarpop parameters for the quenched population.
        Default is DEFAULT_SFH_PDF_QUENCH_PARAMS.

    Returns
    -------
    mean_sm : ndarray of shape (n_m0, n_t)
        Average log10 Stellar Mass.
    variance_sm : ndarray of shape (n_m0, n_t)
        Variance of log10 Stellar Mass.
    mean_fstar_MS : ndarray of shape (n_m0, n_t)
        Average fstar (average SFH within some timescale) for main sequence galaxies.
    mean_fstar_Q : ndarray of shape (n_m0, n_t)
        Average fstar (average SFH within some timescale) for quenched galaxies.
    variance_fstar_MS : ndarray of shape (n_m0, n_t)
        Variance of fstar (average SFH within some timescale) for MS galaxies.
    variance_fstar_Q : ndarray of shape (n_m0, n_t)
        Variance of fstar (average SFH within some timescale) for Q galaxies.
    quench_frac : ndarray of shape (n_m0, n_t)
        Fraction of quenched galaxies.

    Notes
    -----
    PDF is a mixture Gaussian model distribution:
        PDF = f_Q * PDF_Q + (1-f_Q) * PDF_MS
    where:
        f_Q: quenched Fraction
        PDF_Q: Unimodal Gaussian for the quenched sub-population
        PDF_MS: Unimodal Gaussian for the main sequence sub-population

    Details of kernel implementation:
        Diffstar tobs and tcons loops: vmap
        Diffstarpop Xmah loop: scan
        Diffstarpop Xsfh loop: scan
    """

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
    _res = compute_histories_on_grids_MS_diffstar_vmap_Xsfh_scan_Xmah_scan(
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
    _res = compute_histories_on_grids_diffstar_vmap_Xsfh_scan_Xmah_scan(
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

    weights_MS /= weights_MS.sum()
    weights_Q /= weights_Q.sum()

    weights_MS = jnp.einsum("ij,i->ij", weights_MS, (1.0 - fracs_Q))
    weights_Q = jnp.einsum("ij,i->ij", weights_Q, fracs_Q)

    weights = jnp.concatenate((weights_MS, weights_Q), axis=1)
    weights_bin = jnp.concatenate((weights_MS_bin, weights_Q_bin), axis=1)

    return compute_target_sumstats_from_histories(
        weights, weights_bin, mstar_histories, sfr_histories, fstar_histories
    )


def get_default_pdf_SFH_prediction_MIX_diffstar_vmap_Xsfh_vmap_Xmah_vmap(
    t_table,
    sfh_lh_sig,
    n_sfh_param_grid,
    logm0_binmids,
    logm0_bin_widths,
    n_halos_per_bin,
    halo_data,
    fstar_tdelay,
    pdf_model_params_MS=DEFAULT_SFH_PDF_MAINSEQ_PARAMS,
    pdf_model_params_Q=DEFAULT_SFH_PDF_QUENCH_PARAMS,
):
    """
    Compute Diffstarpop summary statistic predictions for a mixed population of
    both main sequence and quenched galaxies.

    Parameters
    ----------
    t_table : ndarray of shape (n_t, )
        Cosmic time array in Gyr.
    sfh_lh_sig : float
        Number of sigma used to define the latin hypercube box length.
    n_sfh_param_grid : int
        Number of sample points in latin hypercube box.
    logm0_binmids : ndarray of shape (n_m0, )
        Midpoint of the logarithmic halo mass bins
    logm0_bin_widths : ndarray of shape (n_m0, )
        Logarithmic width of the halo mass bin
    n_halos_per_bin : int
        Number of halos to be randomly sub-selected in each halo mass bin.
    halo_data : ndarray of shape (4, n_halos)
        Array containing the following Diffmah parameters
        (logm0, tauc, early, late):
            logmp : float
                Base-10 log of present-day peak halo mass in units of Msun assuming h=1
            tauc : float or ndarray of shape (n_halos, )
                Transition time between the fast- and slow-accretion regimes in Gyr
            early : float or ndarray of shape (n_halos, )
                Early-time power-law index in the scaling relation M(t)~t^a
            late : float or ndarray of shape (n_halos, )
                Late-time power-law index in the scaling relation M(t)~t^a
    fstar_tdelay: float
        Time interval in Gyr for fstar definition.
        fstar = (mstar(t) - mstar(t-fstar_tdelay)) / fstar_tdelay[Gyr]
    pdf_model_params_MS : dict
        Dictionary containing the Diffstarpop parameters for the main sequence population.
        Default is DEFAULT_SFH_PDF_MAINSEQ_PARAMS.
    pdf_model_params_Q : dict
        Dictionary containing the Diffstarpop parameters for the quenched population.
        Default is DEFAULT_SFH_PDF_QUENCH_PARAMS.

    Returns
    -------
    mean_sm : ndarray of shape (n_m0, n_t)
        Average log10 Stellar Mass.
    variance_sm : ndarray of shape (n_m0, n_t)
        Variance of log10 Stellar Mass.
    mean_fstar_MS : ndarray of shape (n_m0, n_t)
        Average fstar (average SFH within some timescale) for main sequence galaxies.
    mean_fstar_Q : ndarray of shape (n_m0, n_t)
        Average fstar (average SFH within some timescale) for quenched galaxies.
    variance_fstar_MS : ndarray of shape (n_m0, n_t)
        Variance of fstar (average SFH within some timescale) for MS galaxies.
    variance_fstar_Q : ndarray of shape (n_m0, n_t)
        Variance of fstar (average SFH within some timescale) for Q galaxies.
    quench_frac : ndarray of shape (n_m0, n_t)
        Fraction of quenched galaxies.

    Notes
    -----
    PDF is a mixture Gaussian model distribution:
        PDF = f_Q * PDF_Q + (1-f_Q) * PDF_MS
    where:
        f_Q: quenched Fraction
        PDF_Q: Unimodal Gaussian for the quenched sub-population
        PDF_MS: Unimodal Gaussian for the main sequence sub-population

    Details of kernel implementation:
        Diffstar tobs and tcons loops: vmap
        Diffstarpop Xmah loop: vmap
        Diffstarpop Xsfh loop: vmap
    """

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
    _res = compute_histories_on_grids_MS_diffstar_vmap_Xsfh_vmap_Xmah_vmap(
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
    _res = compute_histories_on_grids_diffstar_vmap_Xsfh_vmap_Xmah_vmap(
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

    weights_MS /= weights_MS.sum()
    weights_Q /= weights_Q.sum()

    weights_MS = jnp.einsum("ij,i->ij", weights_MS, (1.0 - fracs_Q))
    weights_Q = jnp.einsum("ij,i->ij", weights_Q, fracs_Q)

    weights = jnp.concatenate((weights_MS, weights_Q), axis=1)
    weights_bin = jnp.concatenate((weights_MS_bin, weights_Q_bin), axis=1)

    return compute_target_sumstats_from_histories(
        weights, weights_bin, mstar_histories, sfr_histories, fstar_histories
    )


def get_default_pdf_SFH_prediction_Q_diffstar_scan_Xsfh_scan_Xmah_scan(
    t_table,
    sfh_lh_sig,
    n_sfh_param_grid,
    logm0_binmids,
    logm0_bin_widths,
    n_halos_per_bin,
    halo_data,
    fstar_tdelay,
    pdf_model_params=DEFAULT_SFH_PDF_QUENCH_PARAMS,
):
    """
    Compute Diffstarpop summary statistic predictions for a population of
    quenched galaxies.

    Parameters
    ----------
    t_table : ndarray of shape (n_t, )
        Cosmic time array in Gyr.
    sfh_lh_sig : float
        Number of sigma used to define the latin hypercube box length.
    n_sfh_param_grid : int
        Number of sample points in latin hypercube box.
    logm0_binmids : ndarray of shape (n_m0, )
        Midpoint of the logarithmic halo mass bins
    logm0_bin_widths : ndarray of shape (n_m0, )
        Logarithmic width of the halo mass bin
    n_halos_per_bin : int
        Number of halos to be randomly sub-selected in each halo mass bin.
    halo_data : ndarray of shape (4, n_halos)
        Array containing the following Diffmah parameters
        (logm0, tauc, early, late):
            logmp : float
                Base-10 log of present-day peak halo mass in units of Msun assuming h=1
            tauc : float or ndarray of shape (n_halos, )
                Transition time between the fast- and slow-accretion regimes in Gyr
            early : float or ndarray of shape (n_halos, )
                Early-time power-law index in the scaling relation M(t)~t^a
            late : float or ndarray of shape (n_halos, )
                Late-time power-law index in the scaling relation M(t)~t^a
    fstar_tdelay: float
        Time interval in Gyr for fstar definition.
        fstar = (mstar(t) - mstar(t-fstar_tdelay)) / fstar_tdelay[Gyr]
    pdf_model_params : dict
        Dictionary containing the Diffstarpop parameters for the quenched population.
        Default is DEFAULT_SFH_PDF_QUENCH_PARAMS.

    Returns
    -------
    mean_sm : ndarray of shape (n_m0, n_t)
        Average log10 Stellar Mass.
    variance_sm : ndarray of shape (n_m0, n_t)
        Variance of log10 Stellar Mass.
    mean_fstar_MS : ndarray of shape (n_m0, n_t)
        Average fstar (average SFH within some timescale) for main sequence galaxies.
    mean_fstar_Q : ndarray of shape (n_m0, n_t)
        Average fstar (average SFH within some timescale) for quenched galaxies.
    variance_fstar_MS : ndarray of shape (n_m0, n_t)
        Variance of fstar (average SFH within some timescale) for MS galaxies.
    variance_fstar_Q : ndarray of shape (n_m0, n_t)
        Variance of fstar (average SFH within some timescale) for Q galaxies.
    quench_frac : ndarray of shape (n_m0, n_t)
        Fraction of quenched galaxies.

    Notes
    -----
    PDF is a unimodal Gaussian distribution on unbound Diffstar parameters.

    Details of kernel implementation:
        Diffstar tobs and tcons loops: scan
        Diffstarpop Xmah loop: scan
        Diffstarpop Xsfh loop: scan
    """

    index_high = np.searchsorted(t_table, t_table - fstar_tdelay)
    _mask = t_table > fstar_tdelay + fstar_tdelay / 2.0
    index_select = np.arange(len(t_table))[_mask]
    index_high = index_high[_mask]

    lgt_table = jnp.log10(t_table)
    dt_table = _jax_get_dt_array(t_table)

    jran_key = jran.PRNGKey(0)
    logm0_halos, mah_tauc_halos, mah_early_halos, mah_late_halos = halo_data

    # the diffstar scan functions actually want log(tauc)
    mah_logtauc_halos = np.log10(mah_tauc_halos)

    diffmah_params_grid = get_binned_halo_sample(
        n_halos_per_bin,
        jran_key,
        logm0_binmids,
        logm0_bin_widths,
        logm0_halos,
        mah_logtauc_halos,
        mah_early_halos,
        mah_late_halos,
    )
    diffmah_params_grid = np.array(diffmah_params_grid)
    diffmah_params_grid = diffmah_params_grid.reshape(
        (4, len(logm0_binmids), n_halos_per_bin)
    )
    diffmah_params_grid = np.einsum("pmh->mhp", diffmah_params_grid)

    fracs_logm0, sfh_param_grids = get_sfh_param_grid_Q(
        pdf_model_params, logm0_binmids, sfh_lh_sig, n_sfh_param_grid
    )

    fracs, means, covs = get_binned_means_and_covs_Q(pdf_model_params, logm0_binmids)
    _res = compute_histories_on_grids_diffstar_scan_Xsfh_scan_Xmah_scan(
        t_table,
        lgt_table,
        dt_table,
        index_select,
        index_high,
        fstar_tdelay,
        diffmah_params_grid,
        sfh_param_grids,
    )
    mstar_histories, sfr_histories, fstar_histories = _res

    sfstar_histories = fstar_histories / mstar_histories[:, :, :, index_select]

    weights_quench_bin = jnp.where(sfstar_histories > 1e-11, 1.0, 0.0)

    weights = _get_pdf_weights_kern(sfh_param_grids, means, covs)

    return compute_target_sumstats_from_histories(
        weights, weights_quench_bin, mstar_histories, sfr_histories, fstar_histories
    )


def get_default_pdf_SFH_prediction_Q_diffstar_scan_Xsfh_vmap_Xmah_vmap(
    t_table,
    sfh_lh_sig,
    n_sfh_param_grid,
    logm0_binmids,
    logm0_bin_widths,
    n_halos_per_bin,
    halo_data,
    fstar_tdelay,
    pdf_model_params=DEFAULT_SFH_PDF_QUENCH_PARAMS,
):
    """
    Compute Diffstarpop summary statistic predictions for a population of
    quenched galaxies.

    Parameters
    ----------
    t_table : ndarray of shape (n_t, )
        Cosmic time array in Gyr.
    sfh_lh_sig : float
        Number of sigma used to define the latin hypercube box length.
    n_sfh_param_grid : int
        Number of sample points in latin hypercube box.
    logm0_binmids : ndarray of shape (n_m0, )
        Midpoint of the logarithmic halo mass bins
    logm0_bin_widths : ndarray of shape (n_m0, )
        Logarithmic width of the halo mass bin
    n_halos_per_bin : int
        Number of halos to be randomly sub-selected in each halo mass bin.
    halo_data : ndarray of shape (4, n_halos)
        Array containing the following Diffmah parameters
        (logm0, tauc, early, late):
            logmp : float
                Base-10 log of present-day peak halo mass in units of Msun assuming h=1
            tauc : float or ndarray of shape (n_halos, )
                Transition time between the fast- and slow-accretion regimes in Gyr
            early : float or ndarray of shape (n_halos, )
                Early-time power-law index in the scaling relation M(t)~t^a
            late : float or ndarray of shape (n_halos, )
                Late-time power-law index in the scaling relation M(t)~t^a
    fstar_tdelay: float
        Time interval in Gyr for fstar definition.
        fstar = (mstar(t) - mstar(t-fstar_tdelay)) / fstar_tdelay[Gyr]
    pdf_model_params : dict
        Dictionary containing the Diffstarpop parameters for the quenched population.
        Default is DEFAULT_SFH_PDF_QUENCH_PARAMS.

    Returns
    -------
    mean_sm : ndarray of shape (n_m0, n_t)
        Average log10 Stellar Mass.
    variance_sm : ndarray of shape (n_m0, n_t)
        Variance of log10 Stellar Mass.
    mean_fstar_MS : ndarray of shape (n_m0, n_t)
        Average fstar (average SFH within some timescale) for main sequence galaxies.
    mean_fstar_Q : ndarray of shape (n_m0, n_t)
        Average fstar (average SFH within some timescale) for quenched galaxies.
    variance_fstar_MS : ndarray of shape (n_m0, n_t)
        Variance of fstar (average SFH within some timescale) for MS galaxies.
    variance_fstar_Q : ndarray of shape (n_m0, n_t)
        Variance of fstar (average SFH within some timescale) for Q galaxies.
    quench_frac : ndarray of shape (n_m0, n_t)
        Fraction of quenched galaxies.

    Notes
    -----
    PDF is a unimodal Gaussian distribution on unbound Diffstar parameters.

    Details of kernel implementation:
        Diffstar tobs and tcons loops: scan
        Diffstarpop Xmah loop: vmap
        Diffstarpop Xsfh loop: vmap
    """

    index_high = np.searchsorted(t_table, t_table - fstar_tdelay)
    _mask = t_table > fstar_tdelay + fstar_tdelay / 2.0
    index_select = np.arange(len(t_table))[_mask]
    index_high = index_high[_mask]

    lgt_table = jnp.log10(t_table)
    dt_table = _jax_get_dt_array(t_table)

    jran_key = jran.PRNGKey(0)
    logm0_halos, mah_tauc_halos, mah_early_halos, mah_late_halos = halo_data

    # the diffstar scan functions actually want log(tauc)
    mah_logtauc_halos = np.log10(mah_tauc_halos)

    diffmah_params_grid = get_binned_halo_sample(
        n_halos_per_bin,
        jran_key,
        logm0_binmids,
        logm0_bin_widths,
        logm0_halos,
        mah_logtauc_halos,
        mah_early_halos,
        mah_late_halos,
    )
    diffmah_params_grid = np.array(diffmah_params_grid)
    diffmah_params_grid = diffmah_params_grid.reshape(
        (4, len(logm0_binmids), n_halos_per_bin)
    )
    diffmah_params_grid = np.einsum("pmh->mhp", diffmah_params_grid)

    fracs_logm0, sfh_param_grids = get_sfh_param_grid_Q(
        pdf_model_params, logm0_binmids, sfh_lh_sig, n_sfh_param_grid
    )

    fracs, means, covs = get_binned_means_and_covs_Q(pdf_model_params, logm0_binmids)
    _res = compute_histories_on_grids_diffstar_scan_Xsfh_vmap_Xmah_vmap(
        t_table,
        lgt_table,
        dt_table,
        index_select,
        index_high,
        fstar_tdelay,
        diffmah_params_grid,
        sfh_param_grids,
    )
    mstar_histories, sfr_histories, fstar_histories = _res

    sfstar_histories = fstar_histories / mstar_histories[:, :, :, index_select]

    weights_quench_bin = jnp.where(sfstar_histories > 1e-11, 1.0, 0.0)

    weights = _get_pdf_weights_kern(sfh_param_grids, means, covs)

    return compute_target_sumstats_from_histories(
        weights, weights_quench_bin, mstar_histories, sfr_histories, fstar_histories
    )


def get_default_pdf_SFH_prediction_MS_diffstar_scan_Xsfh_scan_Xmah_scan(
    t_table,
    sfh_lh_sig,
    n_sfh_param_grid,
    logm0_binmids,
    logm0_bin_widths,
    n_halos_per_bin,
    halo_data,
    fstar_tdelay,
    pdf_model_params=DEFAULT_SFH_PDF_MAINSEQ_PARAMS,
):
    """
    Compute Diffstarpop summary statistic predictions for a population of
    main sequence galaxies.

    Parameters
    ----------
    t_table : ndarray of shape (n_t, )
        Cosmic time array in Gyr.
    sfh_lh_sig : float
        Number of sigma used to define the latin hypercube box length.
    n_sfh_param_grid : int
        Number of sample points in latin hypercube box.
    logm0_binmids : ndarray of shape (n_m0, )
        Midpoint of the logarithmic halo mass bins
    logm0_bin_widths : ndarray of shape (n_m0, )
        Logarithmic width of the halo mass bin
    n_halos_per_bin : int
        Number of halos to be randomly sub-selected in each halo mass bin.
    halo_data : ndarray of shape (4, n_halos)
        Array containing the following Diffmah parameters
        (logm0, tauc, early, late):
            logmp : float
                Base-10 log of present-day peak halo mass in units of Msun assuming h=1
            tauc : float or ndarray of shape (n_halos, )
                Transition time between the fast- and slow-accretion regimes in Gyr
            early : float or ndarray of shape (n_halos, )
                Early-time power-law index in the scaling relation M(t)~t^a
            late : float or ndarray of shape (n_halos, )
                Late-time power-law index in the scaling relation M(t)~t^a
    fstar_tdelay: float
        Time interval in Gyr for fstar definition.
        fstar = (mstar(t) - mstar(t-fstar_tdelay)) / fstar_tdelay[Gyr]
    pdf_model_params : dict
        Dictionary containing the Diffstarpop parameters for the quenched population.
        Default is DEFAULT_SFH_PDF_MAINSEQ_PARAMS.

    Returns
    -------
    mean_sm : ndarray of shape (n_m0, n_t)
        Average log10 Stellar Mass.
    variance_sm : ndarray of shape (n_m0, n_t)
        Variance of log10 Stellar Mass.
    mean_fstar_MS : ndarray of shape (n_m0, n_t)
        Average fstar (average SFH within some timescale) for main sequence galaxies.
    mean_fstar_Q : ndarray of shape (n_m0, n_t)
        Average fstar (average SFH within some timescale) for quenched galaxies.
    variance_fstar_MS : ndarray of shape (n_m0, n_t)
        Variance of fstar (average SFH within some timescale) for MS galaxies.
    variance_fstar_Q : ndarray of shape (n_m0, n_t)
        Variance of fstar (average SFH within some timescale) for Q galaxies.
    quench_frac : ndarray of shape (n_m0, n_t)
        Fraction of quenched galaxies.

    Notes
    -----
    PDF is a unimodal Gaussian distribution on unbound Diffstar parameters.

    Details of kernel implementation:
        Diffstar tobs and tcons loops: scan
        Diffstarpop Xmah loop: scan
        Diffstarpop Xsfh loop: scan
    """

    index_high = np.searchsorted(t_table, t_table - fstar_tdelay)
    _mask = t_table > fstar_tdelay + fstar_tdelay / 2.0
    index_select = np.arange(len(t_table))[_mask]
    index_high = index_high[_mask]

    lgt_table = jnp.log10(t_table)
    dt_table = _jax_get_dt_array(t_table)

    jran_key = jran.PRNGKey(0)
    logm0_halos, mah_tauc_halos, mah_early_halos, mah_late_halos = halo_data

    # the diffstar scan functions actually want log(tauc)
    mah_logtauc_halos = np.log10(mah_tauc_halos)

    diffmah_params_grid = get_binned_halo_sample(
        n_halos_per_bin,
        jran_key,
        logm0_binmids,
        logm0_bin_widths,
        logm0_halos,
        mah_logtauc_halos,
        mah_early_halos,
        mah_late_halos,
    )
    diffmah_params_grid = np.array(diffmah_params_grid)
    diffmah_params_grid = diffmah_params_grid.reshape(
        (4, len(logm0_binmids), n_halos_per_bin)
    )
    diffmah_params_grid = np.einsum("pmh->mhp", diffmah_params_grid)

    sfh_param_grids = get_sfh_param_grid_MS(
        pdf_model_params, logm0_binmids, sfh_lh_sig, n_sfh_param_grid
    )

    means, covs = get_binned_means_and_covs_MS(pdf_model_params, logm0_binmids)
    _res = compute_histories_on_grids_MS_diffstar_scan_Xsfh_scan_Xmah_scan(
        t_table,
        lgt_table,
        dt_table,
        index_select,
        index_high,
        fstar_tdelay,
        diffmah_params_grid,
        sfh_param_grids,
    )
    mstar_histories, sfr_histories, fstar_histories = _res

    sfstar_histories = fstar_histories / mstar_histories[:, :, :, index_select]

    weights_quench_bin = jnp.where(sfstar_histories > 1e-11, 1.0, 0.0)

    weights = _get_pdf_weights_kern(sfh_param_grids, means, covs)

    return compute_target_sumstats_from_histories(
        weights, weights_quench_bin, mstar_histories, sfr_histories, fstar_histories
    )


def get_default_pdf_SFH_prediction_MS_diffstar_scan_Xsfh_vmap_Xmah_vmap(
    t_table,
    sfh_lh_sig,
    n_sfh_param_grid,
    logm0_binmids,
    logm0_bin_widths,
    n_halos_per_bin,
    halo_data,
    fstar_tdelay,
    pdf_model_params=DEFAULT_SFH_PDF_MAINSEQ_PARAMS,
):
    """
    Compute Diffstarpop summary statistic predictions for a population of
    main sequence galaxies.

    Parameters
    ----------
    t_table : ndarray of shape (n_t, )
        Cosmic time array in Gyr.
    sfh_lh_sig : float
        Number of sigma used to define the latin hypercube box length.
    n_sfh_param_grid : int
        Number of sample points in latin hypercube box.
    logm0_binmids : ndarray of shape (n_m0, )
        Midpoint of the logarithmic halo mass bins
    logm0_bin_widths : ndarray of shape (n_m0, )
        Logarithmic width of the halo mass bin
    n_halos_per_bin : int
        Number of halos to be randomly sub-selected in each halo mass bin.
    halo_data : ndarray of shape (4, n_halos)
        Array containing the following Diffmah parameters
        (logm0, tauc, early, late):
            logmp : float
                Base-10 log of present-day peak halo mass in units of Msun assuming h=1
            tauc : float or ndarray of shape (n_halos, )
                Transition time between the fast- and slow-accretion regimes in Gyr
            early : float or ndarray of shape (n_halos, )
                Early-time power-law index in the scaling relation M(t)~t^a
            late : float or ndarray of shape (n_halos, )
                Late-time power-law index in the scaling relation M(t)~t^a
    fstar_tdelay: float
        Time interval in Gyr for fstar definition.
        fstar = (mstar(t) - mstar(t-fstar_tdelay)) / fstar_tdelay[Gyr]
    pdf_model_params : dict
        Dictionary containing the Diffstarpop parameters for the quenched population.
        Default is DEFAULT_SFH_PDF_MAINSEQ_PARAMS.

    Returns
    -------
    mean_sm : ndarray of shape (n_m0, n_t)
        Average log10 Stellar Mass.
    variance_sm : ndarray of shape (n_m0, n_t)
        Variance of log10 Stellar Mass.
    mean_fstar_MS : ndarray of shape (n_m0, n_t)
        Average fstar (average SFH within some timescale) for main sequence galaxies.
    mean_fstar_Q : ndarray of shape (n_m0, n_t)
        Average fstar (average SFH within some timescale) for quenched galaxies.
    variance_fstar_MS : ndarray of shape (n_m0, n_t)
        Variance of fstar (average SFH within some timescale) for MS galaxies.
    variance_fstar_Q : ndarray of shape (n_m0, n_t)
        Variance of fstar (average SFH within some timescale) for Q galaxies.
    quench_frac : ndarray of shape (n_m0, n_t)
        Fraction of quenched galaxies.

    Notes
    -----
    PDF is a unimodal Gaussian distribution on unbound Diffstar parameters.

    Details of kernel implementation:
        Diffstar tobs and tcons loops: scan
        Diffstarpop Xmah loop: vmap
        Diffstarpop Xsfh loop: vmap
    """

    index_high = np.searchsorted(t_table, t_table - fstar_tdelay)
    _mask = t_table > fstar_tdelay + fstar_tdelay / 2.0
    index_select = np.arange(len(t_table))[_mask]
    index_high = index_high[_mask]

    lgt_table = jnp.log10(t_table)
    dt_table = _jax_get_dt_array(t_table)

    jran_key = jran.PRNGKey(0)
    logm0_halos, mah_tauc_halos, mah_early_halos, mah_late_halos = halo_data

    # the diffstar scan functions actually want log(tauc)
    mah_logtauc_halos = np.log10(mah_tauc_halos)

    diffmah_params_grid = get_binned_halo_sample(
        n_halos_per_bin,
        jran_key,
        logm0_binmids,
        logm0_bin_widths,
        logm0_halos,
        mah_logtauc_halos,
        mah_early_halos,
        mah_late_halos,
    )
    diffmah_params_grid = np.array(diffmah_params_grid)
    diffmah_params_grid = diffmah_params_grid.reshape(
        (4, len(logm0_binmids), n_halos_per_bin)
    )
    diffmah_params_grid = np.einsum("pmh->mhp", diffmah_params_grid)

    sfh_param_grids = get_sfh_param_grid_MS(
        pdf_model_params, logm0_binmids, sfh_lh_sig, n_sfh_param_grid
    )

    means, covs = get_binned_means_and_covs_MS(pdf_model_params, logm0_binmids)
    _res = compute_histories_on_grids_MS_diffstar_scan_Xsfh_vmap_Xmah_vmap(
        t_table,
        lgt_table,
        dt_table,
        index_select,
        index_high,
        fstar_tdelay,
        diffmah_params_grid,
        sfh_param_grids,
    )
    mstar_histories, sfr_histories, fstar_histories = _res

    sfstar_histories = fstar_histories / mstar_histories[:, :, :, index_select]

    weights_quench_bin = jnp.where(sfstar_histories > 1e-11, 1.0, 0.0)

    weights = _get_pdf_weights_kern(sfh_param_grids, means, covs)

    return compute_target_sumstats_from_histories(
        weights, weights_quench_bin, mstar_histories, sfr_histories, fstar_histories
    )


def get_default_pdf_SFH_prediction_MIX_diffstar_scan_Xsfh_scan_Xmah_scan(
    t_table,
    sfh_lh_sig,
    n_sfh_param_grid,
    logm0_binmids,
    logm0_bin_widths,
    n_halos_per_bin,
    halo_data,
    fstar_tdelay,
    pdf_model_params_MS=DEFAULT_SFH_PDF_MAINSEQ_PARAMS,
    pdf_model_params_Q=DEFAULT_SFH_PDF_QUENCH_PARAMS,
):
    """
    Compute Diffstarpop summary statistic predictions for a mixed population of
    both main sequence and quenched galaxies.

    Parameters
    ----------
    t_table : ndarray of shape (n_t, )
        Cosmic time array in Gyr.
    sfh_lh_sig : float
        Number of sigma used to define the latin hypercube box length.
    n_sfh_param_grid : int
        Number of sample points in latin hypercube box.
    logm0_binmids : ndarray of shape (n_m0, )
        Midpoint of the logarithmic halo mass bins
    logm0_bin_widths : ndarray of shape (n_m0, )
        Logarithmic width of the halo mass bin
    n_halos_per_bin : int
        Number of halos to be randomly sub-selected in each halo mass bin.
    halo_data : ndarray of shape (4, n_halos)
        Array containing the following Diffmah parameters
        (logm0, tauc, early, late):
            logmp : float
                Base-10 log of present-day peak halo mass in units of Msun assuming h=1
            tauc : float or ndarray of shape (n_halos, )
                Transition time between the fast- and slow-accretion regimes in Gyr
            early : float or ndarray of shape (n_halos, )
                Early-time power-law index in the scaling relation M(t)~t^a
            late : float or ndarray of shape (n_halos, )
                Late-time power-law index in the scaling relation M(t)~t^a
    fstar_tdelay: float
        Time interval in Gyr for fstar definition.
        fstar = (mstar(t) - mstar(t-fstar_tdelay)) / fstar_tdelay[Gyr]
    pdf_model_params_MS : dict
        Dictionary containing the Diffstarpop parameters for the main sequence population.
        Default is DEFAULT_SFH_PDF_MAINSEQ_PARAMS.
    pdf_model_params_Q : dict
        Dictionary containing the Diffstarpop parameters for the quenched population.
        Default is DEFAULT_SFH_PDF_QUENCH_PARAMS.

    Returns
    -------
    mean_sm : ndarray of shape (n_m0, n_t)
        Average log10 Stellar Mass.
    variance_sm : ndarray of shape (n_m0, n_t)
        Variance of log10 Stellar Mass.
    mean_fstar_MS : ndarray of shape (n_m0, n_t)
        Average fstar (average SFH within some timescale) for main sequence galaxies.
    mean_fstar_Q : ndarray of shape (n_m0, n_t)
        Average fstar (average SFH within some timescale) for quenched galaxies.
    variance_fstar_MS : ndarray of shape (n_m0, n_t)
        Variance of fstar (average SFH within some timescale) for MS galaxies.
    variance_fstar_Q : ndarray of shape (n_m0, n_t)
        Variance of fstar (average SFH within some timescale) for Q galaxies.
    quench_frac : ndarray of shape (n_m0, n_t)
        Fraction of quenched galaxies.

    Notes
    -----
    PDF is a mixture Gaussian model distribution:
        PDF = f_Q * PDF_Q + (1-f_Q) * PDF_MS
    where:
        f_Q: quenched Fraction
        PDF_Q: Unimodal Gaussian for the quenched sub-population
        PDF_MS: Unimodal Gaussian for the main sequence sub-population

    Details of kernel implementation:
        Diffstar tobs and tcons loops: scan
        Diffstarpop Xmah loop: scan
        Diffstarpop Xsfh loop: scan
    """

    index_high = np.searchsorted(t_table, t_table - fstar_tdelay)
    _mask = t_table > fstar_tdelay + fstar_tdelay / 2.0
    index_select = np.arange(len(t_table))[_mask]
    index_high = index_high[_mask]

    lgt_table = jnp.log10(t_table)
    dt_table = _jax_get_dt_array(t_table)

    jran_key = jran.PRNGKey(0)
    logm0_halos, mah_tauc_halos, mah_early_halos, mah_late_halos = halo_data

    # the diffstar scan functions actually want log(tauc)
    mah_logtauc_halos = np.log10(mah_tauc_halos)

    diffmah_params_grid = get_binned_halo_sample(
        n_halos_per_bin,
        jran_key,
        logm0_binmids,
        logm0_bin_widths,
        logm0_halos,
        mah_logtauc_halos,
        mah_early_halos,
        mah_late_halos,
    )
    diffmah_params_grid = np.array(diffmah_params_grid)
    diffmah_params_grid = diffmah_params_grid.reshape(
        (4, len(logm0_binmids), n_halos_per_bin)
    )
    diffmah_params_grid = np.einsum("pmh->mhp", diffmah_params_grid)

    # Main sequence histories and weights
    sfh_param_grids_MS = get_sfh_param_grid_MS(
        pdf_model_params_MS, logm0_binmids, sfh_lh_sig, n_sfh_param_grid
    )

    means_MS, covs_MS = get_binned_means_and_covs_MS(pdf_model_params_MS, logm0_binmids)
    _res = compute_histories_on_grids_MS_diffstar_scan_Xsfh_scan_Xmah_scan(
        t_table,
        lgt_table,
        dt_table,
        index_select,
        index_high,
        fstar_tdelay,
        diffmah_params_grid,
        sfh_param_grids_MS,
    )
    mstar_histories_MS, sfr_histories_MS, fstar_histories_MS = _res

    sfstar_histories_MS = fstar_histories_MS / mstar_histories_MS[:, :, :, index_select]

    weights_MS_bin = jnp.where(sfstar_histories_MS > 1e-11, 1.0, 0.0)
    weights_MS = _get_pdf_weights_kern(sfh_param_grids_MS, means_MS, covs_MS)

    # Quenched histories and weights
    fracs_logm0, sfh_param_grids_Q = get_sfh_param_grid_Q(
        pdf_model_params_Q, logm0_binmids, sfh_lh_sig, n_sfh_param_grid
    )

    fracs_Q, means_Q, covs_Q = get_binned_means_and_covs_Q(
        pdf_model_params_Q, logm0_binmids
    )
    _res = compute_histories_on_grids_diffstar_scan_Xsfh_scan_Xmah_scan(
        t_table,
        lgt_table,
        dt_table,
        index_select,
        index_high,
        fstar_tdelay,
        diffmah_params_grid,
        sfh_param_grids_Q,
    )
    mstar_histories_Q, sfr_histories_Q, fstar_histories_Q = _res
    sfstar_histories_Q = fstar_histories_Q / mstar_histories_Q[:, :, :, index_select]
    weights_Q_bin = jnp.where(sfstar_histories_Q > 1e-11, 1.0, 0.0)
    weights_Q = _get_pdf_weights_kern(sfh_param_grids_Q, means_Q, covs_Q)

    # Concatenate arrays
    mstar_histories = jnp.concatenate((mstar_histories_MS, mstar_histories_Q), axis=1)
    sfr_histories = jnp.concatenate((sfr_histories_MS, sfr_histories_Q), axis=1)
    fstar_histories = jnp.concatenate((fstar_histories_MS, fstar_histories_Q), axis=1)

    weights_MS /= weights_MS.sum()
    weights_Q /= weights_Q.sum()

    weights_MS = jnp.einsum("ij,i->ij", weights_MS, (1.0 - fracs_Q))
    weights_Q = jnp.einsum("ij,i->ij", weights_Q, fracs_Q)

    weights = jnp.concatenate((weights_MS, weights_Q), axis=1)
    weights_bin = jnp.concatenate((weights_MS_bin, weights_Q_bin), axis=1)

    return compute_target_sumstats_from_histories(
        weights, weights_bin, mstar_histories, sfr_histories, fstar_histories
    )


def get_default_pdf_SFH_prediction_MIX_diffstar_scan_Xsfh_vmap_Xmah_vmap(
    t_table,
    sfh_lh_sig,
    n_sfh_param_grid,
    logm0_binmids,
    logm0_bin_widths,
    n_halos_per_bin,
    halo_data,
    fstar_tdelay,
    pdf_model_params_MS=DEFAULT_SFH_PDF_MAINSEQ_PARAMS,
    pdf_model_params_Q=DEFAULT_SFH_PDF_QUENCH_PARAMS,
):
    """
    Compute Diffstarpop summary statistic predictions for a mixed population of
    both main sequence and quenched galaxies.

    Parameters
    ----------
    t_table : ndarray of shape (n_t, )
        Cosmic time array in Gyr.
    sfh_lh_sig : float
        Number of sigma used to define the latin hypercube box length.
    n_sfh_param_grid : int
        Number of sample points in latin hypercube box.
    logm0_binmids : ndarray of shape (n_m0, )
        Midpoint of the logarithmic halo mass bins
    logm0_bin_widths : ndarray of shape (n_m0, )
        Logarithmic width of the halo mass bin
    n_halos_per_bin : int
        Number of halos to be randomly sub-selected in each halo mass bin.
    halo_data : ndarray of shape (4, n_halos)
        Array containing the following Diffmah parameters
        (logm0, tauc, early, late):
            logmp : float
                Base-10 log of present-day peak halo mass in units of Msun assuming h=1
            tauc : float or ndarray of shape (n_halos, )
                Transition time between the fast- and slow-accretion regimes in Gyr
            early : float or ndarray of shape (n_halos, )
                Early-time power-law index in the scaling relation M(t)~t^a
            late : float or ndarray of shape (n_halos, )
                Late-time power-law index in the scaling relation M(t)~t^a
    fstar_tdelay: float
        Time interval in Gyr for fstar definition.
        fstar = (mstar(t) - mstar(t-fstar_tdelay)) / fstar_tdelay[Gyr]
    pdf_model_params_MS : dict
        Dictionary containing the Diffstarpop parameters for the main sequence population.
        Default is DEFAULT_SFH_PDF_MAINSEQ_PARAMS.
    pdf_model_params_Q : dict
        Dictionary containing the Diffstarpop parameters for the quenched population.
        Default is DEFAULT_SFH_PDF_QUENCH_PARAMS.

    Returns
    -------
    mean_sm : ndarray of shape (n_m0, n_t)
        Average log10 Stellar Mass.
    variance_sm : ndarray of shape (n_m0, n_t)
        Variance of log10 Stellar Mass.
    mean_fstar_MS : ndarray of shape (n_m0, n_t)
        Average fstar (average SFH within some timescale) for main sequence galaxies.
    mean_fstar_Q : ndarray of shape (n_m0, n_t)
        Average fstar (average SFH within some timescale) for quenched galaxies.
    variance_fstar_MS : ndarray of shape (n_m0, n_t)
        Variance of fstar (average SFH within some timescale) for MS galaxies.
    variance_fstar_Q : ndarray of shape (n_m0, n_t)
        Variance of fstar (average SFH within some timescale) for Q galaxies.
    quench_frac : ndarray of shape (n_m0, n_t)
        Fraction of quenched galaxies.

    Notes
    -----
    PDF is a mixture Gaussian model distribution:
        PDF = f_Q * PDF_Q + (1-f_Q) * PDF_MS
    where:
        f_Q: quenched Fraction
        PDF_Q: Unimodal Gaussian for the quenched sub-population
        PDF_MS: Unimodal Gaussian for the main sequence sub-population

    Details of kernel implementation:
        Diffstar tobs and tcons loops: scan
        Diffstarpop Xmah loop: vmap
        Diffstarpop Xsfh loop: vmap
    """

    index_high = np.searchsorted(t_table, t_table - fstar_tdelay)
    _mask = t_table > fstar_tdelay + fstar_tdelay / 2.0
    index_select = np.arange(len(t_table))[_mask]
    index_high = index_high[_mask]

    lgt_table = jnp.log10(t_table)
    dt_table = _jax_get_dt_array(t_table)

    jran_key = jran.PRNGKey(0)
    logm0_halos, mah_tauc_halos, mah_early_halos, mah_late_halos = halo_data

    # the diffstar scan functions actually want log(tauc)
    mah_logtauc_halos = np.log10(mah_tauc_halos)

    diffmah_params_grid = get_binned_halo_sample(
        n_halos_per_bin,
        jran_key,
        logm0_binmids,
        logm0_bin_widths,
        logm0_halos,
        mah_logtauc_halos,
        mah_early_halos,
        mah_late_halos,
    )
    diffmah_params_grid = np.array(diffmah_params_grid)
    diffmah_params_grid = diffmah_params_grid.reshape(
        (4, len(logm0_binmids), n_halos_per_bin)
    )
    diffmah_params_grid = np.einsum("pmh->mhp", diffmah_params_grid)

    # Main sequence histories and weights
    sfh_param_grids_MS = get_sfh_param_grid_MS(
        pdf_model_params_MS, logm0_binmids, sfh_lh_sig, n_sfh_param_grid
    )

    means_MS, covs_MS = get_binned_means_and_covs_MS(pdf_model_params_MS, logm0_binmids)
    _res = compute_histories_on_grids_MS_diffstar_scan_Xsfh_vmap_Xmah_vmap(
        t_table,
        lgt_table,
        dt_table,
        index_select,
        index_high,
        fstar_tdelay,
        diffmah_params_grid,
        sfh_param_grids_MS,
    )
    mstar_histories_MS, sfr_histories_MS, fstar_histories_MS = _res

    sfstar_histories_MS = fstar_histories_MS / mstar_histories_MS[:, :, :, index_select]

    weights_MS_bin = jnp.where(sfstar_histories_MS > 1e-11, 1.0, 0.0)
    weights_MS = _get_pdf_weights_kern(sfh_param_grids_MS, means_MS, covs_MS)

    # Quenched histories and weights
    fracs_logm0, sfh_param_grids_Q = get_sfh_param_grid_Q(
        pdf_model_params_Q, logm0_binmids, sfh_lh_sig, n_sfh_param_grid
    )

    fracs_Q, means_Q, covs_Q = get_binned_means_and_covs_Q(
        pdf_model_params_Q, logm0_binmids
    )
    _res = compute_histories_on_grids_diffstar_scan_Xsfh_vmap_Xmah_vmap(
        t_table,
        lgt_table,
        dt_table,
        index_select,
        index_high,
        fstar_tdelay,
        diffmah_params_grid,
        sfh_param_grids_Q,
    )
    mstar_histories_Q, sfr_histories_Q, fstar_histories_Q = _res
    sfstar_histories_Q = fstar_histories_Q / mstar_histories_Q[:, :, :, index_select]
    weights_Q_bin = jnp.where(sfstar_histories_Q > 1e-11, 1.0, 0.0)
    weights_Q = _get_pdf_weights_kern(sfh_param_grids_Q, means_Q, covs_Q)

    # Concatenate arrays
    mstar_histories = jnp.concatenate((mstar_histories_MS, mstar_histories_Q), axis=1)
    sfr_histories = jnp.concatenate((sfr_histories_MS, sfr_histories_Q), axis=1)
    fstar_histories = jnp.concatenate((fstar_histories_MS, fstar_histories_Q), axis=1)

    weights_MS /= weights_MS.sum()
    weights_Q /= weights_Q.sum()

    weights_MS = jnp.einsum("ij,i->ij", weights_MS, (1.0 - fracs_Q))
    weights_Q = jnp.einsum("ij,i->ij", weights_Q, fracs_Q)

    weights = jnp.concatenate((weights_MS, weights_Q), axis=1)
    weights_bin = jnp.concatenate((weights_MS_bin, weights_Q_bin), axis=1)

    return compute_target_sumstats_from_histories(
        weights, weights_bin, mstar_histories, sfr_histories, fstar_histories
    )


def PDF_weight_sfh_population_sumstats_wrapper(
    t_table,
    sfh_lh_sig,
    n_sfh_param_grid,
    logm0_binmids,
    logm0_bin_widths,
    n_halos_per_bin,
    halo_data,
    fstar_tdelay,
    population_model="Q",
    diffstar_kernel="scan",
    diffstarpop_kernel="vmap",
    pdf_model_params_MS=DEFAULT_SFH_PDF_MAINSEQ_PARAMS,
    pdf_model_params_Q=DEFAULT_SFH_PDF_QUENCH_PARAMS,
):
    """
    Wrapper function to compute Diffstarpop summary statistic predictions.

    Parameters
    ----------
    t_table : ndarray of shape (n_t, )
        Cosmic time array in Gyr.
    sfh_lh_sig : float
        Number of sigma used to define the latin hypercube box length.
    n_sfh_param_grid : int
        Number of sample points in latin hypercube box.
    logm0_binmids : ndarray of shape (n_m0, )
        Midpoint of the logarithmic halo mass bins
    logm0_bin_widths : ndarray of shape (n_m0, )
        Logarithmic width of the halo mass bin
    n_halos_per_bin : int
        Number of halos to be randomly sub-selected in each halo mass bin.
    halo_data : ndarray of shape (4, n_halos)
        Array containing the following Diffmah parameters
        (logm0, tauc, early, late):
            logmp : float
                Base-10 log of present-day peak halo mass in units of Msun assuming h=1
            tauc : float or ndarray of shape (n_halos, )
                Transition time between the fast- and slow-accretion regimes in Gyr
            early : float or ndarray of shape (n_halos, )
                Early-time power-law index in the scaling relation M(t)~t^a
            late : float or ndarray of shape (n_halos, )
                Late-time power-law index in the scaling relation M(t)~t^a
    fstar_tdelay: float
        Time interval in Gyr for fstar definition.
        fstar = (mstar(t) - mstar(t-fstar_tdelay)) / fstar_tdelay[Gyr]
    population_model: string
        Type of DiffstarPop model. Options are 'Q', 'MS', 'MIX'.
    diffstar_kernel: string
        Type of diffstar kernel implementation. Options are 'vmap', 'scan'.
    diffstarpop_kernel: string
        Type of diffstarpop kernel implementation. Options are 'vmap', 'scan'.
    pdf_model_params_MS : dict
        Dictionary containing the Diffstarpop parameters for the main sequence population.
        Default is DEFAULT_SFH_PDF_MAINSEQ_PARAMS.
    pdf_model_params_Q : dict
        Dictionary containing the Diffstarpop parameters for the quenched population.
        Default is DEFAULT_SFH_PDF_QUENCH_PARAMS.

    Returns
    -------
    mean_sm : ndarray of shape (n_m0, n_t)
        Average log10 Stellar Mass.
    variance_sm : ndarray of shape (n_m0, n_t)
        Variance of log10 Stellar Mass.
    mean_fstar_MS : ndarray of shape (n_m0, n_t)
        Average fstar (average SFH within some timescale) for main sequence galaxies.
    mean_fstar_Q : ndarray of shape (n_m0, n_t)
        Average fstar (average SFH within some timescale) for quenched galaxies.
    variance_fstar_MS : ndarray of shape (n_m0, n_t)
        Variance of fstar (average SFH within some timescale) for MS galaxies.
    variance_fstar_Q : ndarray of shape (n_m0, n_t)
        Variance of fstar (average SFH within some timescale) for Q galaxies.
    quench_frac : ndarray of shape (n_m0, n_t)
        Fraction of quenched galaxies.
    """
    assert population_model in ["Q", "MS", "MIX"], NotImplementedError(
        "Valid 'population_model' options are 'Q', 'MS', 'MIX'."
    )
    assert diffstar_kernel in ["scan", "vmap"], NotImplementedError(
        "Valid 'diffstar_kernel' options are 'scan', 'vmap'."
    )
    assert diffstarpop_kernel in ["scan", "vmap"], NotImplementedError(
        "Valid 'diffstarpop_kernel' options are 'scan', 'vmap'."
    )

    if population_model == "Q":
        if diffstar_kernel == "vmap":
            if diffstarpop_kernel == "scan":
                return get_default_pdf_SFH_prediction_Q_diffstar_vmap_Xsfh_scan_Xmah_scan(
                    t_table,
                    sfh_lh_sig,
                    n_sfh_param_grid,
                    logm0_binmids,
                    logm0_bin_widths,
                    n_halos_per_bin,
                    halo_data,
                    fstar_tdelay,
                    pdf_model_params=pdf_model_params_Q,
                )
            elif diffstarpop_kernel == "vmap":
                return get_default_pdf_SFH_prediction_Q_diffstar_vmap_Xsfh_vmap_Xmah_vmap(
                    t_table,
                    sfh_lh_sig,
                    n_sfh_param_grid,
                    logm0_binmids,
                    logm0_bin_widths,
                    n_halos_per_bin,
                    halo_data,
                    fstar_tdelay,
                    pdf_model_params=pdf_model_params_Q,
                )
        elif diffstar_kernel == "scan":
            if diffstarpop_kernel == "scan":
                return get_default_pdf_SFH_prediction_Q_diffstar_scan_Xsfh_scan_Xmah_scan(
                    t_table,
                    sfh_lh_sig,
                    n_sfh_param_grid,
                    logm0_binmids,
                    logm0_bin_widths,
                    n_halos_per_bin,
                    halo_data,
                    fstar_tdelay,
                    pdf_model_params=pdf_model_params_Q,
                )
            elif diffstarpop_kernel == "vmap":
                return get_default_pdf_SFH_prediction_Q_diffstar_scan_Xsfh_vmap_Xmah_vmap(
                    t_table,
                    sfh_lh_sig,
                    n_sfh_param_grid,
                    logm0_binmids,
                    logm0_bin_widths,
                    n_halos_per_bin,
                    halo_data,
                    fstar_tdelay,
                    pdf_model_params=pdf_model_params_Q,
                )
    elif population_model == "MS":
        if diffstar_kernel == "vmap":
            if diffstarpop_kernel == "scan":
                return get_default_pdf_SFH_prediction_MS_diffstar_vmap_Xsfh_scan_Xmah_scan(
                    t_table,
                    sfh_lh_sig,
                    n_sfh_param_grid,
                    logm0_binmids,
                    logm0_bin_widths,
                    n_halos_per_bin,
                    halo_data,
                    fstar_tdelay,
                    pdf_model_params=pdf_model_params_MS,
                )
            elif diffstarpop_kernel == "vmap":
                return get_default_pdf_SFH_prediction_MS_diffstar_vmap_Xsfh_vmap_Xmah_vmap(
                    t_table,
                    sfh_lh_sig,
                    n_sfh_param_grid,
                    logm0_binmids,
                    logm0_bin_widths,
                    n_halos_per_bin,
                    halo_data,
                    fstar_tdelay,
                    pdf_model_params=pdf_model_params_MS,
                )
        elif diffstar_kernel == "scan":
            if diffstarpop_kernel == "scan":
                return get_default_pdf_SFH_prediction_MS_diffstar_scan_Xsfh_scan_Xmah_scan(
                    t_table,
                    sfh_lh_sig,
                    n_sfh_param_grid,
                    logm0_binmids,
                    logm0_bin_widths,
                    n_halos_per_bin,
                    halo_data,
                    fstar_tdelay,
                    pdf_model_params=pdf_model_params_MS,
                )
            elif diffstarpop_kernel == "vmap":
                return get_default_pdf_SFH_prediction_MS_diffstar_scan_Xsfh_vmap_Xmah_vmap(
                    t_table,
                    sfh_lh_sig,
                    n_sfh_param_grid,
                    logm0_binmids,
                    logm0_bin_widths,
                    n_halos_per_bin,
                    halo_data,
                    fstar_tdelay,
                    pdf_model_params=pdf_model_params_MS,
                )
    elif population_model == "MIX":
        if diffstar_kernel == "vmap":
            if diffstarpop_kernel == "scan":
                return get_default_pdf_SFH_prediction_MIX_diffstar_vmap_Xsfh_scan_Xmah_scan(
                    t_table,
                    sfh_lh_sig,
                    n_sfh_param_grid,
                    logm0_binmids,
                    logm0_bin_widths,
                    n_halos_per_bin,
                    halo_data,
                    fstar_tdelay,
                    pdf_model_params_MS=pdf_model_params_MS,
                    pdf_model_params_Q=pdf_model_params_Q,
                )
            elif diffstarpop_kernel == "vmap":
                return get_default_pdf_SFH_prediction_MIX_diffstar_vmap_Xsfh_vmap_Xmah_vmap(
                    t_table,
                    sfh_lh_sig,
                    n_sfh_param_grid,
                    logm0_binmids,
                    logm0_bin_widths,
                    n_halos_per_bin,
                    halo_data,
                    fstar_tdelay,
                    pdf_model_params_MS=pdf_model_params_MS,
                    pdf_model_params_Q=pdf_model_params_Q,
                )
        elif diffstar_kernel == "scan":
            if diffstarpop_kernel == "scan":
                return get_default_pdf_SFH_prediction_MIX_diffstar_scan_Xsfh_scan_Xmah_scan(
                    t_table,
                    sfh_lh_sig,
                    n_sfh_param_grid,
                    logm0_binmids,
                    logm0_bin_widths,
                    n_halos_per_bin,
                    halo_data,
                    fstar_tdelay,
                    pdf_model_params_MS=pdf_model_params_MS,
                    pdf_model_params_Q=pdf_model_params_Q,
                )
            elif diffstarpop_kernel == "vmap":
                return get_default_pdf_SFH_prediction_MIX_diffstar_scan_Xsfh_vmap_Xmah_vmap(
                    t_table,
                    sfh_lh_sig,
                    n_sfh_param_grid,
                    logm0_binmids,
                    logm0_bin_widths,
                    n_halos_per_bin,
                    halo_data,
                    fstar_tdelay,
                    pdf_model_params_MS=pdf_model_params_MS,
                    pdf_model_params_Q=pdf_model_params_Q,
                )


def get_default_pdf_SFH_prediction_Q_withR_diffstar_scan_Xsfh_vmap_Xmah_vmap(
    t_table,
    sfh_lh_sig,
    n_sfh_param_grid,
    logm0_binmids,
    logm0_bin_widths,
    n_halos_per_bin,
    halo_data,
    p50_halos,
    fstar_tdelay,
    pdf_model_params=DEFAULT_SFH_PDF_QUENCH_PARAMS,
    R_model_params=DEFAULT_R_QUENCH_PARAMS,
):
    """
    Compute Diffstarpop summary statistic predictions for a population of
    quenched galaxies.

    Parameters
    ----------
    t_table : ndarray of shape (n_t, )
        Cosmic time array in Gyr.
    sfh_lh_sig : float
        Number of sigma used to define the latin hypercube box length.
    n_sfh_param_grid : int
        Number of sample points in latin hypercube box.
    logm0_binmids : ndarray of shape (n_m0, )
        Midpoint of the logarithmic halo mass bins
    logm0_bin_widths : ndarray of shape (n_m0, )
        Logarithmic width of the halo mass bin
    n_halos_per_bin : int
        Number of halos to be randomly sub-selected in each halo mass bin.
    halo_data : ndarray of shape (4, n_halos)
        Array containing the following Diffmah parameters
        (logm0, tauc, early, late):
            logmp : float
                Base-10 log of present-day peak halo mass in units of Msun assuming h=1
            tauc : float or ndarray of shape (n_halos, )
                Transition time between the fast- and slow-accretion regimes in Gyr
            early : float or ndarray of shape (n_halos, )
                Early-time power-law index in the scaling relation M(t)~t^a
            late : float or ndarray of shape (n_halos, )
                Late-time power-law index in the scaling relation M(t)~t^a
    fstar_tdelay: float
        Time interval in Gyr for fstar definition.
        fstar = (mstar(t) - mstar(t-fstar_tdelay)) / fstar_tdelay[Gyr]
    pdf_model_params : dict
        Dictionary containing the Diffstarpop parameters for the quenched population.
        Default is DEFAULT_SFH_PDF_QUENCH_PARAMS.
    R_model_params : dict
        Dictionary containing the Diffstarpop parameters for the correlation R with
        halo formation time percentile p50 for the quenched population.
        Default is DEFAULT_R_QUENCH_PARAMS.

    Returns
    -------
    mean_sm : ndarray of shape (n_m0, n_t)
        Average log10 Stellar Mass.
    variance_sm : ndarray of shape (n_m0, n_t)
        Variance of log10 Stellar Mass.
    mean_fstar_MS : ndarray of shape (n_m0, n_t)
        Average fstar (average SFH within some timescale) for main sequence galaxies.
    mean_fstar_Q : ndarray of shape (n_m0, n_t)
        Average fstar (average SFH within some timescale) for quenched galaxies.
    variance_fstar_MS : ndarray of shape (n_m0, n_t)
        Variance of fstar (average SFH within some timescale) for MS galaxies.
    variance_fstar_Q : ndarray of shape (n_m0, n_t)
        Variance of fstar (average SFH within some timescale) for Q galaxies.
    quench_frac : ndarray of shape (n_m0, n_t)
        Fraction of quenched galaxies.

    Notes
    -----
    PDF is a unimodal Gaussian distribution on unbound Diffstar parameters.

    Details of kernel implementation:
        Diffstar tobs and tcons loops: scan
        Diffstarpop Xmah loop: vmap
        Diffstarpop Xsfh loop: vmap
    """

    index_high = np.searchsorted(t_table, t_table - fstar_tdelay)
    _mask = t_table > fstar_tdelay + fstar_tdelay / 2.0
    index_select = np.arange(len(t_table))[_mask]
    index_high = index_high[_mask]

    lgt_table = jnp.log10(t_table)
    dt_table = _jax_get_dt_array(t_table)

    jran_key = jran.PRNGKey(0)
    logm0_halos, mah_tauc_halos, mah_early_halos, mah_late_halos = halo_data

    # the diffstar scan functions actually want log(tauc)
    mah_logtauc_halos = np.log10(mah_tauc_halos)

    diffmah_params_grid = get_binned_halo_sample(
        n_halos_per_bin,
        jran_key,
        logm0_binmids,
        logm0_bin_widths,
        logm0_halos,
        mah_logtauc_halos,
        mah_early_halos,
        mah_late_halos,
        p50_halos,
    )

    diffmah_params_grid = np.array(diffmah_params_grid)
    p50 = diffmah_params_grid[-1, :].reshape((len(logm0_binmids), n_halos_per_bin))
    diffmah_params_grid = diffmah_params_grid[:-1, :]
    diffmah_params_grid = diffmah_params_grid.reshape(
        (4, len(logm0_binmids), n_halos_per_bin)
    )
    diffmah_params_grid = np.einsum("pmh->mhp", diffmah_params_grid)

    fracs_logm0, sfh_param_grids = get_sfh_param_grid_Q(
        pdf_model_params, logm0_binmids, sfh_lh_sig, n_sfh_param_grid
    )

    fracs, means, covs = get_binned_means_and_covs_Q(pdf_model_params, logm0_binmids)

    R_vals_quench = _get_slopes_quench(logm0_binmids, **R_model_params)
    R_vals_quench = jnp.array(R_vals_quench).T

    shifts = jnp.einsum("mp,mh->mhp", R_vals_quench, (p50 - 0.5))

    means_shifted = means[:, None, :] + shifts

    _res = compute_histories_on_grids_diffstar_scan_Xsfh_vmap_Xmah_vmap(
        t_table,
        lgt_table,
        dt_table,
        index_select,
        index_high,
        fstar_tdelay,
        diffmah_params_grid,
        sfh_param_grids,
    )
    mstar_histories, sfr_histories, fstar_histories = _res

    sfstar_histories = fstar_histories / mstar_histories[:, :, :, index_select]

    weights_quench_bin = jnp.where(sfstar_histories > 1e-11, 1.0, 0.0)

    weights = _get_pdf_weights_kern_withR(sfh_param_grids, means_shifted, covs)

    return compute_target_sumstats_from_histories_diffhaloweights(
        weights, weights_quench_bin, mstar_histories, sfr_histories, fstar_histories
    )


def latin_hypercube_quantities(num_dim, num_samples, sfh_lh_sig):
    sampler = qmc.LatinHypercube(d=num_dim, centered=True)
    sample = sampler.random(n=num_samples)
    box_limits = norm.cdf([-sfh_lh_sig, sfh_lh_sig])

    sample = sample * np.diff(box_limits) + box_limits[0]

    pixel_size = np.diff(box_limits) / num_samples

    lower_lim_pixel = np.array([sample for i in range(num_dim)])
    upper_lim_pixel = np.array([sample for i in range(num_dim)])
    for i in range(num_dim):
        lower_lim_pixel[i, :, i] -= pixel_size / 2.0
        upper_lim_pixel[i, :, i] += pixel_size / 2.0

    sample_ppf = stats_jax.norm.ppf(sample)
    lower_lim_pixel_ppf = stats_jax.norm.ppf(lower_lim_pixel)
    upper_lim_pixel_ppf = stats_jax.norm.ppf(upper_lim_pixel)

    return sample_ppf, lower_lim_pixel_ppf, upper_lim_pixel_ppf


@jjit
def _get_eigenbasis_transform_kern(cov):
    """X_orig = X_espace.dot(T)"""
    evals, V = jnp.linalg.eigh(cov)
    R, S = V, jnp.sqrt(jnp.diag(evals))
    T = R.dot(S).T
    return jnp.real(T)


@jjit
def _eigenrotate_and_shift_singlebox(box, mu, cov):
    T = _get_eigenbasis_transform_kern(cov)
    return box.dot(T) + mu


@jjit
def _calculate_samples_and_volume_kern(box, mu, cov, lower_lim_pixel, upper_lim_pixel):

    T = _get_eigenbasis_transform_kern(cov)

    box_ppf_rotated = box.dot(T) + mu

    lower_lim_pixel_ppf_rotated = lower_lim_pixel.dot(T)
    upper_lim_pixel_ppf_rotated = upper_lim_pixel.dot(T)

    pixel_limits_distance = jnp.sqrt(
        jnp.sum(
            (upper_lim_pixel_ppf_rotated - lower_lim_pixel_ppf_rotated) ** 2, axis=1
        )
    )
    pixel_volume = jnp.prod(pixel_limits_distance)

    return box_ppf_rotated, pixel_volume


# _kern1 = jjit(vmap(_eigenrotate_and_shift_singlebox, in_axes=(0, None, None)))
# calculate_samples = jjit(vmap(_kern1, in_axes=(None, 0, 0)))

_kern1 = jjit(vmap(_calculate_samples_and_volume_kern, in_axes=(0, None, None, 1, 1)))
calculate_samples_and_volume = jjit(vmap(_kern1, in_axes=(None, 0, 0, None, None)))


def get_default_pdf_SFH_diffMC_prediction_Q_diffstar_scan_Xsfh_vmap_Xmah_vmap(
    t_table,
    sfh_lh_sig,
    n_sfh_param_grid,
    logm0_binmids,
    logm0_bin_widths,
    n_halos_per_bin,
    halo_data,
    fstar_tdelay,
    pdf_model_params=DEFAULT_SFH_PDF_QUENCH_PARAMS,
):
    """
    Compute Diffstarpop summary statistic predictions for a population of
    quenched galaxies.

    Parameters
    ----------
    t_table : ndarray of shape (n_t, )
        Cosmic time array in Gyr.
    sfh_lh_sig : float
        Number of sigma used to define the latin hypercube box length.
    n_sfh_param_grid : int
        Number of sample points in latin hypercube box.
    logm0_binmids : ndarray of shape (n_m0, )
        Midpoint of the logarithmic halo mass bins
    logm0_bin_widths : ndarray of shape (n_m0, )
        Logarithmic width of the halo mass bin
    n_halos_per_bin : int
        Number of halos to be randomly sub-selected in each halo mass bin.
    halo_data : ndarray of shape (4, n_halos)
        Array containing the following Diffmah parameters
        (logm0, tauc, early, late):
            logmp : float
                Base-10 log of present-day peak halo mass in units of Msun assuming h=1
            tauc : float or ndarray of shape (n_halos, )
                Transition time between the fast- and slow-accretion regimes in Gyr
            early : float or ndarray of shape (n_halos, )
                Early-time power-law index in the scaling relation M(t)~t^a
            late : float or ndarray of shape (n_halos, )
                Late-time power-law index in the scaling relation M(t)~t^a
    fstar_tdelay: float
        Time interval in Gyr for fstar definition.
        fstar = (mstar(t) - mstar(t-fstar_tdelay)) / fstar_tdelay[Gyr]
    pdf_model_params : dict
        Dictionary containing the Diffstarpop parameters for the quenched population.
        Default is DEFAULT_SFH_PDF_QUENCH_PARAMS.

    Returns
    -------
    mean_sm : ndarray of shape (n_m0, n_t)
        Average log10 Stellar Mass.
    variance_sm : ndarray of shape (n_m0, n_t)
        Variance of log10 Stellar Mass.
    mean_fstar_MS : ndarray of shape (n_m0, n_t)
        Average fstar (average SFH within some timescale) for main sequence galaxies.
    mean_fstar_Q : ndarray of shape (n_m0, n_t)
        Average fstar (average SFH within some timescale) for quenched galaxies.
    variance_fstar_MS : ndarray of shape (n_m0, n_t)
        Variance of fstar (average SFH within some timescale) for MS galaxies.
    variance_fstar_Q : ndarray of shape (n_m0, n_t)
        Variance of fstar (average SFH within some timescale) for Q galaxies.
    quench_frac : ndarray of shape (n_m0, n_t)
        Fraction of quenched galaxies.

    Notes
    -----
    PDF is a unimodal Gaussian distribution on unbound Diffstar parameters.

    Details of kernel implementation:
        Diffstar tobs and tcons loops: scan
        Diffstarpop Xmah loop: vmap
        Diffstarpop Xsfh loop: vmap
    """

    index_high = np.searchsorted(t_table, t_table - fstar_tdelay)
    _mask = t_table > fstar_tdelay + fstar_tdelay / 2.0
    index_select = np.arange(len(t_table))[_mask]
    index_high = index_high[_mask]

    lgt_table = jnp.log10(t_table)
    dt_table = _jax_get_dt_array(t_table)

    jran_key = jran.PRNGKey(0)
    logm0_halos, mah_tauc_halos, mah_early_halos, mah_late_halos = halo_data

    # the diffstar scan functions actually want log(tauc)
    mah_logtauc_halos = np.log10(mah_tauc_halos)

    diffmah_params_grid = get_binned_halo_sample(
        n_halos_per_bin,
        jran_key,
        logm0_binmids,
        logm0_bin_widths,
        logm0_halos,
        mah_logtauc_halos,
        mah_early_halos,
        mah_late_halos,
    )
    diffmah_params_grid = np.array(diffmah_params_grid)
    diffmah_params_grid = diffmah_params_grid.reshape(
        (4, len(logm0_binmids), n_halos_per_bin)
    )
    diffmah_params_grid = np.einsum("pmh->mhp", diffmah_params_grid)

    num_dim = 8

    fracs, means, covs = get_smah_means_and_covs_quench(
        logm0_binmids, **pdf_model_params
    )
    _res = latin_hypercube_quantities(num_dim, n_sfh_param_grid, sfh_lh_sig)
    unit_gaussian_box_samples, lower_lim_pixel, upper_lim_pixel = _res
    _res = calculate_samples_and_volume(
        unit_gaussian_box_samples, means, covs, lower_lim_pixel, upper_lim_pixel
    )
    sfh_param_grids, volumes = _res
    # sfh_param_grids = calculate_samples(unit_gaussian_box_samples, means, covs)

    # volumes have a very small value,
    # which creates problems later on when computing gradients
    volumes = volumes / jnp.mean(volumes, axis=1)[:, None]

    _res = compute_histories_on_grids_diffstar_scan_Xsfh_vmap_Xmah_vmap(
        t_table,
        lgt_table,
        dt_table,
        index_select,
        index_high,
        fstar_tdelay,
        diffmah_params_grid,
        sfh_param_grids,
    )
    mstar_histories, sfr_histories, fstar_histories = _res

    sfstar_histories = fstar_histories / mstar_histories[:, :, :, index_select]

    weights_quench_bin = jnp.where(sfstar_histories > 1e-11, 1.0, 0.0)

    mstar_histories = jnp.where(mstar_histories > 0.0, jnp.log10(mstar_histories), 0.0)
    sfr_histories = jnp.where(sfr_histories > 0.0, jnp.log10(sfr_histories), 0.0)
    fstar_histories = jnp.where(fstar_histories > 0.0, jnp.log10(fstar_histories), 0.0)

    weights = _get_pdf_weights_kern(sfh_param_grids, means, covs)

    weights = weights * volumes

    return compute_target_sumstats_from_histories(
        weights, weights_quench_bin, mstar_histories, sfr_histories, fstar_histories
    )


def get_default_pdf_SFH_diffMC_prediction_Q_withR_diffstar_scan_Xsfh_vmap_Xmah_vmap(
    t_table,
    sfh_lh_sig,
    n_sfh_param_grid,
    logm0_binmids,
    logm0_bin_widths,
    p50_binmids,
    p50_bin_widths,
    n_halos_per_bin,
    halo_data,
    p50_halos,
    p50_index,
    fstar_tdelay,
    pdf_model_params=DEFAULT_SFH_PDF_QUENCH_PARAMS,
    R_model_params=DEFAULT_R_QUENCH_PARAMS,
):
    """
    Compute Diffstarpop summary statistic predictions for a population of
    quenched galaxies.

    Parameters
    ----------
    t_table : ndarray of shape (n_t, )
        Cosmic time array in Gyr.
    sfh_lh_sig : float
        Number of sigma used to define the latin hypercube box length.
    n_sfh_param_grid : int
        Number of sample points in latin hypercube box.
    logm0_binmids : ndarray of shape (n_m0, )
        Midpoint of the logarithmic halo mass bins
    logm0_bin_widths : ndarray of shape (n_m0, )
        Logarithmic width of the halo mass bin
    p50_binmids : ndarray of shape (n_bins_p50, )
        Midpoint of the formation time percentile bins.
    p50_bin_widths : ndarray of shape (n_bins_p50, )
        Width of the formation time percentile bins.
    n_halos_per_bin : int
        Number of halos to be randomly sub-selected in each halo mass bin.
    halo_data : ndarray of shape (4, n_halos)
        Array containing the following Diffmah parameters
        (logm0, tauc, early, late):
            logmp : float
                Base-10 log of present-day peak halo mass in units of Msun assuming h=1
            tauc : float or ndarray of shape (n_halos, )
                Transition time between the fast- and slow-accretion regimes in Gyr
            early : float or ndarray of shape (n_halos, )
                Early-time power-law index in the scaling relation M(t)~t^a
            late : float or ndarray of shape (n_halos, )
                Late-time power-law index in the scaling relation M(t)~t^a
    p50_halos : ndarray of shape (n_halos, )
        Formation time percentile for every halo.
    p50_index : ndarray of shape (m, n_bins_p50 / m)
        Indices defining m formation time bins for summary statistic
        predictions conditioned on formation time.
    fstar_tdelay: float
        Time interval in Gyr for fstar definition.
        fstar = (mstar(t) - mstar(t-fstar_tdelay)) / fstar_tdelay[Gyr]
    pdf_model_params : dict
        Dictionary containing the Diffstarpop parameters for the quenched population.
        Default is DEFAULT_SFH_PDF_QUENCH_PARAMS.

    Returns
    -------
    mean_sm : ndarray of shape (n_m0, n_t)
        Average log10 Stellar Mass.
    variance_sm : ndarray of shape (n_m0, n_t)
        Variance of log10 Stellar Mass.
    mean_fstar_MS : ndarray of shape (n_m0, n_t)
        Average fstar (average SFH within some timescale) for main sequence galaxies.
    mean_fstar_Q : ndarray of shape (n_m0, n_t)
        Average fstar (average SFH within some timescale) for quenched galaxies.
    variance_fstar_MS : ndarray of shape (n_m0, n_t)
        Variance of fstar (average SFH within some timescale) for MS galaxies.
    variance_fstar_Q : ndarray of shape (n_m0, n_t)
        Variance of fstar (average SFH within some timescale) for Q galaxies.
    quench_frac : ndarray of shape (n_m0, n_t)
        Fraction of quenched galaxies.

    Notes
    -----
    PDF is a unimodal Gaussian distribution on unbound Diffstar parameters.

    Details of kernel implementation:
        Diffstar tobs and tcons loops: scan
        Diffstarpop Xmah loop: vmap
        Diffstarpop Xsfh loop: vmap
    """
    assert (
        np.prod(p50_index.shape) == p50_binmids.shape[0]
    ), "p50_index needs to have the same number of indices as p50_binmids bins."

    index_high = np.searchsorted(t_table, t_table - fstar_tdelay)
    _mask = t_table > fstar_tdelay + fstar_tdelay / 2.0
    index_select = np.arange(len(t_table))[_mask]
    index_high = index_high[_mask]

    lgt_table = jnp.log10(t_table)
    dt_table = _jax_get_dt_array(t_table)

    jran_key = jran.PRNGKey(0)
    logm0_halos, mah_tauc_halos, mah_early_halos, mah_late_halos = halo_data

    # the diffstar scan functions actually want log(tauc)
    mah_logtauc_halos = np.log10(mah_tauc_halos)

    nbins_mass = len(logm0_binmids)
    nbins_p50 = len(p50_binmids)
    nt = len(t_table)
    nt_fstar = len(index_select)

    diffmah_params_grid = get_binned_halo_sample_p50(
        n_halos_per_bin,
        jran_key,
        logm0_binmids,
        logm0_bin_widths,
        logm0_halos,
        p50_binmids,
        p50_bin_widths,
        p50_halos,
        mah_logtauc_halos,
        mah_early_halos,
        mah_late_halos,
        # p50_halos,
    )
    # breakpoint()
    num_dim = 8

    fracs, means, covs = get_smah_means_and_covs_quench(
        logm0_binmids, **pdf_model_params
    )
    _res = latin_hypercube_quantities(num_dim, n_sfh_param_grid, sfh_lh_sig)
    unit_gaussian_box_samples, lower_lim_pixel, upper_lim_pixel = _res
    _res = calculate_samples_and_volume(
        unit_gaussian_box_samples, means, covs, lower_lim_pixel, upper_lim_pixel
    )
    sfh_param_grids, volumes = _res

    R_vals_quench = _get_slopes_quench(logm0_binmids, **R_model_params)
    R_vals_quench = jnp.array(R_vals_quench).T

    shifts = jnp.einsum("mp,h->mhp", R_vals_quench, (p50_binmids - 0.5))

    means_shifted = means[:, None, :] + shifts

    # sfh_param_grids = calculate_samples(unit_gaussian_box_samples, means, covs)

    # volumes have a very small value,
    # which creates problems later on when computing gradients
    volumes = volumes / jnp.mean(volumes, axis=1)[:, None]

    diffmah_params_grid = diffmah_params_grid.reshape(
        (nbins_mass * nbins_p50, n_halos_per_bin, 4)
    )

    sfh_param_grids = shifts[:, :, None, :] + sfh_param_grids[:, None, :, :]
    sfh_param_grids = sfh_param_grids.reshape(
        (nbins_mass * nbins_p50, n_sfh_param_grid, num_dim)
    )
    # means_shifted = means_shifted.reshape((nbins_mass * nbins_p50, num_dim))

    _res = compute_histories_on_grids_diffstar_scan_Xsfh_vmap_Xmah_vmap(
        t_table,
        lgt_table,
        dt_table,
        index_select,
        index_high,
        fstar_tdelay,
        diffmah_params_grid,
        sfh_param_grids,
    )
    mstar_histories, sfr_histories, fstar_histories = _res
    sfstar_histories = fstar_histories / mstar_histories[:, :, :, index_select]

    mstar_histories = mstar_histories.reshape(
        (nbins_mass, nbins_p50, n_sfh_param_grid, n_halos_per_bin, nt)
    )
    sfr_histories = sfr_histories.reshape(
        (nbins_mass, nbins_p50, n_sfh_param_grid, n_halos_per_bin, nt)
    )
    fstar_histories = fstar_histories.reshape(
        (nbins_mass, nbins_p50, n_sfh_param_grid, n_halos_per_bin, nt_fstar)
    )
    sfstar_histories = sfstar_histories.reshape(
        (nbins_mass, nbins_p50, n_sfh_param_grid, n_halos_per_bin, nt_fstar)
    )

    weights_quench_bin = jnp.where(sfstar_histories > 1e-11, 1.0, 0.0)

    mstar_histories = jnp.where(mstar_histories > 0.0, jnp.log10(mstar_histories), 0.0)
    sfr_histories = jnp.where(sfr_histories > 0.0, jnp.log10(sfr_histories), 0.0)
    fstar_histories = jnp.where(fstar_histories > 0.0, jnp.log10(fstar_histories), 0.0)

    sfh_param_grids = sfh_param_grids.reshape(
        (nbins_mass, nbins_p50, n_sfh_param_grid, num_dim)
    )
    weights = _get_pdf_weights_kern_withR(sfh_param_grids, means_shifted, covs)

    weights = jnp.einsum("mpg,mg->mpg", weights, volumes)

    # breakpoint()
    return compute_target_sumstats_from_histories_p50(
        weights,
        weights_quench_bin,
        mstar_histories,
        sfr_histories,
        fstar_histories,
        p50_index,
    )


@jjit
def compute_target_sumstats_from_histories_p50(
    weights_pdf,
    weights_quench_bin,
    mstar_histories,
    sfr_histories,
    fstar_histories,
    p50_index,
):
    """
    Compute differentiable summary statistics from pdf-weighting the histories
    computed in a latin hypercube grid (or other uniform volume schemes).

    Parameters
    ----------
    weights_pdf : ndarray of shape (n_m0, n_p50, n_sfh_grid)
        PDF weight of the latin hypercube grid points.
    weights_quench_bin : ndarray of shape (n_m0, n_p50, n_sfh_grid, n_per_m0, n_t)
        Weight array indicating when galaxy history is quenched.
            0: sSFR(t) < 1e-11
            1: sSFR(t) > 1e-11
    mstar_histories : ndarray of shape (n_m0, n_p50, n_sfh_grid, n_per_m0, n_t)
        log10 SMH at each latin hypercube grid point.
    sfr_histories : ndarray of shape (n_m0, n_p50, n_sfh_grid, n_per_m0, n_t)
        log10 SFH at each latin hypercube grid point.
    fstar_histories : ndarray of shape (n_m0, n_p50, n_sfh_grid, n_per_m0, n_t)
        log10 Fstar history at each latin hypercube grid point.

    Returns
    -------
    mean_sm : ndarray of shape (n_m0, n_t)
        Average log10 Stellar Mass.
    variance_sm : ndarray of shape (n_m0, n_t)
        Variance of log10 Stellar Mass.
    mean_fstar_MS : ndarray of shape (n_m0, n_t)
        Average fstar (average SFH within some timescale) for main sequence galaxies.
    mean_fstar_Q : ndarray of shape (n_m0, n_t)
        Average fstar (average SFH within some timescale) for quenched galaxies.
    variance_fstar_MS : ndarray of shape (n_m0, n_t)
        Variance of fstar (average SFH within some timescale) for MS galaxies.
    variance_fstar_Q : ndarray of shape (n_m0, n_t)
        Variance of fstar (average SFH within some timescale) for Q galaxies.
    quench_frac : ndarray of shape (n_m0, n_t)
        Fraction of quenched galaxies.
    mean_sm_p50 : ndarray of shape (n_m0, n_p50, n_t)
        Average log10 Stellar Mass for formation time percentile bins.
    variance_sm_p50 : ndarray of shape (n_m0, n_p50, n_t)
        Variance of log10 Stellar Mass for formation time percentile bins.
    quench_frac_p50 : ndarray of shape (n_m0, n_p50, n_t)
        Fraction of quenched galaxies for formation time percentile bins.
    """

    mstar_histories = mstar_histories[:, p50_index]
    # sfr_histories = sfr_histories[:, p50_index]
    fstar_histories = fstar_histories[:, p50_index]
    weights_quench_bin = weights_quench_bin[:, p50_index]
    weights_pdf = weights_pdf[:, p50_index]

    fstar_MS = fstar_histories * weights_quench_bin
    fstar_Q = fstar_histories * (1.0 - weights_quench_bin)

    NM, NP, NP2, NSFH, NHALO, NT = mstar_histories.shape

    w_sum = 1.0 / jnp.sum(weights_pdf, axis=3)
    weights_pdf = jnp.einsum("abcd,abc->abcd", weights_pdf, w_sum)
    weights_pdf_MS = jnp.einsum("abcd,abcdef->abcdef", weights_pdf, weights_quench_bin)
    weights_pdf_Q = jnp.einsum(
        "abcd,abcdef->abcdef", weights_pdf, 1.0 - weights_quench_bin
    )

    # w_sum_MS = jnp.einsum("abcdef->abcef", weights_pdf_MS)
    # w_sum_Q = jnp.einsum("abcdef->abcef", weights_pdf_Q)
    # w_sum_MS_norm = jnp.where(w_sum_MS > 0.0, 1.0 / w_sum_MS, 1.0)
    # w_sum_Q_norm = jnp.where(w_sum_Q > 0.0, 1.0 / w_sum_Q, 1.0)

    # weights_pdf_MS = jnp.einsum("abcdef,abcef->abcdef", weights_pdf_MS, w_sum_MS_norm)
    # weights_pdf_Q = jnp.einsum("abcdef,abcef->abcdef", weights_pdf_Q, w_sum_Q_norm)

    NHALO_MS_p50 = jnp.einsum("abcdef->abf", weights_pdf_MS)
    NHALO_Q_p50 = jnp.einsum("abcdef->abf", weights_pdf_Q)
    quench_frac_p50 = NHALO_Q_p50 / (NHALO_Q_p50 + NHALO_MS_p50)

    NHALO_MS = jnp.sum(NHALO_MS_p50, axis=1)
    NHALO_Q = jnp.sum(NHALO_Q_p50, axis=1)
    quench_frac = NHALO_Q / (NHALO_Q + NHALO_MS)

    NHALO_MS_p50 = jnp.where(NHALO_MS_p50 > 0.0, NHALO_MS_p50, 1.0)
    NHALO_Q_p50 = jnp.where(NHALO_Q_p50 > 0.0, NHALO_Q_p50, 1.0)
    NHALO_MS = jnp.where(NHALO_MS > 0.0, NHALO_MS, 1.0)
    NHALO_Q = jnp.where(NHALO_Q > 0.0, NHALO_Q, 1.0)

    _mean_sm_tmp = jnp.einsum("abcd,abcdef->abf", weights_pdf, mstar_histories)
    mean_sm_p50 = _mean_sm_tmp / (NHALO * NP2)
    mean_sm = jnp.sum(_mean_sm_tmp, axis=1) / (NHALO * NP * NP2)

    _mean_fstar_MS_tmp = jnp.einsum("abcd,abcdef->abf", weights_pdf, fstar_MS)
    _mean_fstar_Q_tmp = jnp.einsum("abcd,abcdef->abf", weights_pdf, fstar_Q)

    mean_fstar_MS = jnp.sum(_mean_fstar_MS_tmp, axis=1) / NHALO_MS
    mean_fstar_Q = jnp.sum(_mean_fstar_Q_tmp, axis=1) / NHALO_Q
    # mean_fstar_MS_p50 = _mean_fstar_MS_tmp / NHALO_MS_p50
    # mean_fstar_Q_p50 = _mean_fstar_Q_tmp / NHALO_Q_p50

    delta_sm = (mstar_histories - mean_sm[:, None, None, None, None, :]) ** 2
    delta_sm_p50 = (mstar_histories - mean_sm_p50[:, :, None, None, None, :]) ** 2
    delta_fstar_MS = (fstar_MS - mean_fstar_MS[:, None, None, None, None, :]) ** 2
    delta_fstar_Q = (fstar_Q - mean_fstar_Q[:, None, None, None, None, :]) ** 2

    delta_fstar_MS = delta_fstar_MS * weights_quench_bin
    delta_fstar_Q = delta_fstar_Q * (1.0 - weights_quench_bin)

    variance_sm = jnp.einsum("abcd,abcdef->af", weights_pdf, delta_sm) / (
        NHALO * NP * NP2
    )
    variance_sm_p50 = jnp.einsum("abcd,abcdef->abf", weights_pdf, delta_sm_p50) / (
        NHALO * NP2
    )
    variance_fstar_MS = (
        jnp.einsum("abcd,abcdef->af", weights_pdf, delta_fstar_MS) / NHALO_MS
    )
    variance_fstar_Q = (
        jnp.einsum("abcd,abcdef->af", weights_pdf, delta_fstar_Q) / NHALO_Q
    )

    _out = (
        mean_sm,
        variance_sm,
        mean_fstar_MS,
        mean_fstar_Q,
        variance_fstar_MS,
        variance_fstar_Q,
        quench_frac,
        mean_sm_p50,
        variance_sm_p50,
        quench_frac_p50,
    )
    return _out
