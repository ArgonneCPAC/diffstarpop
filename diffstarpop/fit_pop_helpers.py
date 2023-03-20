import numpy as np
from jax import numpy as jnp, jit as jjit, vmap, random as jran, grad
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

from .pdf_diffmah import get_diffmah_grid, get_binned_halo_sample
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
from .pdfmodel import get_sfh_param_grid_Q
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
from jax.scipy.stats import multivariate_normal as jnorm


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
    Compute differentiable summary statistics from pdf-weighting the histories
    computed in a latin hypercube grid (or other uniform volume schemes).

    Parameters
    ----------
    weights : ndarray of shape (n_m0, n_sfh_grid)
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
    # mstar_histories = jnp.log10(mstar_histories)
    # sfr_histories = jnp.log10(sfr_histories)
    # fstar_histories = jnp.log10(fstar_histories)

    fstar_MS = fstar_histories * weights_quench_bin
    fstar_Q = fstar_histories * (1.0 - weights_quench_bin)
    # fstar_MS = jnp.where(fstar_MS > 0.0, jnp.log10(fstar_MS), 0.0)
    # fstar_Q = jnp.where(fstar_Q > 0.0, jnp.log10(fstar_Q), 0.0)

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
def compute_target_sumstats_from_histories_diffMC(
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
    weights_SM = jnp.einsum(
        "ab,abcd->abcd", weights_pdf, jnp.ones_like(mstar_histories)
    )
    weights_Q = jnp.einsum("ab,abcd->abcd", weights_pdf, weights_quench_bin)
    weights_MS = jnp.einsum("ab,abcd->abcd", weights_pdf, 1.0 - weights_quench_bin)

    mstar_histories = jnp.log10(mstar_histories)
    fstar_histories = jnp.where(fstar_histories > 0.0, jnp.log10(fstar_histories), 0.0)

    mean_sm = jnp.average(mstar_histories, weights=weights_SM, axis=(1, 2))
    mean_fstar_MS = jnp.average(fstar_histories, weights=weights_MS, axis=(1, 2))
    mean_fstar_Q = jnp.average(fstar_histories, weights=weights_Q, axis=(1, 2))

    variance_sm = jnp.average(
        (mstar_histories - mean_sm[:, None, None, :]) ** 2,
        weights=weights_SM,
        axis=(1, 2),
    )
    variance_fstar_MS = jnp.average(
        (fstar_histories - mean_fstar_MS[:, None, None, :]) ** 2,
        weights=weights_MS,
        axis=(1, 2),
    )
    variance_fstar_Q = jnp.average(
        (fstar_histories - mean_fstar_Q[:, None, None, :]) ** 2,
        weights=weights_Q,
        axis=(1, 2),
    )

    NHALO_MS = jnp.einsum("ab,abcd->ad", weights_pdf, weights_quench_bin)
    NHALO_Q = jnp.einsum("ab,abcd->ad", weights_pdf, 1.0 - weights_quench_bin)
    quench_frac = NHALO_Q / (NHALO_Q + NHALO_MS)

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

    fracs, means, covs = get_smah_means_and_covs_quench(
        logm0_binmids, **pdf_model_params
    )
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

    mstar_histories = jnp.where(mstar_histories > 0.0, jnp.log10(mstar_histories), 0.0)
    sfr_histories = jnp.where(sfr_histories > 0.0, jnp.log10(sfr_histories), 0.0)
    fstar_histories = jnp.where(fstar_histories > 0.0, jnp.log10(fstar_histories), 0.0)

    return compute_target_sumstats_from_histories(
        weights, weights_quench_bin, mstar_histories, sfr_histories, fstar_histories
    )


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

    # return compute_target_sumstats_from_histories_diffMC(
    #     weights, weights_quench_bin, mstar_histories, sfr_histories, fstar_histories
    # )


def loss(params, loss_data):
    (
        t_table,
        lgt_table,
        dt_table,
        index_select,
        index_high,
        fstar_tdelay,
        diffmah_params_grid,
        sfh_param_grids,
        logm0_binmids,
        mstar_histories,
        sfr_histories,
        fstar_histories,
        weights_quench_bin,
        target_data,
        target_data_weights,
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
    (
        mean_sm_weights,
        variance_sm_weights,
        mean_fstar_MS_weights,
        mean_fstar_Q_weights,
        variance_fstar_MS_weights,
        variance_fstar_Q_weights,
        quench_frac_weights,
    ) = target_data_weights

    fracs, means, covs = get_smah_means_and_covs_quench(logm0_binmids, *params)

    weights = _get_pdf_weights_kern(sfh_param_grids, means, covs)

    _res = compute_target_sumstats_from_histories(
        weights, weights_quench_bin, mstar_histories, sfr_histories, fstar_histories
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
    return loss


loss_deriv = jjit(grad(loss, argnums=(0)))


def loss_deriv_np(params, data):
    return np.array(loss_deriv(params, data)).astype(float)


def get_loss_data(
    t_table,
    sfh_lh_sig,
    n_sfh_param_grid,
    logm0_binmids,
    logm0_bin_widths,
    n_halos_per_bin,
    halo_data,
    fstar_tdelay,
    initial_guess_dict,
    target_data,
    target_data_weights,
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
        initial_guess_dict, logm0_binmids, sfh_lh_sig, n_sfh_param_grid
    )

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

    initial_guess = list(initial_guess_dict.values())

    loss_data = (
        t_table,
        lgt_table,
        dt_table,
        index_select,
        index_high,
        fstar_tdelay,
        diffmah_params_grid,
        sfh_param_grids,
        logm0_binmids,
        mstar_histories,
        sfr_histories,
        fstar_histories,
        weights_quench_bin,
        target_data,
        target_data_weights,
    )

    return initial_guess, loss_data


def loss_diffMC(params, loss_data):
    (
        t_table,
        lgt_table,
        dt_table,
        index_select,
        index_high,
        fstar_tdelay,
        diffmah_params_grid,
        sfh_param_grids,
        logm0_binmids,
        mstar_histories,
        sfr_histories,
        fstar_histories,
        weights_quench_bin,
        volumes,
        target_data,
        target_data_weights,
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
    (
        mean_sm_weights,
        variance_sm_weights,
        mean_fstar_MS_weights,
        mean_fstar_Q_weights,
        variance_fstar_MS_weights,
        variance_fstar_Q_weights,
        quench_frac_weights,
    ) = target_data_weights

    fracs, means, covs = get_smah_means_and_covs_quench(logm0_binmids, *params)

    weights = _get_pdf_weights_kern(sfh_param_grids, means, covs)

    weights = weights * volumes

    _res = compute_target_sumstats_from_histories(
        weights, weights_quench_bin, mstar_histories, sfr_histories, fstar_histories
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
    return loss


loss_diffMC_deriv = jjit(grad(loss_diffMC, argnums=(0)))


def loss_diffMC_deriv_np(params, data):
    return np.array(loss_diffMC_deriv(params, data)).astype(float)


def get_loss_data_diffMC(
    t_table,
    sfh_lh_sig,
    n_sfh_param_grid,
    logm0_binmids,
    logm0_bin_widths,
    n_halos_per_bin,
    halo_data,
    fstar_tdelay,
    initial_guess_dict,
    target_data,
    target_data_weights,
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
        logm0_binmids, **initial_guess_dict
    )
    _res = latin_hypercube_quantities(num_dim, n_sfh_param_grid, sfh_lh_sig)
    unit_gaussian_box_samples, lower_lim_pixel, upper_lim_pixel = _res
    _res = calculate_samples_and_volume(
        unit_gaussian_box_samples, means, covs, lower_lim_pixel, upper_lim_pixel
    )
    sfh_param_grids, volumes = _res
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

    initial_guess = list(initial_guess_dict.values())

    loss_data = (
        t_table,
        lgt_table,
        dt_table,
        index_select,
        index_high,
        fstar_tdelay,
        diffmah_params_grid,
        sfh_param_grids,
        logm0_binmids,
        mstar_histories,
        sfr_histories,
        fstar_histories,
        weights_quench_bin,
        volumes,
        target_data,
        target_data_weights,
    )

    return initial_guess, loss_data
