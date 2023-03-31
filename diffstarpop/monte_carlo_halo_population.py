import numpy as np
from numpy.random import RandomState
from jax import vmap
from jax import jit as jjit
from jax import numpy as jnp
from collections import OrderedDict

from diffstar.stars import (
    calculate_sm_sfr_fstar_history_from_mah,
    DEFAULT_SFR_PARAMS as DEFAULT_SFR_PARAMS_DICT,
    _get_unbounded_sfr_params,
    _integrate_sfr,
    compute_fstar,
    fstar_tools,
)
from diffstar.quenching import (
    DEFAULT_Q_PARAMS as DEFAULT_Q_PARAMS_DICT,
    _get_unbounded_q_params,
    quenching_function,
)
from diffstar.main_sequence import get_ms_sfh_from_mah_kern
from diffstar.utils import jax_np_interp, _get_dt_array

from diffmah.individual_halo_assembly import _calc_halo_history

from .pdf_quenched import get_smah_means_and_covs_quench, DEFAULT_SFH_PDF_QUENCH_PARAMS
from .pdf_mainseq import get_smah_means_and_covs_mainseq, DEFAULT_SFH_PDF_MAINSEQ_PARAMS
from .pdf_model_assembly_bias_shifts import (
    DEFAULT_R_QUENCH_PARAMS,
    DEFAULT_R_MAINSEQ_PARAMS,
    _get_slopes_quench,
    _get_slopes_mainseq,
    _get_shift_to_PDF_mean,
)

# from .pdf_model_assembly_bias_shifts import _get_shift_to_PDF_mean, _get_slopes

sfh_scan_tobs_kern = get_ms_sfh_from_mah_kern(tobs_loop="scan")


jax_np_interp_vmap = jjit(vmap(jax_np_interp, in_axes=(0, 0, None, 0)))


def return_searchsorted_like_results(history, fraction):
    _tmp = history - history[:, [-1]] * fraction
    _tmp[_tmp < 0.0] = np.inf
    return np.argmin(_tmp, axis=1)


DEFAULT_UNBOUND_SFR_PARAMS = _get_unbounded_sfr_params(
    *tuple(DEFAULT_SFR_PARAMS_DICT.values())
)
DEFAULT_UNBOUND_SFR_PARAMS_DICT = OrderedDict(
    zip(DEFAULT_SFR_PARAMS_DICT.keys(), DEFAULT_UNBOUND_SFR_PARAMS)
)
DEFAULT_UNBOUND_Q_PARAMS = np.array(
    _get_unbounded_q_params(*tuple(DEFAULT_Q_PARAMS_DICT.values()))
)
UH = DEFAULT_UNBOUND_SFR_PARAMS_DICT["indx_hi"]

DEFAULT_UNBOUND_Q_PARAMS_MAIN_SEQ = DEFAULT_UNBOUND_Q_PARAMS.copy()
DEFAULT_UNBOUND_Q_PARAMS_MAIN_SEQ[0] = 1.9

SFH_PDF_Q_KEYS = list(DEFAULT_SFH_PDF_QUENCH_PARAMS.keys())
SFH_PDF_Q_VALUES = list(DEFAULT_SFH_PDF_QUENCH_PARAMS.values())

SFH_PDF_MS_KEYS = list(DEFAULT_SFH_PDF_MAINSEQ_PARAMS.keys())
SFH_PDF_MS_VALUES = list(DEFAULT_SFH_PDF_MAINSEQ_PARAMS.values())


def _calculate_sm(
    lgt, dt, mah_params, sfr_params, q_params, index_select, index_high, fstar_tdelay
):
    dmhdt, log_mah = _calc_halo_history(lgt, *mah_params)
    mstar, sfr, fstar = calculate_sm_sfr_fstar_history_from_mah(
        lgt,
        dt,
        dmhdt,
        log_mah,
        sfr_params,
        q_params,
        index_select,
        index_high,
        fstar_tdelay,
    )
    return mstar, sfr, fstar


_calc_halo_history_vmap = jjit(vmap(_calc_halo_history, in_axes=(None, *[0] * 6)))

calculate_sm = jjit(vmap(_calculate_sm, in_axes=(*[None] * 2, *[0] * 3, *[None] * 3)))


@jjit
def sm_sfr_history_diffstar_scan(
    tarr,
    lgt,
    dt,
    mah_params,
    sfr_ms_params,
    q_params,
    index_select,
    index_high,
    fstar_tdelay,
):
    ms_sfr = sfh_scan_tobs_kern(tarr, mah_params, sfr_ms_params)
    qfrac = quenching_function(lgt, *q_params)
    sfr = qfrac * ms_sfr
    mstar = _integrate_sfr(sfr, dt)
    fstar = compute_fstar(10**lgt, mstar, index_select, index_high, fstar_tdelay)
    return mstar, sfr, fstar


_A = (*[None] * 3, *[0] * 3, *[None] * 3)
sm_sfr_history_diffstar_scan_vmap = jjit(vmap(sm_sfr_history_diffstar_scan, in_axes=_A))
# _A = (*[None] * 3, *[0] * 2, *[None] * 3)
# sm_sfr_history_diffstar_scan_MS_vmap = jjit(vmap(sm_sfr_history_diffstar_scan_MS, in_axes=_A))


def calculate_sm_vmap_batch(
    lgt, dt, mah_params, sfr_params, q_params, index_select, index_high, fstar_tdelay
):
    ng = len(mah_params)
    nt = len(lgt)
    nt2 = len(index_high)
    mstar = np.zeros((ng, nt))
    sfr = np.zeros((ng, nt))
    fstar = np.zeros((ng, nt2))

    indices = np.array_split(np.arange(ng), max(int(ng / 5000), 1))

    for inds in indices:
        _res = calculate_sm(
            lgt,
            dt,
            mah_params[inds],
            sfr_params[inds],
            q_params[inds],
            index_select,
            index_high,
            fstar_tdelay,
        )
        mstar[inds] = _res[0]
        sfr[inds] = _res[1]
        fstar[inds] = _res[2]
    return mstar, sfr, fstar


def calculate_sm_scan_batch(
    lgt, dt, mah_params, sfr_params, q_params, index_select, index_high, fstar_tdelay
):
    ng = len(mah_params)
    nt = len(lgt)
    nt2 = len(index_high)
    mstar = np.zeros((ng, nt))
    sfr = np.zeros((ng, nt))
    fstar = np.zeros((ng, nt2))

    tarr = 10**lgt

    indices = np.array_split(np.arange(ng), max(int(ng / 5000), 1))

    for inds in indices:
        _res = sm_sfr_history_diffstar_scan_vmap(
            tarr,
            lgt,
            dt,
            mah_params[inds],
            sfr_params[inds],
            q_params[inds],
            index_select,
            index_high,
            fstar_tdelay,
        )
        mstar[inds] = _res[0]
        sfr[inds] = _res[1]
        fstar[inds] = _res[2]
    return mstar, sfr, fstar


def mc_sfh_histories_MIX(
    lgt,
    dt,
    logmh,
    mah_params,
    n_histories,
    index_select,
    index_high,
    fstar_tdelay,
    seed=0,
    pdf_parameters_MS=DEFAULT_SFH_PDF_MAINSEQ_PARAMS,
    pdf_parameters_Q=DEFAULT_SFH_PDF_QUENCH_PARAMS,
    diffstar_kernel="vmap",
):
    """
    Generate Monte Carlo realization of the star formation histories of
    a mixed population of main sequence and quenched galaxies
    for a single halo mass bin.

    Parameters
    ----------
    lgt : ndarray of shape (n_times, )
        Array of log10 cosmic times in units of Gyr
    dt : ndarray of shape (n_times, )
        Time step sizes in units of Gyr
    logmh : float
        Base-10 log of present-day halo mass of the halo population
    mah_params : ndarray, size (n_mah_haloes x n_mah_params)
        Array with the diffmah parameters that will be marginalized over. Could
        be either individual fits of n_mah_haloes haloes, or be n_mah_haloes
        samples from a population model. They are chosen at random n_halos times.
    mah_params : ndarray of shape (n_halos, 6)
        Array containing the following Diffmah parameters
        (logt0, logm0, log10tauc, k, early, late):
            logt0 : ndarray of shape (n_halos, )
                Base-10 log of present-day cosmic time.
            logmp : ndarray of shape (n_halos, )
                Base-10 log of present-day peak halo mass in units of Msun assuming h=1
            log10tauc : ndarray of shape (n_halos, )
                Base-10 log of transition time between the fast- and slow-accretion regimes in Gyr
            k : ndarray of shape (n_halos, )
                Transition speed.
            early : ndarray of shape (n_halos, )
                Early-time power-law index in the scaling relation M(t)~t^a
            late : ndarray of shape (n_halos, )
                Late-time power-law index in the scaling relation M(t)~t^a
    n_histories : int
        Number of SFH histories to generate by DiffstarPop.
    index_select: ndarray of shape (n_times_fstar, )
        Snapshot indices used in fstar computation
    index_high: ndarray of shape (n_times_fstar, )
        Indices of np.searchsorted(t, t - fstar_tdelay)[index_select]
    fstar_tdelay: float
        Time interval in Gyr for fstar definition.
        fstar = (mstar(t) - mstar(t-fstar_tdelay)) / fstar_tdelay[Gyr]
    seed : int, optional
        Random number seed
    pdf_model_params_MS : dict
        Dictionary containing the Diffstarpop parameters for the main sequence population.
        Default is DEFAULT_SFH_PDF_MAINSEQ_PARAMS.
    pdf_model_params_Q : dict
        Dictionary containing the Diffstarpop parameters for the main sequence population.
        Default is DEFAULT_SFH_PDF_QUENCH_PARAMS.
    diffstar_kernel: string
        Type of diffstar kernel implementation. Options are 'vmap', 'scan'.

    Returns
    -------
    mstar : ndarray of shape (n_histories, n_times)
        Stores cumulative stellar mass history in units of Msun/yr.
    sfr : ndarray of shape (n_histories, n_times)
        Stores star formation rate history in units of Msun/yr.
    fstar : ndarray of shape (n_histories, n_times_fstar)
        SFH averaged over timescale fstar_tdelay in units of Msun/yr assuming h=1.
    """

    logmh = np.atleast_1d(logmh).astype("f4")
    assert logmh.size == 1, "Input halo mass must be a scalar"

    n_mah = len(mah_params)

    sampled_mahs_inds = RandomState(seed).choice(n_mah, n_histories, replace=True)
    mah_params_sampled = mah_params[sampled_mahs_inds]

    _res = get_smah_means_and_covs_quench(logmh, **pdf_parameters_Q)
    frac_quench, means_quench, covs_quench = _res
    frac_quench = frac_quench[0]
    means_quench = means_quench[0]
    covs_quench = covs_quench[0]

    n_histories_Q = int(n_histories * frac_quench)
    n_histories_MS = n_histories - n_histories_Q
    # n_histories_Q = int(n_histories / 2.0)
    # n_histories_MS = n_histories - n_histories_Q

    sfr_params_quench = np.zeros((n_histories_Q, 5))
    sfr_params_mainseq = np.zeros((n_histories_MS, 5))
    q_params_mainseq = np.zeros((n_histories_MS, 4))

    if n_histories_Q > 0:
        sfh_params_quench = RandomState(seed + 1).multivariate_normal(
            means_quench, covs_quench, size=n_histories_Q
        )

        sfr_params_quench[:, :3] = sfh_params_quench[:, :3]
        sfr_params_quench[:, 3] = UH
        sfr_params_quench[:, 4] = sfh_params_quench[:, 3]
        q_params_quench = sfh_params_quench[:, 4:8]

    if n_histories_MS > 0:
        _res = get_smah_means_and_covs_mainseq(logmh, **pdf_parameters_MS)
        means_mainseq, covs_mainseq = _res
        means_mainseq = means_mainseq[0]
        covs_mainseq = covs_mainseq[0]

        sfh_params_mainseq = RandomState(seed + 1).multivariate_normal(
            means_mainseq, covs_mainseq, size=n_histories_MS
        )

        sfr_params_mainseq[:, :3] = sfh_params_mainseq[:, :3]
        sfr_params_mainseq[:, 3] = UH
        sfr_params_mainseq[:, 4] = sfh_params_mainseq[:, 3]
        q_params_mainseq[:, np.arange(4)] = DEFAULT_UNBOUND_Q_PARAMS_MAIN_SEQ

    sfr_params = np.concatenate((sfr_params_mainseq, sfr_params_quench))
    q_params = np.concatenate((q_params_mainseq, q_params_quench))

    if diffstar_kernel == "vmap":
        _res = calculate_sm_vmap_batch(
            lgt,
            dt,
            mah_params_sampled,
            sfr_params,
            q_params,
            index_select,
            index_high,
            fstar_tdelay,
        )
    elif diffstar_kernel == "scan":
        _res = calculate_sm_scan_batch(
            lgt,
            dt,
            mah_params_sampled[:, [1, 2, 4, 5]],
            sfr_params,
            q_params,
            index_select,
            index_high,
            fstar_tdelay,
        )
    mstar = _res[0]
    sfr = _res[1]
    fstar = _res[2]

    weights = np.concatenate(
        (
            np.ones(n_histories_MS) * (1 - frac_quench),
            np.ones(n_histories_Q) * frac_quench,
        )
    )

    return mstar, sfr, fstar, weights


def mc_sfh_histories_Q(
    lgt,
    dt,
    logmh,
    mah_params,
    n_histories,
    index_select,
    index_high,
    fstar_tdelay,
    seed=0,
    pdf_parameters=DEFAULT_SFH_PDF_QUENCH_PARAMS,
    diffstar_kernel="vmap",
):
    """
    Generate Monte Carlo realization of the star formation histories of
    quenched galaxies for a single halo mass bin.

    Parameters
    ----------
    lgt : ndarray of shape (n_times, )
        Array of log10 cosmic times in units of Gyr
    dt : ndarray of shape (n_times, )
        Time step sizes in units of Gyr
    logmh : float
        Base-10 log of present-day halo mass of the halo population
    mah_params : ndarray, size (n_mah_haloes x n_mah_params)
        Array with the diffmah parameters that will be marginalized over. Could
        be either individual fits of n_mah_haloes haloes, or be n_mah_haloes
        samples from a population model. They are chosen at random n_halos times.
    mah_params : ndarray of shape (n_halos, 6)
        Array containing the following Diffmah parameters
        (logt0, logm0, log10tauc, k, early, late):
            logt0 : ndarray of shape (n_halos, )
                Base-10 log of present-day cosmic time.
            logmp : ndarray of shape (n_halos, )
                Base-10 log of present-day peak halo mass in units of Msun assuming h=1
            log10tauc : ndarray of shape (n_halos, )
                Base-10 log of transition time between the fast- and slow-accretion regimes in Gyr
            k : ndarray of shape (n_halos, )
                Transition speed.
            early : ndarray of shape (n_halos, )
                Early-time power-law index in the scaling relation M(t)~t^a
            late : ndarray of shape (n_halos, )
                Late-time power-law index in the scaling relation M(t)~t^a
    n_histories : int
        Number of SFH histories to generate by DiffstarPop.
    index_select: ndarray of shape (n_times_fstar, )
        Snapshot indices used in fstar computation
    index_high: ndarray of shape (n_times_fstar, )
        Indices of np.searchsorted(t, t - fstar_tdelay)[index_select]
    fstar_tdelay: float
        Time interval in Gyr for fstar definition.
        fstar = (mstar(t) - mstar(t-fstar_tdelay)) / fstar_tdelay[Gyr]
    seed : int, optional
        Random number seed
    pdf_model_params : dict
        Dictionary containing the Diffstarpop parameters for the main sequence population.
        Default is DEFAULT_SFH_PDF_QUENCH_PARAMS.
    diffstar_kernel: string
        Type of diffstar kernel implementation. Options are 'vmap', 'scan'.

    Returns
    -------
    mstar : ndarray of shape (n_histories, n_times)
        Stores cumulative stellar mass history in units of Msun/yr.
    sfr : ndarray of shape (n_histories, n_times)
        Stores star formation rate history in units of Msun/yr.
    fstar : ndarray of shape (n_histories, n_times_fstar)
        SFH averaged over timescale fstar_tdelay in units of Msun/yr assuming h=1.
    """

    logmh = np.atleast_1d(logmh).astype("f4")
    assert logmh.size == 1, "Input halo mass must be a scalar"

    n_mah = len(mah_params)

    sampled_mahs_inds = RandomState(seed).choice(n_mah, n_histories, replace=True)
    mah_params_sampled = mah_params[sampled_mahs_inds]

    _res = get_smah_means_and_covs_quench(logmh, **pdf_parameters)
    frac_quench, means_quench, covs_quench = _res
    frac_quench = frac_quench[0]
    means_quench = means_quench[0]
    covs_quench = covs_quench[0]

    sfh_params = RandomState(seed + 1).multivariate_normal(
        means_quench, covs_quench, size=n_histories
    )
    sfr_params = np.zeros((n_histories, 5))
    sfr_params[:, :3] = sfh_params[:, :3]
    sfr_params[:, 3] = UH
    sfr_params[:, 4] = sfh_params[:, 3]
    q_params = sfh_params[:, 4:8]

    if diffstar_kernel == "vmap":
        _res = calculate_sm_vmap_batch(
            lgt,
            dt,
            mah_params_sampled,
            sfr_params,
            q_params,
            index_select,
            index_high,
            fstar_tdelay,
        )
    elif diffstar_kernel == "scan":
        _res = calculate_sm_scan_batch(
            lgt,
            dt,
            mah_params_sampled[:, [1, 2, 4, 5]],
            sfr_params,
            q_params,
            index_select,
            index_high,
            fstar_tdelay,
        )

    mstar = _res[0]
    sfr = _res[1]
    fstar = _res[2]

    return mstar, sfr, fstar


def mc_sfh_histories_MS(
    lgt,
    dt,
    logmh,
    mah_params,
    n_histories,
    index_select,
    index_high,
    fstar_tdelay,
    seed=0,
    pdf_parameters=DEFAULT_SFH_PDF_MAINSEQ_PARAMS,
    diffstar_kernel="vmap",
):
    """
    Generate Monte Carlo realization of the star formation histories of
    main sequence galaxies for a single halo mass bin.

    Parameters
    ----------
    lgt : ndarray of shape (n_times, )
        Array of log10 cosmic times in units of Gyr
    dt : ndarray of shape (n_times, )
        Time step sizes in units of Gyr
    logmh : float
        Base-10 log of present-day halo mass of the halo population
    mah_params : ndarray, size (n_mah_haloes x n_mah_params)
        Array with the diffmah parameters that will be marginalized over. Could
        be either individual fits of n_mah_haloes haloes, or be n_mah_haloes
        samples from a population model. They are chosen at random n_halos times.
    mah_params : ndarray of shape (n_halos, 6)
        Array containing the following Diffmah parameters
        (logt0, logm0, log10tauc, k, early, late):
            logt0 : ndarray of shape (n_halos, )
                Base-10 log of present-day cosmic time.
            logmp : ndarray of shape (n_halos, )
                Base-10 log of present-day peak halo mass in units of Msun assuming h=1
            log10tauc : ndarray of shape (n_halos, )
                Base-10 log of transition time between the fast- and slow-accretion regimes in Gyr
            k : ndarray of shape (n_halos, )
                Transition speed.
            early : ndarray of shape (n_halos, )
                Early-time power-law index in the scaling relation M(t)~t^a
            late : ndarray of shape (n_halos, )
                Late-time power-law index in the scaling relation M(t)~t^a
    n_histories : int
        Number of SFH histories to generate by DiffstarPop.
    index_select: ndarray of shape (n_times_fstar, )
        Snapshot indices used in fstar computation
    index_high: ndarray of shape (n_times_fstar, )
        Indices of np.searchsorted(t, t - fstar_tdelay)[index_select]
    fstar_tdelay: float
        Time interval in Gyr for fstar definition.
        fstar = (mstar(t) - mstar(t-fstar_tdelay)) / fstar_tdelay[Gyr]
    seed : int, optional
        Random number seed
    pdf_model_params : dict
        Dictionary containing the Diffstarpop parameters for the main sequence population.
        Default is DEFAULT_SFH_PDF_MAINSEQ_PARAMS.
    diffstar_kernel: string
        Type of diffstar kernel implementation. Options are 'vmap', 'scan'.

    Returns
    -------
    mstar : ndarray of shape (n_histories, n_times)
        Stores cumulative stellar mass history in units of Msun/yr.
    sfr : ndarray of shape (n_histories, n_times)
        Stores star formation rate history in units of Msun/yr.
    fstar : ndarray of shape (n_histories, n_times_fstar)
        SFH averaged over timescale fstar_tdelay in units of Msun/yr assuming h=1.
    """

    logmh = np.atleast_1d(logmh).astype("f4")
    assert logmh.size == 1, "Input halo mass must be a scalar"

    n_mah = len(mah_params)

    sampled_mahs_inds = RandomState(seed).choice(n_mah, n_histories, replace=True)
    mah_params_sampled = mah_params[sampled_mahs_inds]

    _res = get_smah_means_and_covs_mainseq(logmh, **pdf_parameters)
    means_mainseq, covs_mainseq = _res
    means_mainseq = means_mainseq[0]
    covs_mainseq = covs_mainseq[0]

    sfh_params = RandomState(seed + 1).multivariate_normal(
        means_mainseq, covs_mainseq, size=n_histories
    )
    sfr_params = np.zeros((n_histories, 5))
    sfr_params[:, :3] = sfh_params[:, :3]
    sfr_params[:, 3] = UH
    sfr_params[:, 4] = sfh_params[:, 3]
    q_params = np.zeros((n_histories, 4))
    q_params[:, np.arange(4)] = DEFAULT_UNBOUND_Q_PARAMS_MAIN_SEQ

    if diffstar_kernel == "vmap":
        _res = calculate_sm_vmap_batch(
            lgt,
            dt,
            mah_params_sampled,
            sfr_params,
            q_params,
            index_select,
            index_high,
            fstar_tdelay,
        )
    elif diffstar_kernel == "scan":
        _res = calculate_sm_scan_batch(
            lgt,
            dt,
            mah_params_sampled[:, [1, 2, 4, 5]],
            sfr_params,
            q_params,
            index_select,
            index_high,
            fstar_tdelay,
        )
    mstar = _res[0]
    sfr = _res[1]
    fstar = _res[2]

    return mstar, sfr, fstar


def mc_sfh_population_draw(
    t_table,
    logm0_binmids,
    logm0_bin_widths,
    n_halos_per_bin,
    mah_params,
    n_histories,
    fstar_tdelay=1.0,
    seed=0,
    population="MIX",
    pdf_parameters_MS=DEFAULT_SFH_PDF_MAINSEQ_PARAMS,
    pdf_parameters_Q=DEFAULT_SFH_PDF_QUENCH_PARAMS,
    diffstar_kernel="vmap",
):
    """
    Wrapper function to generate Monte Carlo realization of the star formation
    histories of galaxy population for multiple halo mass bin.

    Parameters
    ----------
    t_table : ndarray of shape (n_t, )
        Cosmic time array in Gyr.
    logm0_binmids : ndarray of shape (n_m0, )
        Midpoint of the logarithmic halo mass bins
    logm0_bin_widths : ndarray of shape (n_m0, )
        Logarithmic width of the halo mass bin
    n_halos_per_bin : int
        Number of halos to be randomly sub-selected in each halo mass bin.
    mah_params : ndarray, size (n_mah_haloes x n_mah_params)
        Array with the diffmah parameters that will be marginalized over. Could
        be either individual fits of n_mah_haloes haloes, or be n_mah_haloes
        samples from a population model. They are chosen at random n_halos times.
    mah_params : ndarray of shape (n_halos, 6)
        Array containing the following Diffmah parameters
        (logt0, logm0, log10tauc, k, early, late):
            logt0 : ndarray of shape (n_halos, )
                Base-10 log of present-day cosmic time.
            logmp : ndarray of shape (n_halos, )
                Base-10 log of present-day peak halo mass in units of Msun assuming h=1
            log10tauc : ndarray of shape (n_halos, )
                Base-10 log of transition time between the fast- and slow-accretion regimes in Gyr
            k : ndarray of shape (n_halos, )
                Transition speed.
            early : ndarray of shape (n_halos, )
                Early-time power-law index in the scaling relation M(t)~t^a
            late : ndarray of shape (n_halos, )
                Late-time power-law index in the scaling relation M(t)~t^a
    n_histories : int
        Number of SFH histories to generate by DiffstarPop.
    fstar_tdelay: float
        Time interval in Gyr for fstar definition.
        fstar = (mstar(t) - mstar(t-fstar_tdelay)) / fstar_tdelay[Gyr]
    seed : int, optional
        Random number seed
    population: string
        Type of DiffstarPop model. Options are 'Q', 'MS', 'MIX'.
    pdf_model_params_MS : dict
        Dictionary containing the Diffstarpop parameters for the main sequence population.
        Default is DEFAULT_SFH_PDF_MAINSEQ_PARAMS.
    pdf_model_params_Q : dict
        Dictionary containing the Diffstarpop parameters for the main sequence population.
        Default is DEFAULT_SFH_PDF_QUENCH_PARAMS.
    diffstar_kernel: string
        Type of diffstar kernel implementation. Options are 'vmap', 'scan'.

    Returns
    -------
    mstar : ndarray of shape (n_m0, n_histories, n_times)
        Stores cumulative stellar mass history in units of Msun/yr.
    sfr : ndarray of shape (n_m0, n_histories, n_times)
        Stores star formation rate history in units of Msun/yr.
    fstar : ndarray of shape (n_m0, n_histories, n_times_fstar)
        SFH averaged over timescale fstar_tdelay in units of Msun/yr assuming h=1.
    """

    index_select, index_high = fstar_tools(t_table, fstar_tdelay)
    lgt = jnp.log10(t_table)
    dt = _get_dt_array(t_table)

    mstar_hists = np.zeros((len(logm0_binmids), n_histories, len(t_table)))
    sfr_hists = np.zeros((len(logm0_binmids), n_histories, len(t_table)))
    fstar_hists = np.zeros((len(logm0_binmids), n_histories, len(index_select)))

    logmpeak = mah_params[:, 1]

    for i in range(len(logm0_binmids)):

        mask = (logmpeak > logm0_binmids[i] - logm0_bin_widths[i]) & (
            logmpeak < logm0_binmids[i] + logm0_bin_widths[i]
        )

        mah_params_bin = mah_params[mask]
        choose = np.random.choice(len(mah_params_bin), n_halos_per_bin, replace=False)
        mah_params_bin = mah_params_bin[choose]

        logm0_bin = np.array([logm0_binmids[i]])

        if population == "Q":
            _res = mc_sfh_histories_Q(
                lgt,
                dt,
                logm0_bin,
                mah_params_bin,
                n_histories,
                index_select,
                index_high,
                fstar_tdelay,
                seed=0,
                pdf_parameters=pdf_parameters_Q,
            )
        elif population == "MS":
            _res = mc_sfh_histories_MS(
                lgt,
                dt,
                logm0_bin,
                mah_params_bin,
                n_histories,
                index_select,
                index_high,
                fstar_tdelay,
                seed=0,
                pdf_parameters=pdf_parameters_MS,
            )
        elif population == "MIX":
            _res = mc_sfh_histories_MIX(
                lgt,
                dt,
                logm0_bin,
                mah_params_bin,
                n_histories,
                index_select,
                index_high,
                fstar_tdelay,
                seed=0,
                pdf_parameters_MS=pdf_parameters_MS,
                pdf_parameters_Q=pdf_parameters_Q,
                diffstar_kernel=diffstar_kernel,
            )
        else:
            raise NotImplementedError("'population' needs to be 'Q', 'MS' or 'MIX'")

        mstar_hists[i] = _res[0]
        sfr_hists[i] = _res[1]
        fstar_hists[i] = _res[2]

    return mstar_hists, sfr_hists, fstar_hists


def mc_sfh_population_sumstats(
    t_table,
    logm0_binmids,
    logm0_bin_widths,
    n_halos_per_bin,
    mah_params,
    n_histories,
    fstar_tdelay=1.0,
    seed=0,
    population="MIX",
    pdf_parameters_MS=DEFAULT_SFH_PDF_MAINSEQ_PARAMS,
    pdf_parameters_Q=DEFAULT_SFH_PDF_QUENCH_PARAMS,
    diffstar_kernel="vmap",
):
    """
    Wrapper function to compute Diffstarpop summary statistic predictions
    from Monte Carlo realization of the star formation histories of galaxy
    population for multiple halo mass bin.

    Parameters
    ----------
    t_table : ndarray of shape (n_t, )
        Cosmic time array in Gyr.
    logm0_binmids : ndarray of shape (n_m0, )
        Midpoint of the logarithmic halo mass bins
    logm0_bin_widths : ndarray of shape (n_m0, )
        Logarithmic width of the halo mass bin
    n_halos_per_bin : int
        Number of halos to be randomly sub-selected in each halo mass bin.
    mah_params : ndarray, size (n_mah_haloes x n_mah_params)
        Array with the diffmah parameters that will be marginalized over. Could
        be either individual fits of n_mah_haloes haloes, or be n_mah_haloes
        samples from a population model. They are chosen at random n_halos times.
    mah_params : ndarray of shape (n_halos, 6)
        Array containing the following Diffmah parameters
        (logt0, logm0, log10tauc, k, early, late):
            logt0 : ndarray of shape (n_halos, )
                Base-10 log of present-day cosmic time.
            logmp : ndarray of shape (n_halos, )
                Base-10 log of present-day peak halo mass in units of Msun assuming h=1
            log10tauc : ndarray of shape (n_halos, )
                Base-10 log of transition time between the fast- and slow-accretion regimes in Gyr
            k : ndarray of shape (n_halos, )
                Transition speed.
            early : ndarray of shape (n_halos, )
                Early-time power-law index in the scaling relation M(t)~t^a
            late : ndarray of shape (n_halos, )
                Late-time power-law index in the scaling relation M(t)~t^a
    n_histories : int
        Number of SFH histories to generate by DiffstarPop.
    fstar_tdelay: float
        Time interval in Gyr for fstar definition.
        fstar = (mstar(t) - mstar(t-fstar_tdelay)) / fstar_tdelay[Gyr]
    seed : int, optional
        Random number seed
    population: string
        Type of DiffstarPop model. Options are 'Q', 'MS', 'MIX'.
    pdf_model_params_MS : dict
        Dictionary containing the Diffstarpop parameters for the main sequence population.
        Default is DEFAULT_SFH_PDF_MAINSEQ_PARAMS.
    pdf_model_params_Q : dict
        Dictionary containing the Diffstarpop parameters for the main sequence population.
        Default is DEFAULT_SFH_PDF_QUENCH_PARAMS.
    diffstar_kernel: string
        Type of diffstar kernel implementation. Options are 'vmap', 'scan'.

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

    index_select, index_high = fstar_tools(t_table, fstar_tdelay)
    lgt = jnp.log10(t_table)
    dt = _get_dt_array(t_table)

    sm_mean_MC = np.zeros((len(logm0_binmids), len(t_table)))
    sm_var_MC = np.zeros((len(logm0_binmids), len(t_table)))
    fstar_mean_MS_MC = np.zeros((len(logm0_binmids), len(index_select)))
    fstar_mean_Q_MC = np.zeros((len(logm0_binmids), len(index_select)))
    fstar_var_MS_MC = np.zeros((len(logm0_binmids), len(index_select)))
    fstar_var_Q_MC = np.zeros((len(logm0_binmids), len(index_select)))
    quench_frac_MC = np.zeros((len(logm0_binmids), len(index_select)))

    logmpeak = mah_params[:, 1]

    for i in range(len(logm0_binmids)):

        mask = (logmpeak > logm0_binmids[i] - logm0_bin_widths[i]) & (
            logmpeak < logm0_binmids[i] + logm0_bin_widths[i]
        )

        mah_params_bin = mah_params[mask]
        choose = np.random.choice(len(mah_params_bin), n_halos_per_bin, replace=False)
        mah_params_bin = mah_params_bin[choose]

        logm0_bin = np.array([logm0_binmids[i]])

        if population == "Q":
            _res = mc_sfh_histories_Q(
                lgt,
                dt,
                logm0_bin,
                mah_params_bin,
                n_histories,
                index_select,
                index_high,
                fstar_tdelay,
                seed=0,
                pdf_parameters=pdf_parameters_Q,
            )
        elif population == "MS":
            _res = mc_sfh_histories_MS(
                lgt,
                dt,
                logm0_bin,
                mah_params_bin,
                n_histories,
                index_select,
                index_high,
                fstar_tdelay,
                seed=0,
                pdf_parameters=pdf_parameters_MS,
            )
        elif population == "MIX":
            _res = mc_sfh_histories_MIX(
                lgt,
                dt,
                logm0_bin,
                mah_params_bin,
                n_histories,
                index_select,
                index_high,
                fstar_tdelay,
                seed=0,
                pdf_parameters_MS=pdf_parameters_MS,
                pdf_parameters_Q=pdf_parameters_Q,
                diffstar_kernel=diffstar_kernel,
            )
        else:
            raise NotImplementedError("'population' needs to be 'Q', 'MS' or 'MIX'")

        mstar_MC, sfr_MC, fstar_MC = _res[:3]

        sFstar = fstar_MC / mstar_MC[:, index_select]

        sm_mean_MC[i] = np.nanmean(np.log10(mstar_MC), axis=0)
        sm_var_MC[i] = np.nanstd(np.log10(mstar_MC), axis=0) ** 2

        fstar_MC_MS = fstar_MC.copy()
        fstar_MC_Q = fstar_MC.copy()
        fstar_MC_MS[(sFstar < 1e-11)] = np.nan
        fstar_MC_Q[(sFstar > 1e-11)] = np.nan
        fstar_mean_MS_MC[i] = np.nanmean(np.log10(fstar_MC_MS), axis=0)
        fstar_mean_Q_MC[i] = np.nanmean(np.log10(fstar_MC_Q), axis=0)
        fstar_var_MS_MC[i] = np.nanstd(np.log10(fstar_MC_MS), axis=0) ** 2
        fstar_var_Q_MC[i] = np.nanstd(np.log10(fstar_MC_Q), axis=0) ** 2
        quench_frac_MC[i] = np.sum(sFstar < 1e-11, axis=0) / (
            np.sum(sFstar < 1e-11, axis=0) + np.sum(sFstar > 1e-11, axis=0)
        )

    _out = (
        sm_mean_MC,
        sm_var_MC,
        fstar_mean_MS_MC,
        fstar_mean_Q_MC,
        fstar_var_MS_MC,
        fstar_var_Q_MC,
        quench_frac_MC,
    )

    return _out


def mc_sfh_histories_withR_Q(
    lgt,
    dt,
    logmh,
    mah_params,
    n_histories,
    index_select,
    index_high,
    fstar_tdelay,
    p50,
    seed=0,
    pdf_parameters=DEFAULT_SFH_PDF_QUENCH_PARAMS,
    R_model_params=DEFAULT_R_QUENCH_PARAMS,
    diffstar_kernel="vmap",
):
    """
    Generate Monte Carlo realization of the star formation histories of
    quenched galaxies for a single halo mass bin.

    There is correlation with p50.

    Parameters
    ----------
    lgt : ndarray of shape (n_times, )
        Array of log10 cosmic times in units of Gyr
    dt : ndarray of shape (n_times, )
        Time step sizes in units of Gyr
    logmh : float
        Base-10 log of present-day halo mass of the halo population
    mah_params : ndarray, size (n_mah_haloes x n_mah_params)
        Array with the diffmah parameters that will be marginalized over. Could
        be either individual fits of n_mah_haloes haloes, or be n_mah_haloes
        samples from a population model. They are chosen at random n_halos times.
    mah_params : ndarray of shape (n_halos, 6)
        Array containing the following Diffmah parameters
        (logt0, logm0, log10tauc, k, early, late):
            logt0 : ndarray of shape (n_halos, )
                Base-10 log of present-day cosmic time.
            logmp : ndarray of shape (n_halos, )
                Base-10 log of present-day peak halo mass in units of Msun assuming h=1
            log10tauc : ndarray of shape (n_halos, )
                Base-10 log of transition time between the fast- and slow-accretion regimes in Gyr
            k : ndarray of shape (n_halos, )
                Transition speed.
            early : ndarray of shape (n_halos, )
                Early-time power-law index in the scaling relation M(t)~t^a
            late : ndarray of shape (n_halos, )
                Late-time power-law index in the scaling relation M(t)~t^a
    n_histories : int
        Number of SFH histories to generate by DiffstarPop.
    index_select: ndarray of shape (n_times_fstar, )
        Snapshot indices used in fstar computation
    index_high: ndarray of shape (n_times_fstar, )
        Indices of np.searchsorted(t, t - fstar_tdelay)[index_select]
    fstar_tdelay: float
        Time interval in Gyr for fstar definition.
        fstar = (mstar(t) - mstar(t-fstar_tdelay)) / fstar_tdelay[Gyr]
    seed : int, optional
        Random number seed
    pdf_model_params : dict
        Dictionary containing the Diffstarpop parameters for the main sequence population.
        Default is DEFAULT_SFH_PDF_QUENCH_PARAMS.
    diffstar_kernel: string
        Type of diffstar kernel implementation. Options are 'vmap', 'scan'.

    Returns
    -------
    mstar : ndarray of shape (n_histories, n_times)
        Stores cumulative stellar mass history in units of Msun/yr.
    sfr : ndarray of shape (n_histories, n_times)
        Stores star formation rate history in units of Msun/yr.
    fstar : ndarray of shape (n_histories, n_times_fstar)
        SFH averaged over timescale fstar_tdelay in units of Msun/yr assuming h=1.
    """

    logmh = np.atleast_1d(logmh).astype("f4")
    assert logmh.size == 1, "Input halo mass must be a scalar"

    n_mah = len(mah_params)

    sampled_mahs_inds = RandomState(seed).choice(n_mah, n_histories, replace=True)
    mah_params_sampled = mah_params[sampled_mahs_inds]
    p50_sampled = p50[sampled_mahs_inds]

    _res = get_smah_means_and_covs_quench(logmh, **pdf_parameters)
    frac_quench, means_quench, covs_quench = _res
    frac_quench = frac_quench[0]
    means_quench = means_quench[0]
    covs_quench = covs_quench[0]

    R_vals_quench = _get_slopes_quench(logmh, **R_model_params)
    R_vals_quench = jnp.array(R_vals_quench)[:, 0]

    shifts_quench = jnp.einsum("p,h->hp", R_vals_quench, (p50_sampled - 0.5))

    sfh_params = RandomState(seed + 1).multivariate_normal(
        means_quench, covs_quench, size=n_histories
    )
    sfh_params = sfh_params + shifts_quench

    sfr_params = np.zeros((n_histories, 5))
    sfr_params[:, :3] = sfh_params[:, :3]
    sfr_params[:, 3] = UH
    sfr_params[:, 4] = sfh_params[:, 3]
    q_params = sfh_params[:, 4:8]

    if diffstar_kernel == "vmap":
        _res = calculate_sm_vmap_batch(
            lgt,
            dt,
            mah_params_sampled,
            sfr_params,
            q_params,
            index_select,
            index_high,
            fstar_tdelay,
        )
    elif diffstar_kernel == "scan":
        _res = calculate_sm_scan_batch(
            lgt,
            dt,
            mah_params_sampled[:, [1, 2, 4, 5]],
            sfr_params,
            q_params,
            index_select,
            index_high,
            fstar_tdelay,
        )

    mstar = _res[0]
    sfr = _res[1]
    fstar = _res[2]

    return mstar, sfr, fstar, p50_sampled


def mc_sfh_population_withR_sumstats(
    t_table,
    logm0_binmids,
    logm0_bin_widths,
    p50_bins,
    n_halos_per_bin,
    mah_params,
    p50,
    n_histories,
    fstar_tdelay=1.0,
    seed=0,
    population="MIX",
    pdf_parameters_MS=DEFAULT_SFH_PDF_MAINSEQ_PARAMS,
    pdf_parameters_Q=DEFAULT_SFH_PDF_QUENCH_PARAMS,
    R_model_params_MS=DEFAULT_R_MAINSEQ_PARAMS,
    R_model_params_Q=DEFAULT_R_QUENCH_PARAMS,
    diffstar_kernel="vmap",
):
    """
    Wrapper function to compute Diffstarpop summary statistic predictions
    from Monte Carlo realization of the star formation histories of galaxy
    population for multiple halo mass bin.

    Parameters
    ----------
    t_table : ndarray of shape (n_t, )
        Cosmic time array in Gyr.
    logm0_binmids : ndarray of shape (n_m0, )
        Midpoint of the logarithmic halo mass bins
    logm0_bin_widths : ndarray of shape (n_m0, )
        Logarithmic width of the halo mass bin
    n_halos_per_bin : int
        Number of halos to be randomly sub-selected in each halo mass bin.
    mah_params : ndarray, size (n_mah_haloes x n_mah_params)
        Array with the diffmah parameters that will be marginalized over. Could
        be either individual fits of n_mah_haloes haloes, or be n_mah_haloes
        samples from a population model. They are chosen at random n_halos times.
    mah_params : ndarray of shape (n_halos, 6)
        Array containing the following Diffmah parameters
        (logt0, logm0, log10tauc, k, early, late):
            logt0 : ndarray of shape (n_halos, )
                Base-10 log of present-day cosmic time.
            logmp : ndarray of shape (n_halos, )
                Base-10 log of present-day peak halo mass in units of Msun assuming h=1
            log10tauc : ndarray of shape (n_halos, )
                Base-10 log of transition time between the fast- and slow-accretion regimes in Gyr
            k : ndarray of shape (n_halos, )
                Transition speed.
            early : ndarray of shape (n_halos, )
                Early-time power-law index in the scaling relation M(t)~t^a
            late : ndarray of shape (n_halos, )
                Late-time power-law index in the scaling relation M(t)~t^a
    n_histories : int
        Number of SFH histories to generate by DiffstarPop.
    fstar_tdelay: float
        Time interval in Gyr for fstar definition.
        fstar = (mstar(t) - mstar(t-fstar_tdelay)) / fstar_tdelay[Gyr]
    seed : int, optional
        Random number seed
    population: string
        Type of DiffstarPop model. Options are 'Q', 'MS', 'MIX'.
    pdf_model_params_MS : dict
        Dictionary containing the Diffstarpop parameters for the main sequence population.
        Default is DEFAULT_SFH_PDF_MAINSEQ_PARAMS.
    pdf_model_params_Q : dict
        Dictionary containing the Diffstarpop parameters for the main sequence population.
        Default is DEFAULT_SFH_PDF_QUENCH_PARAMS.
    diffstar_kernel: string
        Type of diffstar kernel implementation. Options are 'vmap', 'scan'.

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

    index_select, index_high = fstar_tools(t_table, fstar_tdelay)
    lgt = jnp.log10(t_table)
    dt = _get_dt_array(t_table)

    sm_mean_MC = np.zeros((len(logm0_binmids), len(t_table)))
    sm_var_MC = np.zeros((len(logm0_binmids), len(t_table)))
    fstar_mean_MS_MC = np.zeros((len(logm0_binmids), len(index_select)))
    fstar_mean_Q_MC = np.zeros((len(logm0_binmids), len(index_select)))
    fstar_var_MS_MC = np.zeros((len(logm0_binmids), len(index_select)))
    fstar_var_Q_MC = np.zeros((len(logm0_binmids), len(index_select)))
    quench_frac_MC = np.zeros((len(logm0_binmids), len(index_select)))

    sm_mean_MC_p50 = np.zeros((len(logm0_binmids), len(p50_bins) - 1, len(t_table)))
    sm_var_MC_p50 = np.zeros((len(logm0_binmids), len(p50_bins) - 1, len(t_table)))
    quench_frac_MC_p50 = np.zeros(
        (len(logm0_binmids), len(p50_bins) - 1, len(index_select))
    )

    logmpeak = mah_params[:, 1]

    for i in range(len(logm0_binmids)):

        mask = (logmpeak > logm0_binmids[i] - logm0_bin_widths[i]) & (
            logmpeak < logm0_binmids[i] + logm0_bin_widths[i]
        )

        mah_params_bin = mah_params[mask]
        p50_bin = p50[mask]
        choose = np.random.choice(len(mah_params_bin), n_halos_per_bin, replace=False)
        mah_params_bin = mah_params_bin[choose]
        p50_bin = p50_bin[choose]

        logm0_bin = np.array([logm0_binmids[i]])

        if population == "Q":
            _res = mc_sfh_histories_withR_Q(
                lgt,
                dt,
                logm0_bin,
                mah_params_bin,
                n_histories,
                index_select,
                index_high,
                fstar_tdelay,
                p50_bin,
                seed=0,
                pdf_parameters=pdf_parameters_Q,
                R_model_params=R_model_params_Q,
            )
        elif population == "MS":
            _res = mc_sfh_histories_MS(
                lgt,
                dt,
                logm0_bin,
                mah_params_bin,
                n_histories,
                index_select,
                index_high,
                fstar_tdelay,
                seed=0,
                pdf_parameters=pdf_parameters_MS,
            )
        elif population == "MIX":
            _res = mc_sfh_histories_MIX(
                lgt,
                dt,
                logm0_bin,
                mah_params_bin,
                n_histories,
                index_select,
                index_high,
                fstar_tdelay,
                seed=0,
                pdf_parameters_MS=pdf_parameters_MS,
                pdf_parameters_Q=pdf_parameters_Q,
                diffstar_kernel=diffstar_kernel,
            )
        else:
            raise NotImplementedError("'population' needs to be 'Q', 'MS' or 'MIX'")

        mstar_MC, sfr_MC, fstar_MC, p50_MC = _res

        sFstar = fstar_MC / mstar_MC[:, index_select]

        sm_mean_MC[i] = np.nanmean(np.log10(mstar_MC), axis=0)
        sm_var_MC[i] = np.nanstd(np.log10(mstar_MC), axis=0) ** 2

        fstar_MC_MS = fstar_MC.copy()
        fstar_MC_Q = fstar_MC.copy()
        fstar_MC_MS[(sFstar < 1e-11)] = np.nan
        fstar_MC_Q[(sFstar > 1e-11)] = np.nan
        fstar_mean_MS_MC[i] = np.nanmean(np.log10(fstar_MC_MS), axis=0)
        fstar_mean_Q_MC[i] = np.nanmean(np.log10(fstar_MC_Q), axis=0)
        fstar_var_MS_MC[i] = np.nanstd(np.log10(fstar_MC_MS), axis=0) ** 2
        fstar_var_Q_MC[i] = np.nanstd(np.log10(fstar_MC_Q), axis=0) ** 2
        quench_frac_MC[i] = np.sum(sFstar < 1e-11, axis=0) / (
            np.sum(sFstar < 1e-11, axis=0) + np.sum(sFstar > 1e-11, axis=0)
        )

        p50_bin_id = np.digitize(p50_MC, p50_bins) - 1
        p50_bin_id = np.clip(p50_bin_id, 0, len(p50_bins) - 1).astype(int)
        for j in range(len(p50_bins) - 1):
            msk = p50_bin_id == j
            sm_mean_MC_p50[i, j] = np.nanmean(np.log10(mstar_MC[msk]), axis=0)
            sm_var_MC_p50[i, j] = np.nanstd(np.log10(mstar_MC[msk]), axis=0) ** 2
            quench_frac_MC_p50[i, j] = np.sum(sFstar[msk] < 1e-11, axis=0) / (
                np.sum(sFstar[msk] < 1e-11, axis=0)
                + np.sum(sFstar[msk] > 1e-11, axis=0)
            )
    _out = (
        sm_mean_MC,
        sm_var_MC,
        fstar_mean_MS_MC,
        fstar_mean_Q_MC,
        fstar_var_MS_MC,
        fstar_var_Q_MC,
        quench_frac_MC,
        sm_mean_MC_p50,
        sm_var_MC_p50,
        quench_frac_MC_p50,
    )

    return _out


# Ignore the rest of this script for now.


def mc_sfh_population_R(
    lgt,
    dt,
    logmh,
    mah_params,
    n_histories,
    index_select,
    index_high,
    fstar_tdelay,
    p50,
    seed=0,
    kwargs_pdf={},
    kwargs_R={},
):
    """Generate Monte Carlo realization of the assembly of a population of halos.
    Parameters
    ----------
    lgt : ndarray
        Array of log10 cosmic times in units of Gyr
    dt : float
        Time step sizes in units of Gyr
    logmh : float
        Base-10 log of present-day halo mass of the halo population
    mah_params : ndarray, size (n_mah_haloes x n_mah_params)
        Array with the diffmah parameters that will be marginalized over. Could
        be either individual fits of n_mah_haloes haloes, or be n_mah_haloes
        samples from a population model. They are chosen at random n_halos times.
    n_halos : int
        Number of halos in the population.
    seed : int, optional
        Random number seed
    **kwargs : floats
        All parameters of the SFH PDF model are accepted as keyword arguments.
        Default values are set by rockstar_pdf_model.DEFAULT_SFH_PDF_PARAMS
    Returns
    -------
    mstar : ndarray of shape (n_halos, n_times)
        Stores cumulative stellar mass history in units of Msun/yr.
    sfr : ndarray of shape (n_halos, n_times)
        Stores star formation rate history in units of Msun/yr.
    """

    logmh = np.atleast_1d(logmh).astype("f4")
    assert logmh.size == 1, "Input halo mass must be a scalar"

    n_mah = len(mah_params)

    sampled_mahs_inds = RandomState(seed).choice(n_mah, n_histories, replace=True)
    mah_params_sampled = mah_params[sampled_mahs_inds]
    p50_sampled = p50[sampled_mahs_inds]

    _res = get_smah_means_and_covs_quench(logmh, **kwargs_pdf)
    frac_quench, means_quench, covs_quench = _res
    frac_quench = frac_quench[0]
    means_quench = means_quench[0]
    covs_quench = covs_quench[0]

    Rvals = np.array(_get_slopes(logmh, **kwargs_R))[:, 0]
    shifts = _get_shift_to_PDF_mean(p50_sampled, Rvals)

    sfh_params = np.zeros((n_histories, 9))

    means_quench_shifted = means_quench[:, None] + shifts
    for i in range(n_histories):

        sfh_params[i] = RandomState(seed + 1 + i).multivariate_normal(
            means_quench_shifted[:, i], covs_quench, size=1
        )
    sfr_params = np.zeros((n_histories, 6))
    sfr_params[:, :2] = sfh_params[:, :2]
    sfr_params[:, 2] = UK
    sfr_params[:, 3:] = sfh_params[:, 2:5]
    q_params = sfh_params[:, 5:9]

    _res = calculate_sm_vmap_batch(
        lgt,
        dt,
        mah_params_sampled,
        sfr_params,
        q_params,
        index_select,
        index_high,
        fstar_tdelay,
    )
    mstar = _res[0]
    sfr = _res[1]
    fstar = _res[2]

    return mstar, sfr, fstar
