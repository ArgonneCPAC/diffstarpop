import numpy as np
from numpy.random import RandomState
from jax import vmap
from jax import jit as jjit
from jax import numpy as jnp
from jax import random as jran
from jax import lax
from collections import OrderedDict
from functools import partial

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
from diffstar.utils import jax_np_interp, _jax_get_dt_array


from diffmah.individual_halo_assembly import _calc_halo_history


from .pdf_quenched import (
    get_smah_means_and_covs_quench,
    DEFAULT_SFH_PDF_QUENCH_PARAMS,
    frac_quench_vs_lgm0,
)
from .pdf_mainseq import get_smah_means_and_covs_mainseq, DEFAULT_SFH_PDF_MAINSEQ_PARAMS
from .pdf_model_assembly_bias_shifts import (
    DEFAULT_R_QUENCH_PARAMS,
    DEFAULT_R_MAINSEQ_PARAMS,
    _get_slopes_quench,
    _get_slopes_mainseq,
    _get_shift_to_PDF_mean,
)
from .star_wrappers import (
    sm_sfr_history_diffstar_scan_XsfhXmah_vmap,
    sm_sfr_history_diffstar_scan_MS_XsfhXmah_vmap,
    sfr_history_diffstar_scan,
    sfr_history_diffstar_scan_MS,
)
from .utils import _tw_cuml_lax_kern

from diffsky.diffndhist import tw_ndhist_weighted


# from .pdf_model_assembly_bias_shifts import _get_shift_to_PDF_mean, _get_slopes
_tw_cuml_lax_kern_vmap = jjit(vmap(_tw_cuml_lax_kern, in_axes=(0, None, None)))
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


@partial(jjit, static_argnames=["n_histories"])
def draw_sfh_Q(
    t_table,
    logmh,
    mah_params,
    p50,
    n_histories,
    ran_key,
    index_select,
    index_high,
    fstar_tdelay,
    pdf_parameters=DEFAULT_SFH_PDF_QUENCH_PARAMS,
    R_model_params=DEFAULT_R_QUENCH_PARAMS,
):
    """
    Generate Monte Carlo realization of the star formation histories of
    quenched galaxies for a single halo mass bin.

    There is correlation with p50.

    Parameters
    ----------
    t_table : ndarray of shape (n_times, )
        Cosmic time array in Gyr.
    logmh : float
        Base-10 log of present-day halo mass of the halo population
    mah_params : ndarray of shape (n_mah_haloes x n_mah_params)
        Array with the diffmah parameters that will be marginalized over. Could
        be either individual fits of n_mah_haloes haloes, or be n_mah_haloes
        samples from a population model. They are chosen at random n_halos times.
    p50 : mah_params : ndarray of shape (n_mah_haloes, )
        Formation time percentile of each halo conditioned on halo mass.
    n_histories : int
        Number of SFH histories to generate by DiffstarPop.
    ran_key : ndarray of shape 2
        JAX random key.
    index_select: ndarray of shape (n_times_fstar, )
        Snapshot indices used in fstar computation
    index_high: ndarray of shape (n_times_fstar, )
        Indices of np.searchsorted(t, t - fstar_tdelay)[index_select]
    fstar_tdelay: float
        Time interval in Gyr for fstar definition.
        fstar = (mstar(t) - mstar(t-fstar_tdelay)) / fstar_tdelay[Gyr]
    pdf_model_params : ndarray of shape (n_pdf, )
        Array containing the Diffstarpop parameters for the quenched population.
        Default is DEFAULT_SFH_PDF_QUENCH_PARAMS.
    R_model_params: ndarray of shape (n_R, )
        Array containing the Diffstarpop parameters for the correlation between
        diffstar and diffmah parameters for the quenched population.

    Returns
    -------
    mstar : ndarray of shape (n_histories, n_times)
        Stores cumulative stellar mass history in units of Msun/yr.
    sfr : ndarray of shape (n_histories, n_times)
        Stores star formation rate history in units of Msun/yr.
    fstar : ndarray of shape (n_histories, n_times_fstar)
        SFH averaged over timescale fstar_tdelay in units of Msun/yr assuming h=1.
    """
    lgt = jnp.log10(t_table)
    dt = _jax_get_dt_array(t_table)
    logmh = jnp.atleast_1d(logmh)

    choice_key, quench_key, ran_key = jran.split(ran_key, 3)
    n_mah = len(mah_params)

    sampled_mahs_inds = jran.choice(
        choice_key, n_mah, shape=(n_histories,), replace=True
    )
    mah_params_sampled = mah_params[sampled_mahs_inds]
    p50_sampled = p50[sampled_mahs_inds]

    _res = get_smah_means_and_covs_quench(logmh, *pdf_parameters)
    frac_quench, means_quench, covs_quench = _res
    frac_quench = frac_quench[0]
    means_quench = means_quench[0]
    covs_quench = covs_quench[0]

    R_vals_quench = _get_slopes_quench(logmh, *R_model_params)
    R_vals_quench = jnp.array(R_vals_quench)[:, 0]

    shifts_quench = jnp.einsum("p,h->hp", R_vals_quench, (p50_sampled - 0.5))

    sfh_params = jran.multivariate_normal(
        quench_key, means_quench, covs_quench, shape=(n_histories,)
    )
    sfh_params = sfh_params + shifts_quench

    sfr_params = sfh_params[:, 0:4]
    q_params = sfh_params[:, 4:8]

    _res = sm_sfr_history_diffstar_scan_XsfhXmah_vmap(
        t_table,
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
    return mstar, sfr, fstar, p50_sampled


@partial(jjit, static_argnames=["n_histories"])
def sumstats_sfh_Q(
    t_table,
    logmh,
    mah_params,
    p50,
    n_histories,
    ran_key,
    index_select,
    index_high,
    fstar_tdelay,
    pdf_parameters=DEFAULT_SFH_PDF_QUENCH_PARAMS,
    R_model_params=DEFAULT_R_QUENCH_PARAMS,
):
    """
    Compute differentiable summary statistics from monte-carlo histories for
    a quenched population of galaxies.

    There is correlation with p50.

    Parameters
    ----------
    t_table : ndarray of shape (n_t, )
        Cosmic time array in Gyr.
    logmh : float
        Base-10 log of present-day halo mass of the halo population
    mah_params : ndarray of shape (n_mah_haloes x n_mah_params)
        Array with the diffmah parameters that will be marginalized over. Could
        be either individual fits of n_mah_haloes haloes, or be n_mah_haloes
        samples from a population model. They are chosen at random n_halos times.
    p50 : mah_params : ndarray of shape (n_mah_haloes, )
        Formation time percentile of each halo conditioned on halo mass.
    n_histories : int
        Number of SFH histories to generate by DiffstarPop.
    ran_key : ndarray of shape 2
        JAX random key.
    index_select: ndarray of shape (n_times_fstar, )
        Snapshot indices used in fstar computation
    index_high: ndarray of shape (n_times_fstar, )
        Indices of np.searchsorted(t, t - fstar_tdelay)[index_select]
    fstar_tdelay: float
        Time interval in Gyr for fstar definition.
        fstar = (mstar(t) - mstar(t-fstar_tdelay)) / fstar_tdelay[Gyr]
    pdf_model_params : ndarray of shape (n_pdf, )
        Array containing the Diffstarpop parameters for the quenched population.
        Default is DEFAULT_SFH_PDF_QUENCH_PARAMS.
    R_model_params: ndarray of shape (n_R, )
        Array containing the Diffstarpop parameters for the correlation between
        diffstar and diffmah parameters for the quenched population.

    Returns
    -------
    mean_sm : ndarray of shape (n_t, )
        Average log10 Stellar Mass.
    variance_sm : ndarray of shape (n_t, )
        Variance of log10 Stellar Mass.
    mean_fstar_MS : ndarray of shape (n_t_fstar, )
        Average fstar (average SFH within some timescale) for main sequence galaxies.
    mean_fstar_Q : ndarray of shape (n_t_fstar, )
        Average fstar (average SFH within some timescale) for quenched galaxies.
    variance_fstar_MS : ndarray of shape (n_t_fstar, )
        Variance of fstar (average SFH within some timescale) for MS galaxies.
    variance_fstar_Q : ndarray of shape (n_t_fstar, )
        Variance of fstar (average SFH within some timescale) for Q galaxies.
    quench_frac : ndarray of shape (n_t_fstar, )
        Fraction of quenched galaxies.
    """

    mstar, sfr, fstar, p50_sampled = draw_sfh_Q(
        t_table,
        logmh,
        mah_params,
        p50,
        n_histories,
        ran_key,
        index_select,
        index_high,
        fstar_tdelay,
        pdf_parameters,
        R_model_params,
    )

    sFstar = fstar / mstar[:, index_select]
    weights_quench_bin = jnp.where(sFstar > 1e-11, 1.0, 0.0)

    return compute_sumstats(mstar, sfr, fstar, p50_sampled, weights_quench_bin)


_A = (None, 0, 0, 0, *[None] * 7)
draw_sfh_Q_vmap = jjit(vmap(draw_sfh_Q, in_axes=_A), static_argnames=["n_histories"])
sumstats_sfh_Q_vmap = jjit(
    vmap(sumstats_sfh_Q, in_axes=_A), static_argnames=["n_histories"]
)


@partial(jjit, static_argnames=["n_histories"])
def draw_sfh_MS(
    t_table,
    logmh,
    mah_params,
    p50,
    n_histories,
    ran_key,
    index_select,
    index_high,
    fstar_tdelay,
    pdf_parameters=DEFAULT_SFH_PDF_MAINSEQ_PARAMS,
    R_model_params=DEFAULT_R_MAINSEQ_PARAMS,
):
    """
    Generate Monte Carlo realization of the star formation histories of
    main sequence galaxies for a single halo mass bin.

    There is correlation with p50.

    Parameters
    ----------
    t_table : ndarray of shape (n_times, )
        Cosmic time array in Gyr.
    logmh : float
        Base-10 log of present-day halo mass of the halo population
    mah_params : ndarray of shape (n_mah_haloes x n_mah_params)
        Array with the diffmah parameters that will be marginalized over. Could
        be either individual fits of n_mah_haloes haloes, or be n_mah_haloes
        samples from a population model. They are chosen at random n_halos times.
    p50 : mah_params : ndarray of shape (n_mah_haloes, )
        Formation time percentile of each halo conditioned on halo mass.
    n_histories : int
        Number of SFH histories to generate by DiffstarPop.
    ran_key : ndarray of shape 2
        JAX random key.
    index_select: ndarray of shape (n_times_fstar, )
        Snapshot indices used in fstar computation
    index_high: ndarray of shape (n_times_fstar, )
        Indices of np.searchsorted(t, t - fstar_tdelay)[index_select]
    fstar_tdelay: float
        Time interval in Gyr for fstar definition.
        fstar = (mstar(t) - mstar(t-fstar_tdelay)) / fstar_tdelay[Gyr]
    pdf_model_params : ndarray of shape (n_pdf, )
        Array containing the Diffstarpop parameters for the main sequence population.
        Default is DEFAULT_SFH_PDF_MAINSEQ_PARAMS.
    R_model_params: ndarray of shape (n_R, )
        Array containing the Diffstarpop parameters for the correlation between
        diffstar and diffmah parameters for the main sequence population.

    Returns
    -------
    mstar : ndarray of shape (n_histories, n_times)
        Stores cumulative stellar mass history in units of Msun/yr.
    sfr : ndarray of shape (n_histories, n_times)
        Stores star formation rate history in units of Msun/yr.
    fstar : ndarray of shape (n_histories, n_times_fstar)
        SFH averaged over timescale fstar_tdelay in units of Msun/yr assuming h=1.
    """
    lgt = jnp.log10(t_table)
    dt = _jax_get_dt_array(t_table)
    logmh = jnp.atleast_1d(logmh)

    choice_key, mainseq_key, ran_key = jran.split(ran_key, 3)
    n_mah = len(mah_params)

    sampled_mahs_inds = jran.choice(
        choice_key, n_mah, shape=(n_histories,), replace=True
    )
    mah_params_sampled = mah_params[sampled_mahs_inds]
    p50_sampled = p50[sampled_mahs_inds]

    _res = get_smah_means_and_covs_mainseq(logmh, *pdf_parameters)
    means_mainseq, covs_mainseq = _res
    means_mainseq = means_mainseq[0]
    covs_mainseq = covs_mainseq[0]

    R_vals_mainseq = _get_slopes_mainseq(logmh, *R_model_params)
    R_vals_mainseq = jnp.array(R_vals_mainseq)[:, 0]

    shifts_mainseq = jnp.einsum("p,h->hp", R_vals_mainseq, (p50_sampled - 0.5))

    sfr_params = jran.multivariate_normal(
        mainseq_key, means_mainseq, covs_mainseq, shape=(n_histories,)
    )
    sfr_params = sfr_params + shifts_mainseq

    _res = sm_sfr_history_diffstar_scan_MS_XsfhXmah_vmap(
        t_table,
        lgt,
        dt,
        mah_params_sampled,
        sfr_params,
        index_select,
        index_high,
        fstar_tdelay,
    )

    mstar = _res[0]
    sfr = _res[1]
    fstar = _res[2]

    return mstar, sfr, fstar, p50_sampled


@partial(jjit, static_argnames=["n_histories"])
def sumstats_sfh_MS(
    t_table,
    logmh,
    mah_params,
    p50,
    n_histories,
    ran_key,
    index_select,
    index_high,
    fstar_tdelay,
    pdf_parameters=DEFAULT_SFH_PDF_MAINSEQ_PARAMS,
    R_model_params=DEFAULT_R_MAINSEQ_PARAMS,
):
    """
    Compute differentiable summary statistics from monte-carlo histories for
    a main Sequence population of galaxies.

    There is correlation with p50.

    Parameters
    ----------
    t_table : ndarray of shape (n_t, )
        Cosmic time array in Gyr.
    logmh : float
        Base-10 log of present-day halo mass of the halo population
    mah_params : ndarray of shape (n_mah_haloes x n_mah_params)
        Array with the diffmah parameters that will be marginalized over. Could
        be either individual fits of n_mah_haloes haloes, or be n_mah_haloes
        samples from a population model. They are chosen at random n_halos times.
    p50 : mah_params : ndarray of shape (n_mah_haloes, )
        Formation time percentile of each halo conditioned on halo mass.
    n_histories : int
        Number of SFH histories to generate by DiffstarPop.
    ran_key : ndarray of shape 2
        JAX random key.
    index_select: ndarray of shape (n_times_fstar, )
        Snapshot indices used in fstar computation
    index_high: ndarray of shape (n_times_fstar, )
        Indices of np.searchsorted(t, t - fstar_tdelay)[index_select]
    fstar_tdelay: float
        Time interval in Gyr for fstar definition.
        fstar = (mstar(t) - mstar(t-fstar_tdelay)) / fstar_tdelay[Gyr]
    pdf_model_params : ndarray of shape (n_pdf, )
        Array containing the Diffstarpop parameters for the main sequence population.
        Default is DEFAULT_SFH_PDF_MAINSEQ_PARAMS.
    R_model_params: ndarray of shape (n_R, )
        Array containing the Diffstarpop parameters for the correlation between
        diffstar and diffmah parameters for the main sequence population.

    Returns
    -------
    mean_sm : ndarray of shape (n_t, )
        Average log10 Stellar Mass.
    variance_sm : ndarray of shape (n_t, )
        Variance of log10 Stellar Mass.
    mean_fstar_MS : ndarray of shape (n_t_fstar, )
        Average fstar (average SFH within some timescale) for main sequence galaxies.
    mean_fstar_Q : ndarray of shape (n_t_fstar, )
        Average fstar (average SFH within some timescale) for quenched galaxies.
    variance_fstar_MS : ndarray of shape (n_t_fstar, )
        Variance of fstar (average SFH within some timescale) for MS galaxies.
    variance_fstar_Q : ndarray of shape (n_t_fstar, )
        Variance of fstar (average SFH within some timescale) for Q galaxies.
    quench_frac : ndarray of shape (n_t_fstar, )
        Fraction of quenched galaxies.
    """
    mstar, sfr, fstar, p50_sampled = draw_sfh_MS(
        t_table,
        logmh,
        mah_params,
        p50,
        n_histories,
        ran_key,
        index_select,
        index_high,
        fstar_tdelay,
        pdf_parameters,
        R_model_params,
    )

    sFstar = fstar / mstar[:, index_select]
    weights_quench_bin = jnp.where(sFstar > 1e-11, 1.0, 0.0)

    return compute_sumstats(mstar, sfr, fstar, p50_sampled, weights_quench_bin)


_A = (None, 0, 0, 0, *[None] * 7)
draw_sfh_MS_vmap = jjit(vmap(draw_sfh_MS, in_axes=_A), static_argnames=["n_histories"])
sumstats_sfh_MS_vmap = jjit(
    vmap(sumstats_sfh_MS, in_axes=_A), static_argnames=["n_histories"]
)


@partial(jjit, static_argnames=["n_histories"])
def draw_sfh_MIX(
    t_table,
    logmh,
    mah_params,
    p50,
    n_histories,
    ran_key,
    index_select,
    index_high,
    fstar_tdelay,
    pdf_parameters_Q=DEFAULT_SFH_PDF_QUENCH_PARAMS,
    pdf_parameters_MS=DEFAULT_SFH_PDF_MAINSEQ_PARAMS,
    R_model_params_Q=DEFAULT_R_QUENCH_PARAMS,
    R_model_params_MS=DEFAULT_R_MAINSEQ_PARAMS,
):
    """
    Generate Monte Carlo realization of the star formation histories of
    a mixed population of quenched and main sequence galaxies
    for a single halo mass bin.

    There is correlation with p50.

    Parameters
    ----------
    t_table : ndarray of shape (n_times, )
        Cosmic time array in Gyr.
    logmh : float
        Base-10 log of present-day halo mass of the halo population
    mah_params : ndarray of shape (n_mah_haloes x n_mah_params)
        Array with the diffmah parameters that will be marginalized over. Could
        be either individual fits of n_mah_haloes haloes, or be n_mah_haloes
        samples from a population model. They are chosen at random n_halos times.
    p50 : mah_params : ndarray of shape (n_mah_haloes, )
        Formation time percentile of each halo conditioned on halo mass.
    n_histories : int
        Number of SFH histories to generate by DiffstarPop.
    ran_key : ndarray of shape 2
        JAX random key.
    index_select: ndarray of shape (n_times_fstar, )
        Snapshot indices used in fstar computation
    index_high: ndarray of shape (n_times_fstar, )
        Indices of np.searchsorted(t, t - fstar_tdelay)[index_select]
    fstar_tdelay: float
        Time interval in Gyr for fstar definition.
        fstar = (mstar(t) - mstar(t-fstar_tdelay)) / fstar_tdelay[Gyr]
    pdf_model_params_Q : ndarray of shape (n_pdf, )
        Array containing the Diffstarpop parameters for the quenched population.
        Default is DEFAULT_SFH_PDF_QUENCH_PARAMS.
    pdf_model_params_MS : ndarray of shape (n_pdf, )
        Array containing the Diffstarpop parameters for the main sequence population.
        Default is DEFAULT_SFH_PDF_MAINSEQ_PARAMS.
    R_model_params_Q: ndarray of shape (n_R, )
        Array containing the Diffstarpop parameters for the correlation between
        diffstar and diffmah parameters for the quenched population.
    R_model_params_MS: ndarray of shape (n_R, )
        Array containing the Diffstarpop parameters for the correlation between
        diffstar and diffmah parameters for the main sequence population.

    Returns
    -------
    mstar : ndarray of shape (n_histories, n_times)
        Stores cumulative stellar mass history in units of Msun/yr.
    sfr : ndarray of shape (n_histories, n_times)
        Stores star formation rate history in units of Msun/yr.
    fstar : ndarray of shape (n_histories, n_times_fstar)
        SFH averaged over timescale fstar_tdelay in units of Msun/yr assuming h=1.
    """
    lgt = jnp.log10(t_table)
    dt = _jax_get_dt_array(t_table)
    logmh = jnp.atleast_1d(logmh)

    (choice_key, quench_key, mainseq_key, fquench_key, ran_key) = jran.split(ran_key, 5)
    n_mah = len(mah_params)

    sampled_mahs_inds = jran.choice(
        choice_key, n_mah, shape=(n_histories,), replace=True
    )
    mah_params_sampled = mah_params[sampled_mahs_inds]
    p50_sampled = p50[sampled_mahs_inds]

    _res = get_smah_means_and_covs_mainseq(logmh, *pdf_parameters_MS)
    means_mainseq, covs_mainseq = _res
    means_mainseq = means_mainseq[0]
    covs_mainseq = covs_mainseq[0]

    R_vals_mainseq = _get_slopes_mainseq(logmh, *R_model_params_MS)
    R_vals_mainseq = jnp.array(R_vals_mainseq)[:, 0]
    shifts_mainseq = jnp.einsum("p,h->hp", R_vals_mainseq, (p50_sampled - 0.5))

    _res = get_smah_means_and_covs_quench(logmh, *pdf_parameters_Q)
    frac_quench, means_quench, covs_quench = _res
    frac_quench = frac_quench[0]
    means_quench = means_quench[0]
    covs_quench = covs_quench[0]

    _res = _get_slopes_quench(logmh, *R_model_params_Q)
    R_Fquench, R_vals_quench = _res[0], _res[1:]
    R_vals_quench = jnp.array(R_vals_quench)[:, 0]
    shifts_quench = jnp.einsum("p,h->hp", R_vals_quench, (p50_sampled - 0.5))
    shifts_Fquench = R_Fquench * (p50_sampled - 0.5)
    fquench_x0 = pdf_parameters_Q[0] + shifts_Fquench
    frac_quench = frac_quench_vs_lgm0(logmh, fquench_x0, *pdf_parameters_Q[1:4])

    sfh_params_Q = jran.multivariate_normal(
        quench_key, means_quench, covs_quench, shape=(n_histories,)
    )
    sfh_params_Q = sfh_params_Q + shifts_quench

    sfr_params_Q = sfh_params_Q[:, 0:4]
    q_params_Q = sfh_params_Q[:, 4:8]

    sfr_params_MS = jran.multivariate_normal(
        mainseq_key, means_mainseq, covs_mainseq, shape=(n_histories,)
    )
    sfr_params_MS = sfr_params_MS + shifts_mainseq
    # q_params_MS = jnp.ones_like(q_params_Q) * 10.0

    mstar_Q, sfr_Q, fstar_Q = sm_sfr_history_diffstar_scan_XsfhXmah_vmap(
        t_table,
        lgt,
        dt,
        mah_params_sampled,
        sfr_params_Q,
        q_params_Q,
        index_select,
        index_high,
        fstar_tdelay,
    )
    mstar_MS, sfr_MS, fstar_MS = sm_sfr_history_diffstar_scan_MS_XsfhXmah_vmap(
        t_table,
        lgt,
        dt,
        mah_params_sampled,
        sfr_params_MS,
        index_select,
        index_high,
        fstar_tdelay,
    )
    mstar = jnp.concatenate((mstar_Q, mstar_MS))
    sfr = jnp.concatenate((sfr_Q, sfr_MS))
    fstar = jnp.concatenate((fstar_Q, fstar_MS))

    weight = jnp.concatenate(
        (
            frac_quench,
            (1.0 - frac_quench),
        )
    )
    p50_sampled = jnp.concatenate((p50_sampled, p50_sampled))
    return mstar, sfr, fstar, p50_sampled, weight


@partial(jjit, static_argnames=["n_histories"])
def sumstats_sfh_MIX(
    t_table,
    logmh,
    mah_params,
    p50,
    n_histories,
    ran_key,
    index_select,
    index_high,
    fstar_tdelay,
    pdf_parameters_Q=DEFAULT_SFH_PDF_QUENCH_PARAMS,
    pdf_parameters_MS=DEFAULT_SFH_PDF_MAINSEQ_PARAMS,
    R_model_params_Q=DEFAULT_R_QUENCH_PARAMS,
    R_model_params_MS=DEFAULT_R_MAINSEQ_PARAMS,
):
    """
    Compute differentiable summary statistics from monte-carlo histories for
    a mixed population of quenched and main sequence galaxies.

    There is correlation with p50.

    Parameters
    ----------
    t_table : ndarray of shape (n_t, )
        Cosmic time array in Gyr.
    logmh : float
        Base-10 log of present-day halo mass of the halo population
    mah_params : ndarray of shape (n_mah_haloes x n_mah_params)
        Array with the diffmah parameters that will be marginalized over. Could
        be either individual fits of n_mah_haloes haloes, or be n_mah_haloes
        samples from a population model. They are chosen at random n_halos times.
    p50 : mah_params : ndarray of shape (n_mah_haloes, )
        Formation time percentile of each halo conditioned on halo mass.
    n_histories : int
        Number of SFH histories to generate by DiffstarPop.
    ran_key : ndarray of shape 2
        JAX random key.
    index_select: ndarray of shape (n_times_fstar, )
        Snapshot indices used in fstar computation
    index_high: ndarray of shape (n_times_fstar, )
        Indices of np.searchsorted(t, t - fstar_tdelay)[index_select]
    fstar_tdelay: float
        Time interval in Gyr for fstar definition.
        fstar = (mstar(t) - mstar(t-fstar_tdelay)) / fstar_tdelay[Gyr]
    pdf_model_params_Q : ndarray of shape (n_pdf, )
        Array containing the Diffstarpop parameters for the quenched population.
        Default is DEFAULT_SFH_PDF_QUENCH_PARAMS.
    pdf_model_params_MS : ndarray of shape (n_pdf, )
        Array containing the Diffstarpop parameters for the main sequence population.
        Default is DEFAULT_SFH_PDF_MAINSEQ_PARAMS.
    R_model_params_Q: ndarray of shape (n_R, )
        Array containing the Diffstarpop parameters for the correlation between
        diffstar and diffmah parameters for the quenched population.
    R_model_params_MS: ndarray of shape (n_R, )
        Array containing the Diffstarpop parameters for the correlation between
        diffstar and diffmah parameters for the main sequence population.

    Returns
    -------
    mean_sm : ndarray of shape (n_t, )
        Average log10 Stellar Mass.
    variance_sm : ndarray of shape (n_t, )
        Variance of log10 Stellar Mass.
    mean_fstar_MS : ndarray of shape (n_t_fstar, )
        Average fstar (average SFH within some timescale) for main sequence galaxies.
    mean_fstar_Q : ndarray of shape (n_t_fstar, )
        Average fstar (average SFH within some timescale) for quenched galaxies.
    variance_fstar_MS : ndarray of shape (n_t_fstar, )
        Variance of fstar (average SFH within some timescale) for MS galaxies.
    variance_fstar_Q : ndarray of shape (n_t_fstar, )
        Variance of fstar (average SFH within some timescale) for Q galaxies.
    quench_frac : ndarray of shape (n_t_fstar, )
        Fraction of quenched galaxies.
    """
    mstar, sfr, fstar, p50_sampled, weight = draw_sfh_MIX(
        t_table,
        logmh,
        mah_params,
        p50,
        n_histories,
        ran_key,
        index_select,
        index_high,
        fstar_tdelay,
        pdf_parameters_Q,
        pdf_parameters_MS,
        R_model_params_Q,
        R_model_params_MS,
    )

    sFstar = fstar / mstar[:, index_select]
    weights_quench_bin = jnp.where(sFstar > 1e-11, 1.0, 0.0)

    return compute_sumstats_MIX(
        mstar, sfr, fstar, p50_sampled, weights_quench_bin, weight
    )


_A = (None, 0, 0, 0, *[None] * 9)
draw_sfh_MIX_vmap = jjit(
    vmap(draw_sfh_MIX, in_axes=_A), static_argnames=["n_histories"]
)
sumstats_sfh_MIX_vmap = jjit(
    vmap(sumstats_sfh_MIX, in_axes=_A), static_argnames=["n_histories"]
)


@jjit
def compute_sumstats_MIX(
    mstar_histories,
    sfr_histories,
    fstar_histories,
    p50,
    weights_MS,
    w_frac_quench,
):
    """
    Compute differentiable summary statistics from monte carlo sampled histories
    for a single mass bin.

    Parameters
    ----------
    mstar_histories : ndarray of shape (n_histories, n_t)
        SMH history samples
    sfr_histories : ndarray of shape (n_histories, n_t)
        SFH history samples
    fstar_histories : ndarray of shape (n_histories, n_t_fstar)
        Fstar samples
    p50 : ...
    weights_MS : ndarray of shape (n_histories, n_t_fstar)
        Weight array indicating when galaxy history is quenched.
            0: sSFR(t) < 1e-11
            1: sSFR(t) > 1e-11
    w_frac_quench : ndarray of shape (n_histories, )
        Weight array indicating which the population of a SFH history.
    Returns
    -------
    mean_sm : ndarray of shape (n_t, )
        Average log10 Stellar Mass.
    variance_sm : ndarray of shape (n_t, )
        Variance of log10 Stellar Mass.
    mean_fstar_MS : ndarray of shape (n_t_fstar, )
        Average fstar (average SFH within some timescale) for main sequence galaxies.
    mean_fstar_Q : ndarray of shape (n_t_fstar, )
        Average fstar (average SFH within some timescale) for quenched galaxies.
    variance_fstar_MS : ndarray of shape (n_t_fstar, )
        Variance of fstar (average SFH within some timescale) for MS galaxies.
    variance_fstar_Q : ndarray of shape (n_t_fstar, )
        Variance of fstar (average SFH within some timescale) for Q galaxies.
    quench_frac : ndarray of shape (n_t_fstar, )
        Fraction of quenched galaxies.
    """
    weights_Q = 1.0 - weights_MS

    # Clip weights. When all weights in a time
    # step are 0, Nans will occur in gradients.
    eps = 1e-10
    weights_Q = jnp.clip(weights_Q * w_frac_quench[:, None], eps, None)
    weights_MS = jnp.clip(weights_MS * w_frac_quench[:, None], eps, None)
    w_frac_quench = jnp.clip(w_frac_quench, eps, None)

    mstar_histories = jnp.where(mstar_histories > 0.0, jnp.log10(mstar_histories), 0.0)
    sfr_histories = jnp.where(sfr_histories > 0.0, jnp.log10(sfr_histories), 0.0)
    fstar_histories = jnp.where(fstar_histories > 0.0, jnp.log10(fstar_histories), 0.0)

    mean_sm = jnp.average(mstar_histories, weights=w_frac_quench, axis=0)
    mean_fstar_MS = jnp.average(fstar_histories, weights=weights_MS, axis=0)
    mean_fstar_Q = jnp.average(fstar_histories, weights=weights_Q, axis=0)

    variance_sm = jnp.average(
        (mstar_histories - mean_sm[None, :]) ** 2,
        weights=w_frac_quench,
        axis=0,
    )

    variance_fstar_MS = jnp.average(
        (fstar_histories - mean_fstar_MS[None, :]) ** 2,
        weights=weights_MS,
        axis=0,
    )
    variance_fstar_Q = jnp.average(
        (fstar_histories - mean_fstar_Q[None, :]) ** 2,
        weights=weights_Q,
        axis=0,
    )

    NHALO_MS = jnp.sum(weights_MS, axis=0)
    NHALO_Q = jnp.sum(weights_Q, axis=0)
    quench_frac = NHALO_Q / (NHALO_Q + NHALO_MS)

    mean_fstar_Q = jnp.where(quench_frac == 0.0, 0.0, mean_fstar_Q)
    variance_fstar_Q = jnp.where(quench_frac == 0.0, 0.0, variance_fstar_Q)
    mean_fstar_MS = jnp.where(quench_frac == 1.0, 0.0, mean_fstar_MS)
    variance_fstar_MS = jnp.where(quench_frac == 1.0, 0.0, variance_fstar_MS)

    """
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
    """

    _out = (
        mean_sm,
        variance_sm,
        mean_fstar_MS,
        mean_fstar_Q,
        variance_fstar_MS,
        variance_fstar_Q,
        quench_frac,
        # mean_sm_p50,
        # variance_sm_p50,
        # quench_frac_p50,
    )
    return _out


@jjit
def compute_sumstats(mstar_histories, sfr_histories, fstar_histories, p50, weights_MS):
    """
    Compute differentiable summary statistics from monte carlo sampled histories
    for a single mass bin.

    Parameters
    ----------
    mstar_histories : ndarray of shape (n_histories, n_t)
        SMH history samples
    sfr_histories : ndarray of shape (n_histories, n_t)
        SFH history samples
    fstar_histories : ndarray of shape (n_histories, n_t_fstar)
        Fstar samples
    p50 : ...
    weights_MS : ndarray of shape (n_histories, n_t_fstar)
        Weight array indicating when galaxy history is quenched.
            0: sSFR(t) < 1e-11
            1: sSFR(t) > 1e-11

    Returns
    -------
    mean_sm : ndarray of shape (n_t, )
        Average log10 Stellar Mass.
    variance_sm : ndarray of shape (n_t, )
        Variance of log10 Stellar Mass.
    mean_fstar_MS : ndarray of shape (n_t_fstar, )
        Average fstar (average SFH within some timescale) for main sequence galaxies.
    mean_fstar_Q : ndarray of shape (n_t_fstar, )
        Average fstar (average SFH within some timescale) for quenched galaxies.
    variance_fstar_MS : ndarray of shape (n_t_fstar, )
        Variance of fstar (average SFH within some timescale) for MS galaxies.
    variance_fstar_Q : ndarray of shape (n_t_fstar, )
        Variance of fstar (average SFH within some timescale) for Q galaxies.
    quench_frac : ndarray of shape (n_t_fstar, )
        Fraction of quenched galaxies.
    """
    weights_Q = 1.0 - weights_MS

    # Clip weights. When all weights in a time
    # step are 0, Nans will occur in gradients.
    eps = 1e-10
    weights_Q = jnp.clip(weights_Q, eps, None)
    weights_MS = jnp.clip(weights_MS, eps, None)

    mstar_histories = jnp.where(mstar_histories > 0.0, jnp.log10(mstar_histories), 0.0)
    sfr_histories = jnp.where(sfr_histories > 0.0, jnp.log10(sfr_histories), 0.0)
    fstar_histories = jnp.where(fstar_histories > 0.0, jnp.log10(fstar_histories), 0.0)

    mean_sm = jnp.average(mstar_histories, axis=0)
    mean_fstar_MS = jnp.average(fstar_histories, weights=weights_MS, axis=0)
    mean_fstar_Q = jnp.average(fstar_histories, weights=weights_Q, axis=0)

    variance_sm = jnp.average(
        (mstar_histories - mean_sm[None, :]) ** 2,
        axis=0,
    )

    variance_fstar_MS = jnp.average(
        (fstar_histories - mean_fstar_MS[None, :]) ** 2,
        weights=weights_MS,
        axis=0,
    )
    variance_fstar_Q = jnp.average(
        (fstar_histories - mean_fstar_Q[None, :]) ** 2,
        weights=weights_Q,
        axis=0,
    )

    NHALO_MS = jnp.sum(weights_MS, axis=0)
    NHALO_Q = jnp.sum(weights_Q, axis=0)
    quench_frac = NHALO_Q / (NHALO_Q + NHALO_MS)

    mean_fstar_Q = jnp.where(quench_frac == 0.0, 0.0, mean_fstar_Q)
    variance_fstar_Q = jnp.where(quench_frac == 0.0, 0.0, variance_fstar_Q)
    mean_fstar_MS = jnp.where(quench_frac == 1.0, 0.0, mean_fstar_MS)
    variance_fstar_MS = jnp.where(quench_frac == 1.0, 0.0, variance_fstar_MS)

    """
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
    """

    _out = (
        mean_sm,
        variance_sm,
        mean_fstar_MS,
        mean_fstar_Q,
        variance_fstar_MS,
        variance_fstar_Q,
        quench_frac,
        # mean_sm_p50,
        # variance_sm_p50,
        # quench_frac_p50,
    )
    return _out


@jjit
def compute_sumstats_MIX_p50(
    mstar_histories,
    sfr_histories,
    p50,
    weights_MS,
    w_frac_quench,
):
    """
    Compute differentiable summary statistics from monte carlo sampled histories
    for a single mass bin.

    Parameters
    ----------
    mstar_histories : ndarray of shape (n_histories, n_t)
        SMH history samples
    sfr_histories : ndarray of shape (n_histories, n_t)
        SFH history samples
    p50 : ...
    weights_MS : ndarray of shape (n_histories, n_t_fstar)
        Weight array indicating when galaxy history is quenched.
            0: sSFR(t) < 1e-11
            1: sSFR(t) > 1e-11
    w_frac_quench : ndarray of shape (n_histories, )
        Weight array indicating which the population of a SFH history.
    Returns
    -------
    mean_sm : ndarray of shape (n_t, )
        Average log10 Stellar Mass.
    variance_sm : ndarray of shape (n_t, )
        Variance of log10 Stellar Mass.
    mean_fstar_MS : ndarray of shape (n_t_fstar, )
        Average fstar (average SFH within some timescale) for main sequence galaxies.
    mean_fstar_Q : ndarray of shape (n_t_fstar, )
        Average fstar (average SFH within some timescale) for quenched galaxies.
    variance_fstar_MS : ndarray of shape (n_t_fstar, )
        Variance of fstar (average SFH within some timescale) for MS galaxies.
    variance_fstar_Q : ndarray of shape (n_t_fstar, )
        Variance of fstar (average SFH within some timescale) for Q galaxies.
    quench_frac : ndarray of shape (n_t_fstar, )
        Fraction of quenched galaxies.
    """
    weights_Q = 1.0 - weights_MS

    # Clip weights. When all weights in a time
    # step are 0, Nans will occur in gradients.
    eps = 1e-10
    weights_Q = jnp.clip(weights_Q * w_frac_quench[:, None], eps, None)
    weights_MS = jnp.clip(weights_MS * w_frac_quench[:, None], eps, None)
    w_frac_quench = jnp.clip(w_frac_quench, eps, None)

    weights_early = jnp.where(p50 < 0.5, 1.0, 0.0)
    weights_late = 1.0 - weights_early

    weights_early *= w_frac_quench
    weights_late *= w_frac_quench
    weights_early = jnp.clip(weights_early, eps, None)
    weights_late = jnp.clip(weights_late, eps, None)

    mean_sm = jnp.average(mstar_histories, weights=w_frac_quench, axis=0)
    mean_sfr_MS = jnp.average(sfr_histories, weights=weights_MS, axis=0)
    mean_sfr_Q = jnp.average(sfr_histories, weights=weights_Q, axis=0)

    mean_sm_early = jnp.average(mstar_histories, weights=weights_early, axis=0)
    mean_sm_late = jnp.average(mstar_histories, weights=weights_late, axis=0)

    variance_sm = jnp.average(
        (mstar_histories - mean_sm[None, :]) ** 2,
        weights=w_frac_quench,
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


@partial(jjit, static_argnames=["n_histories"])
def sumstats_sfh_MIX_p50(
    t_table,
    logmh,
    mah_params,
    p50,
    n_histories,
    ran_key,
    index_select,
    index_high,
    fstar_tdelay,
    pdf_parameters_Q=DEFAULT_SFH_PDF_QUENCH_PARAMS,
    pdf_parameters_MS=DEFAULT_SFH_PDF_MAINSEQ_PARAMS,
    R_model_params_Q=DEFAULT_R_QUENCH_PARAMS,
    R_model_params_MS=DEFAULT_R_MAINSEQ_PARAMS,
):
    """
    Compute differentiable summary statistics from monte-carlo histories for
    a mixed population of quenched and main sequence galaxies.

    There is correlation with p50.

    Parameters
    ----------
    t_table : ndarray of shape (n_t, )
        Cosmic time array in Gyr.
    logmh : float
        Base-10 log of present-day halo mass of the halo population
    mah_params : ndarray of shape (n_mah_haloes x n_mah_params)
        Array with the diffmah parameters that will be marginalized over. Could
        be either individual fits of n_mah_haloes haloes, or be n_mah_haloes
        samples from a population model. They are chosen at random n_halos times.
    p50 : mah_params : ndarray of shape (n_mah_haloes, )
        Formation time percentile of each halo conditioned on halo mass.
    n_histories : int
        Number of SFH histories to generate by DiffstarPop.
    ran_key : ndarray of shape 2
        JAX random key.
    index_select: ndarray of shape (n_times_fstar, )
        Snapshot indices used in fstar computation
    index_high: ndarray of shape (n_times_fstar, )
        Indices of np.searchsorted(t, t - fstar_tdelay)[index_select]
    fstar_tdelay: float
        Time interval in Gyr for fstar definition.
        fstar = (mstar(t) - mstar(t-fstar_tdelay)) / fstar_tdelay[Gyr]
    pdf_model_params_Q : ndarray of shape (n_pdf, )
        Array containing the Diffstarpop parameters for the quenched population.
        Default is DEFAULT_SFH_PDF_QUENCH_PARAMS.
    pdf_model_params_MS : ndarray of shape (n_pdf, )
        Array containing the Diffstarpop parameters for the main sequence population.
        Default is DEFAULT_SFH_PDF_MAINSEQ_PARAMS.
    R_model_params_Q: ndarray of shape (n_R, )
        Array containing the Diffstarpop parameters for the correlation between
        diffstar and diffmah parameters for the quenched population.
    R_model_params_MS: ndarray of shape (n_R, )
        Array containing the Diffstarpop parameters for the correlation between
        diffstar and diffmah parameters for the main sequence population.

    Returns
    -------
    mean_sm : ndarray of shape (n_t, )
        Average log10 Stellar Mass.
    variance_sm : ndarray of shape (n_t, )
        Variance of log10 Stellar Mass.
    mean_fstar_MS : ndarray of shape (n_t_fstar, )
        Average fstar (average SFH within some timescale) for main sequence galaxies.
    mean_fstar_Q : ndarray of shape (n_t_fstar, )
        Average fstar (average SFH within some timescale) for quenched galaxies.
    variance_fstar_MS : ndarray of shape (n_t_fstar, )
        Variance of fstar (average SFH within some timescale) for MS galaxies.
    variance_fstar_Q : ndarray of shape (n_t_fstar, )
        Variance of fstar (average SFH within some timescale) for Q galaxies.
    quench_frac : ndarray of shape (n_t_fstar, )
        Fraction of quenched galaxies.
    """
    mstar, sfr, fstar, p50_sampled, weight = draw_sfh_MIX(
        t_table,
        logmh,
        mah_params,
        p50,
        n_histories,
        ran_key,
        index_select,
        index_high,
        fstar_tdelay,
        pdf_parameters_Q,
        pdf_parameters_MS,
        R_model_params_Q,
        R_model_params_MS,
    )

    ssfr = sfr / mstar
    # weights_quench_bin = jnp.where(ssfr > 1e-11, 1.0, 0.0)
    nhist, nt = ssfr.shape
    ssfr = jnp.where(ssfr > 0.0, jnp.log10(ssfr), -50.0)
    weights_quench_bin = _tw_cuml_lax_kern_vmap(ssfr.reshape(nhist * nt), -11.0, 0.2)
    weights_quench_bin = weights_quench_bin.reshape(nhist, nt)

    mstar = jnp.where(mstar > 0.0, jnp.log10(mstar), 0.0)
    sfr = jnp.where(sfr > 0.0, jnp.log10(sfr), 0.0)

    return compute_sumstats_MIX_p50(mstar, sfr, p50_sampled, weights_quench_bin, weight)


_A = (None, 0, 0, 0, *[None] * 9)
sumstats_sfh_MIX_p50_vmap = jjit(
    vmap(sumstats_sfh_MIX_p50, in_axes=_A), static_argnames=["n_histories"]
)


@jjit
def calc_hist_mstar_ssfr_kern(
    mstar_snap, ssfr_snap, ndsig, weight, ndbins_lo, ndbins_hi
):
    eps = 1e-10

    nddata = jnp.array([mstar_snap, ssfr_snap]).T
    counts = tw_ndhist_weighted(nddata, ndsig, weight, ndbins_lo, ndbins_hi)
    counts = jnp.clip(counts, eps, None)
    counts = counts / jnp.sum(counts)
    return counts


calc_hist_mstar_ssfr = jjit(
    vmap(calc_hist_mstar_ssfr_kern, in_axes=(0, 0, *[None] * 4))
)


@partial(jjit, static_argnames=["n_histories"])
def sumstats_sfh_with_hists(
    t_table,
    logmh,
    mah_params,
    p50,
    n_histories,
    ran_key,
    index_select,
    index_high,
    fstar_tdelay,
    ndsig,
    bins_LO,
    bins_HI,
    t_sel_hists,
    pdf_parameters_Q=DEFAULT_SFH_PDF_QUENCH_PARAMS,
    pdf_parameters_MS=DEFAULT_SFH_PDF_MAINSEQ_PARAMS,
    R_model_params_Q=DEFAULT_R_QUENCH_PARAMS,
    R_model_params_MS=DEFAULT_R_MAINSEQ_PARAMS,
):
    """
    Compute differentiable summary statistics from monte-carlo histories for
    a mixed population of quenched and main sequence galaxies.

    There is correlation with p50.

    Parameters
    ----------
    t_table : ndarray of shape (n_t, )
        Cosmic time array in Gyr.
    logmh : float
        Base-10 log of present-day halo mass of the halo population
    mah_params : ndarray of shape (n_mah_haloes x n_mah_params)
        Array with the diffmah parameters that will be marginalized over. Could
        be either individual fits of n_mah_haloes haloes, or be n_mah_haloes
        samples from a population model. They are chosen at random n_halos times.
    p50 : mah_params : ndarray of shape (n_mah_haloes, )
        Formation time percentile of each halo conditioned on halo mass.
    n_histories : int
        Number of SFH histories to generate by DiffstarPop.
    ran_key : ndarray of shape 2
        JAX random key.
    index_select: ndarray of shape (n_times_fstar, )
        Snapshot indices used in fstar computation
    index_high: ndarray of shape (n_times_fstar, )
        Indices of np.searchsorted(t, t - fstar_tdelay)[index_select]
    fstar_tdelay: float
        Time interval in Gyr for fstar definition.
        fstar = (mstar(t) - mstar(t-fstar_tdelay)) / fstar_tdelay[Gyr]
    pdf_model_params_Q : ndarray of shape (n_pdf, )
        Array containing the Diffstarpop parameters for the quenched population.
        Default is DEFAULT_SFH_PDF_QUENCH_PARAMS.
    pdf_model_params_MS : ndarray of shape (n_pdf, )
        Array containing the Diffstarpop parameters for the main sequence population.
        Default is DEFAULT_SFH_PDF_MAINSEQ_PARAMS.
    R_model_params_Q: ndarray of shape (n_R, )
        Array containing the Diffstarpop parameters for the correlation between
        diffstar and diffmah parameters for the quenched population.
    R_model_params_MS: ndarray of shape (n_R, )
        Array containing the Diffstarpop parameters for the correlation between
        diffstar and diffmah parameters for the main sequence population.

    Returns
    -------
    mean_sm : ndarray of shape (n_t, )
        Average log10 Stellar Mass.
    variance_sm : ndarray of shape (n_t, )
        Variance of log10 Stellar Mass.
    mean_fstar_MS : ndarray of shape (n_t_fstar, )
        Average fstar (average SFH within some timescale) for main sequence galaxies.
    mean_fstar_Q : ndarray of shape (n_t_fstar, )
        Average fstar (average SFH within some timescale) for quenched galaxies.
    variance_fstar_MS : ndarray of shape (n_t_fstar, )
        Variance of fstar (average SFH within some timescale) for MS galaxies.
    variance_fstar_Q : ndarray of shape (n_t_fstar, )
        Variance of fstar (average SFH within some timescale) for Q galaxies.
    quench_frac : ndarray of shape (n_t_fstar, )
        Fraction of quenched galaxies.
    """
    mstar, sfr, fstar, p50_sampled, weight = draw_sfh_MIX(
        t_table,
        logmh,
        mah_params,
        p50,
        n_histories,
        ran_key,
        index_select,
        index_high,
        fstar_tdelay,
        pdf_parameters_Q,
        pdf_parameters_MS,
        R_model_params_Q,
        R_model_params_MS,
    )

    # weights_quench_bin = jnp.where(ssfr > 1e-11, 1.0, 0.0)
    ssfr = jnp.where(mstar > 0.0, jnp.log10(sfr / mstar), -50.0)
    nhist, nt = ssfr.shape
    weights_quench_bin = _tw_cuml_lax_kern_vmap(ssfr.reshape(nhist * nt), -11.0, 0.2)
    weights_quench_bin = weights_quench_bin.reshape(nhist, nt)

    mstar = jnp.where(mstar > 0.0, jnp.log10(mstar), 0.0)
    sfr = jnp.where(sfr > 0.0, jnp.log10(sfr), 0.0)

    _res = compute_sumstats_MIX_p50(mstar, sfr, p50_sampled, weights_quench_bin, weight)
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

    counts = calc_hist_mstar_ssfr(
        mstar.T[t_sel_hists], ssfr.T[t_sel_hists], ndsig, weight, bins_LO, bins_HI
    )
    out = (
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
    return out


_A = (None, 0, 0, 0, *[None] * 13)
sumstats_sfh_with_hists_vmap = jjit(
    vmap(sumstats_sfh_with_hists, in_axes=_A), static_argnames=["n_histories"]
)


@partial(jjit, static_argnames=["n_histories"])
def sumstats_sfh_with_hists_scan(
    t_table,
    logmh_bins,
    mah_params_bins,
    p50_bins,
    n_histories,
    ran_key,
    index_select,
    index_high,
    fstar_tdelay,
    ndsig,
    bins_LO,
    bins_HI,
    t_sel_hists,
    pdf_parameters_Q=DEFAULT_SFH_PDF_QUENCH_PARAMS,
    pdf_parameters_MS=DEFAULT_SFH_PDF_MAINSEQ_PARAMS,
    R_model_params_Q=DEFAULT_R_QUENCH_PARAMS,
    R_model_params_MS=DEFAULT_R_MAINSEQ_PARAMS,
):
    nt = len(t_table)
    ngrid = len(bins_LO)
    nt_counts = len(t_sel_hists)
    init = (
        jnp.zeros((nt)),
        jnp.zeros((nt)),
        jnp.zeros((nt)),
        jnp.zeros((nt)),
        jnp.zeros((nt)),
        jnp.zeros((nt)),
        jnp.zeros((nt)),
        jnp.zeros((nt)),
        jnp.zeros((nt)),
        jnp.zeros((nt)),
        jnp.zeros((nt)),
        jnp.zeros((nt)),
        jnp.zeros((nt)),
        jnp.zeros((nt_counts, ngrid)),
    )

    @jjit
    def _testfun_scan(carry, data):
        logmh, mah_params, p50 = data
        _res = sumstats_sfh_with_hists(
            t_table,
            logmh,
            mah_params,
            p50,
            n_histories,
            ran_key,
            index_select,
            index_high,
            fstar_tdelay,
            ndsig,
            bins_LO,
            bins_HI,
            t_sel_hists,
            pdf_parameters_Q,
            pdf_parameters_MS,
            R_model_params_Q,
            R_model_params_MS,
        )
        return _res, _res

    data = (logmh_bins, mah_params_bins, p50_bins)
    result = lax.scan(_testfun_scan, init, data)

    return result[1]


@jjit
def calculate_sfh_MIX(
    t_table,
    mah_params,
    p50,
    ran_key,
    index_select,
    index_high,
    fstar_tdelay,
    pdf_parameters_Q=DEFAULT_SFH_PDF_QUENCH_PARAMS,
    pdf_parameters_MS=DEFAULT_SFH_PDF_MAINSEQ_PARAMS,
    R_model_params_Q=DEFAULT_R_QUENCH_PARAMS,
    R_model_params_MS=DEFAULT_R_MAINSEQ_PARAMS,
):
    """
    For each input halo, generate two Monte Carlos SFHs (quenched and main sequence).

    There is correlation with p50.

    Parameters
    ----------
    t_table : ndarray of shape (n_times, )
        Cosmic time array in Gyr.
    mah_params : ndarray of shape (n_mah_haloes x n_mah_params)
        Array with the diffmah parameters that will be marginalized over. Could
        be either individual fits of n_mah_haloes haloes, or be n_mah_haloes
        samples from a population model. They are chosen at random n_halos times.
    p50 : mah_params : ndarray of shape (n_mah_haloes, )
        Formation time (t50) percentile of each halo conditioned on halo mass.
    ran_key : ndarray of shape 2
        JAX random key.
    index_select: ndarray of shape (n_times_fstar, )
        Snapshot indices used in fstar computation
    index_high: ndarray of shape (n_times_fstar, )
        Indices of np.searchsorted(t, t - fstar_tdelay)[index_select]
    fstar_tdelay: float
        Time interval in Gyr for fstar definition.
        fstar = (mstar(t) - mstar(t-fstar_tdelay)) / fstar_tdelay[Gyr]
    pdf_model_params_Q : ndarray of shape (n_pdf, )
        Array containing the Diffstarpop parameters for the quenched population.
        Default is DEFAULT_SFH_PDF_QUENCH_PARAMS.
    pdf_model_params_MS : ndarray of shape (n_pdf, )
        Array containing the Diffstarpop parameters for the main sequence population.
        Default is DEFAULT_SFH_PDF_MAINSEQ_PARAMS.
    R_model_params_Q: ndarray of shape (n_R, )
        Array containing the Diffstarpop parameters for the correlation between
        diffstar and diffmah parameters for the quenched population.
    R_model_params_MS: ndarray of shape (n_R, )
        Array containing the Diffstarpop parameters for the correlation between
        diffstar and diffmah parameters for the main sequence population.

    Returns
    -------
    mstar : ndarray of shape (2 x n_mah_haloes, n_times)
        Stores cumulative stellar mass history in units of Msun/yr. Returns concatenation of
        quenched SFHs (n_mah_haloes, n_times) and MS SFHs (n_mah_haloes, n_times)
    sfr : ndarray of shape (2 x n_mah_haloes, n_times)
        Stores star formation rate history in units of Msun/yr. Returns concatenation of
        quenched SFHs (n_mah_haloes, n_times) and MS SFHs (n_mah_haloes, n_times)
    fstar : ndarray of shape (2 x n_mah_haloes, n_times_fstar)
        SFH averaged over timescale fstar_tdelay in units of Msun/yr assuming h=1. Returns
        concatenation of quenched SFHs (n_mah_haloes, n_times) and MS SFHs (n_mah_haloes, n_times)
    p50_sampled : ndarray of shape (2 x n_mah_haloes, )
        Concatenation of (p50, p50) input array.
    weight : ndarray of shape (2 x n_mah_haloes, )
        Concatenation mixing quench probabilities for eac SFH (f_Q, 1-f_Q), with f_Q being a
        ndarray of shape (n_mah_haloes, ).
    """
    lgt = jnp.log10(t_table)
    dt = _jax_get_dt_array(t_table)

    logmh = mah_params[:, 0]
    n_histories = len(mah_params)

    (quench_key, mainseq_key, ran_key) = jran.split(ran_key, 3)

    _res = get_smah_means_and_covs_mainseq(logmh, *pdf_parameters_MS)
    means_mainseq, covs_mainseq = _res

    R_vals_mainseq = _get_slopes_mainseq(logmh, *R_model_params_MS)
    R_vals_mainseq = jnp.array(R_vals_mainseq)
    shifts_mainseq = jnp.einsum("ph,h->hp", R_vals_mainseq, (p50 - 0.5))

    _res = get_smah_means_and_covs_quench(logmh, *pdf_parameters_Q)
    frac_quench, means_quench, covs_quench = _res

    _res = _get_slopes_quench(logmh, *R_model_params_Q)
    R_Fquench, R_vals_quench = _res[0], _res[1:]
    R_vals_quench = jnp.array(R_vals_quench)  # [:, 0]
    shifts_quench = jnp.einsum("ph,h->hp", R_vals_quench, (p50 - 0.5))
    shifts_Fquench = R_Fquench * (p50 - 0.5)
    fquench_x0 = pdf_parameters_Q[0] + shifts_Fquench
    frac_quench = frac_quench_vs_lgm0(logmh, fquench_x0, *pdf_parameters_Q[1:4])

    sfh_params_Q = jran.multivariate_normal(
        quench_key, means_quench, covs_quench, shape=(n_histories,)
    )
    sfh_params_Q = sfh_params_Q + shifts_quench

    sfr_params_Q = sfh_params_Q[:, 0:4]
    q_params_Q = sfh_params_Q[:, 4:8]

    sfr_params_MS = jran.multivariate_normal(
        mainseq_key, means_mainseq, covs_mainseq, shape=(n_histories,)
    )
    sfr_params_MS = sfr_params_MS + shifts_mainseq
    # q_params_MS = jnp.ones_like(q_params_Q) * 10.0

    mstar_Q, sfr_Q, fstar_Q = sm_sfr_history_diffstar_scan_XsfhXmah_vmap(
        t_table,
        lgt,
        dt,
        mah_params,
        sfr_params_Q,
        q_params_Q,
        index_select,
        index_high,
        fstar_tdelay,
    )
    mstar_MS, sfr_MS, fstar_MS = sm_sfr_history_diffstar_scan_MS_XsfhXmah_vmap(
        t_table,
        lgt,
        dt,
        mah_params,
        sfr_params_MS,
        index_select,
        index_high,
        fstar_tdelay,
    )
    mstar = jnp.concatenate((mstar_Q, mstar_MS))
    sfr = jnp.concatenate((sfr_Q, sfr_MS))
    fstar = jnp.concatenate((fstar_Q, fstar_MS))

    weight = jnp.concatenate(
        (
            frac_quench,
            (1.0 - frac_quench),
        )
    )
    p50_sampled = jnp.concatenate((p50, p50))
    return mstar, sfr, fstar, p50_sampled, weight


@jjit
def draw_single_sfh_MIX_with_exsitu(
    t_table,
    mah_params,
    p50,
    sfr_exsitu,
    ran_key,
    pdf_parameters_Q=DEFAULT_SFH_PDF_QUENCH_PARAMS,
    pdf_parameters_MS=DEFAULT_SFH_PDF_MAINSEQ_PARAMS,
    R_model_params_Q=DEFAULT_R_QUENCH_PARAMS,
    R_model_params_MS=DEFAULT_R_MAINSEQ_PARAMS,
):
    """
    Generate Monte Carlo realization of the star formation histories of
    a mixed population of quenched and main sequence galaxies
    for a single halo mass bin.

    There is correlation with p50.

    Parameters
    ----------
    t_table : ndarray of shape (n_times, )
        Cosmic time array in Gyr.
    logmh : float
        Base-10 log of present-day halo mass of the halo population
    mah_params : ndarray of shape (n_mah_haloes x n_mah_params)
        Array with the diffmah parameters that will be marginalized over. Could
        be either individual fits of n_mah_haloes haloes, or be n_mah_haloes
        samples from a population model. They are chosen at random n_halos times.
    p50 : mah_params : ndarray of shape (n_mah_haloes, )
        Formation time percentile of each halo conditioned on halo mass.
    ran_key : ndarray of shape 2
        JAX random key.
    pdf_model_params_Q : ndarray of shape (n_pdf, )
        Array containing the Diffstarpop parameters for the quenched population.
        Default is DEFAULT_SFH_PDF_QUENCH_PARAMS.
    pdf_model_params_MS : ndarray of shape (n_pdf, )
        Array containing the Diffstarpop parameters for the main sequence population.
        Default is DEFAULT_SFH_PDF_MAINSEQ_PARAMS.
    R_model_params_Q: ndarray of shape (n_R, )
        Array containing the Diffstarpop parameters for the correlation between
        diffstar and diffmah parameters for the quenched population.
    R_model_params_MS: ndarray of shape (n_R, )
        Array containing the Diffstarpop parameters for the correlation between
        diffstar and diffmah parameters for the main sequence population.

    Returns
    -------
    sfr : ndarray of shape (n_histories, n_times)
        Stores star formation rate history in units of Msun/yr.
    """
    logmh = jnp.atleast_1d(mah_params[0])

    (quench_key, mainseq_key, ran_key) = jran.split(ran_key, 3)

    _res = get_smah_means_and_covs_mainseq(logmh, *pdf_parameters_MS)
    means_mainseq, covs_mainseq = _res
    means_mainseq = means_mainseq[0]
    covs_mainseq = covs_mainseq[0]

    R_vals_mainseq = _get_slopes_mainseq(logmh, *R_model_params_MS)
    R_vals_mainseq = jnp.array(R_vals_mainseq)[:, 0]
    shifts_mainseq = jnp.einsum("p,h->hp", R_vals_mainseq, (p50 - 0.5))

    _res = get_smah_means_and_covs_quench(logmh, *pdf_parameters_Q)
    frac_quench, means_quench, covs_quench = _res
    frac_quench = frac_quench[0]
    means_quench = means_quench[0]
    covs_quench = covs_quench[0]

    _res = _get_slopes_quench(logmh, *R_model_params_Q)
    R_Fquench, R_vals_quench = _res[0], _res[1:]
    R_vals_quench = jnp.array(R_vals_quench)[:, 0]
    shifts_quench = jnp.einsum("p,h->hp", R_vals_quench, (p50 - 0.5))
    shifts_Fquench = R_Fquench * (p50 - 0.5)
    fquench_x0 = pdf_parameters_Q[0] + shifts_Fquench
    frac_quench = frac_quench_vs_lgm0(logmh, fquench_x0, *pdf_parameters_Q[1:4])

    sfh_params_Q = jran.multivariate_normal(
        quench_key, means_quench, covs_quench, shape=(1,)
    )
    sfh_params_Q = sfh_params_Q + shifts_quench

    sfr_params_Q = sfh_params_Q[:, 0:4]
    q_params_Q = sfh_params_Q[:, 4:8]

    sfr_params_MS = jran.multivariate_normal(
        mainseq_key, means_mainseq, covs_mainseq, shape=(1,)
    )
    sfr_params_MS = sfr_params_MS + shifts_mainseq
    # q_params_MS = jnp.ones_like(q_params_Q) * 10.0

    sfr_params_Q = sfr_params_Q[0]
    q_params_Q = q_params_Q[0]
    sfr_params_MS = sfr_params_MS[0]

    sfr_Q_insitu = sfr_history_diffstar_scan(
        t_table,
        mah_params,
        sfr_params_Q,
        q_params_Q,
    )
    sfr_MS_insitu = sfr_history_diffstar_scan_MS(
        t_table,
        mah_params,
        sfr_params_MS,
    )

    sfr_Q = sfr_Q_insitu + sfr_exsitu
    sfr_MS = sfr_MS + sfr_exsitu

    sfr = jnp.array([sfr_Q, sfr_MS])

    weight = jnp.array([frac_quench, (1.0 - frac_quench)])

    return sfr, weight


_A = (None, 0, 0, 0, 0, 0, *[None] * 4)
_draw_single_sfh_MIX_with_exsitu_vmap = jjit(
    vmap(draw_single_sfh_MIX_with_exsitu, in_axes=_A)
)


def draw_single_sfh_MIX_with_exsitu_vmap(
    t_table,
    mah_params_arr,
    p50_arr,
    sfr_exsitu_arr,
    ran_key_arr,
    pdf_parameters_Q=DEFAULT_SFH_PDF_QUENCH_PARAMS,
    pdf_parameters_MS=DEFAULT_SFH_PDF_MAINSEQ_PARAMS,
    R_model_params_Q=DEFAULT_R_QUENCH_PARAMS,
    R_model_params_MS=DEFAULT_R_MAINSEQ_PARAMS,
):
    """
    Calculate a Monte Carlo pair realization of the Quenched and Main Sequence
    star formation history for an array of galaxies.

    There is correlation with p50.

    Parameters
    ----------
    t_table : ndarray of shape (n_times, )
        Cosmic time array in Gyr.
    mah_params_arr : ndarray of shape (ngal x 4)
        Array with the diffmah parameters (logm, logtc, early, late)
    p50_arr : mah_params : ndarray of shape (ngal, )
        Formation time percentile of each halo conditioned on halo mass.
    sfr_exsitu_arr : ndarray of shape (ngal, n_times)
        Ex-situ contribution to the star formation history.
    ran_key_arr : ndarray of shape (ngal, )
        JAX random keys.
    pdf_model_params_Q : ndarray of shape (n_pdf, )
        Array containing the Diffstarpop parameters for the quenched population.
        Default is DEFAULT_SFH_PDF_QUENCH_PARAMS.
    pdf_model_params_MS : ndarray of shape (n_pdf, )
        Array containing the Diffstarpop parameters for the main sequence population.
        Default is DEFAULT_SFH_PDF_MAINSEQ_PARAMS.
    R_model_params_Q: ndarray of shape (n_R, )
        Array containing the Diffstarpop parameters for the correlation between
        diffstar and diffmah parameters for the quenched population.
    R_model_params_MS: ndarray of shape (n_R, )
        Array containing the Diffstarpop parameters for the correlation between
        diffstar and diffmah parameters for the main sequence population.

    Returns
    -------
    gal_sfr_arr : ndarray of shape (2 x ngal, n_times)
        Stores star formation rate history in units of Msun/yr. The array is composed by the
        concatenation of two arrays of shape (ngal, n_times) for the quenched and
        main sequence draws for each galaxy, so that sfr_Q = sfr[:ngal] and sfr_MS = sfr[ngal:].
    weight : ndarray of shape (2 x ngal, )
        Array with the mixing weights for the quenched and main sequence SFHs for each galaxy.
        It is the result of concatenating two arrays of shape (ngal, ) for the quenched and the
        main sequence weights.
    """
    gal_sfr_arr, weight = _draw_single_sfh_MIX_with_exsitu_vmap(
        t_table,
        mah_params_arr,
        p50_arr,
        sfr_exsitu_arr,
        ran_key_arr,
        pdf_parameters_Q,
        pdf_parameters_MS,
        R_model_params_Q,
        R_model_params_MS,
    )

    weight = jnp.concatenate(weight.T)
    ng = len(p50_arr)
    nt = len(t_table)
    # sfhs are concatenated by vmap as (q_0, ms_0, q_1, ms_1, ...)
    # reshaping so that (q_0, q_1, ..., ms_0, ms_1, ...)
    gal_sfr_arr = gal_sfr_arr.reshape((ng, 2, nt))
    gal_sfr_arr = gal_sfr_arr.swapaxes(0, 1)
    gal_sfr_arr = jnp.concatenate(gal_sfr_arr, axis=0)
    return gal_sfr_arr, weight
