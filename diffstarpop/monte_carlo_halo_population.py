import numpy as np
from numpy.random import RandomState
from jax import vmap
from jax import jit as jjit
from collections import OrderedDict

from diffstar.stars import (
    calculate_sm_sfr_fstar_history_from_mah,
    DEFAULT_SFR_PARAMS as DEFAULT_SFR_PARAMS_DICT,
    _get_unbounded_sfr_params,
    jax_np_interp,
)
from diffmah.individual_halo_assembly import _calc_halo_history

from .pdf_quenched import get_smah_means_and_covs_quench

# from .pdf_model_assembly_bias_shifts import _get_shift_to_PDF_mean, _get_slopes

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

UH = DEFAULT_UNBOUND_SFR_PARAMS_DICT["indx_hi"]


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


def mc_sfh_population(
    lgt,
    dt,
    logmh,
    mah_params,
    n_haloes,
    index_select,
    index_high,
    fstar_tdelay,
    seed=0,
    pdf_parameters={},
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

    sampled_mahs_inds = RandomState(seed).choice(n_mah, n_haloes, replace=True)
    mah_params_sampled = mah_params[sampled_mahs_inds]

    _res = get_smah_means_and_covs_quench(logmh, **pdf_parameters)
    frac_quench, means_quench, covs_quench = _res
    frac_quench = frac_quench[0]
    means_quench = means_quench[0]
    covs_quench = covs_quench[0]

    sfh_params = RandomState(seed + 1).multivariate_normal(
        means_quench, covs_quench, size=n_haloes
    )
    sfr_params = np.zeros((n_haloes, 5))
    sfr_params[:, :3] = sfh_params[:, :3]
    sfr_params[:, 3] = UH
    sfr_params[:, 4] = sfh_params[:, 3]
    q_params = sfh_params[:, 4:8]

    _res = calculate_sm(
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


def mc_sfh_population_mah_correlation(
    lgt,
    dt,
    logmh,
    mah_params,
    n_haloes,
    index_select,
    index_high,
    fstar_tdelay,
    sfh_correlation_param_id,
    correlation,
    seed=0,
    **kwargs
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

    sampled_mahs_inds = RandomState(seed).choice(n_mah, n_haloes, replace=True)
    mah_params_sampled = mah_params[sampled_mahs_inds]

    _res = get_smah_means_and_covs_quench(logmh, **kwargs)
    frac_quench, means_quench, covs_quench = _res
    frac_quench = frac_quench[0]
    means_quench = means_quench[0]
    covs_quench = covs_quench[0]

    sfh_params = RandomState(seed + 1).multivariate_normal(
        means_quench, covs_quench, size=n_haloes
    )

    sfh_correlation_param = sfh_params[:, sfh_correlation_param_id]

    dmhdt, log_mah = _calc_halo_history_vmap(lgt, *mah_params_sampled.T)

    mhalo = np.array(10 ** log_mah)

    _index_high_50 = return_searchsorted_like_results(mhalo, 0.5)
    t50 = jax_np_interp_vmap(mhalo[:, -1] * 0.5, mhalo, 10 ** lgt, _index_high_50)

    argsort_mah = np.argsort(t50)
    mah_params_sampled = mah_params_sampled[argsort_mah]

    argsort_sfh = np.argsort(sfh_correlation_param)
    if correlation == 1:
        sfh_params = sfh_params[argsort_sfh]
    elif correlation == -1:
        sfh_params = sfh_params[argsort_sfh[::-1]]
    elif correlation == 0:
        sfh_params = sfh_params[argsort_mah]
    else:
        raise ValueError("Input correlation should be -1, 0, or 1")

    sfr_params = np.zeros((n_haloes, 6))
    sfr_params[:, :2] = sfh_params[:, :2]
    sfr_params[:, 2] = UK
    sfr_params[:, 3:] = sfh_params[:, 2:5]
    q_params = sfh_params[:, 5:9]

    _res = calculate_sm(
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


def mc_sfh_population_R(
    lgt,
    dt,
    logmh,
    mah_params,
    n_haloes,
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

    sampled_mahs_inds = RandomState(seed).choice(n_mah, n_haloes, replace=True)
    mah_params_sampled = mah_params[sampled_mahs_inds]
    p50_sampled = p50[sampled_mahs_inds]

    _res = get_smah_means_and_covs_quench(logmh, **kwargs_pdf)
    frac_quench, means_quench, covs_quench = _res
    frac_quench = frac_quench[0]
    means_quench = means_quench[0]
    covs_quench = covs_quench[0]

    Rvals = np.array(_get_slopes(logmh, **kwargs_R))[:, 0]
    shifts = _get_shift_to_PDF_mean(p50_sampled, Rvals)

    sfh_params = np.zeros((n_haloes, 9))

    means_quench_shifted = means_quench[:, None] + shifts
    for i in range(n_haloes):

        sfh_params[i] = RandomState(seed + 1 + i).multivariate_normal(
            means_quench_shifted[:, i], covs_quench, size=1
        )
    sfr_params = np.zeros((n_haloes, 6))
    sfr_params[:, :2] = sfh_params[:, :2]
    sfr_params[:, 2] = UK
    sfr_params[:, 3:] = sfh_params[:, 2:5]
    q_params = sfh_params[:, 5:9]

    _res = calculate_sm(
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
