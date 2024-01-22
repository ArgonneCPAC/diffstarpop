"""This module implements kernels for Monte Carlo generating Diffstar SFHs
"""
from diffstar import (
    DiffstarParams,
    calc_sfh_galpop,
    calc_sfh_singlegal,
    get_bounded_diffstar_params,
)
from diffstar.defaults import FB, LGT0
from jax import jit as jjit
from jax import random as jran
from jax import vmap

from .kernels.kernel_wrapper import mc_diffstar_u_params_singlegal_kernel


@jjit
def mc_diffstar_sfh_singlegal(
    diffstarpop_params,
    mah_params,
    p50,
    lgmu_infall,
    logmhost_infall,
    gyr_since_infall,
    ran_key,
    tarr,
    lgt0=LGT0,
    fb=FB,
):
    """Monte Carlo realization of a single point in Diffstar parameter space,
    along with the computation of SFH for this point.

    Parameters
    ----------
    diffstarpop_params : namedtuple
        See defaults.DEFAULT_DIFFSTARPOP_PARAMS for an example

    mah_params : namedtuple, length 4
        mah_params is a tuple of floats
        DiffmahParams = logmp, logtc, early_index, late_index

    p50 : float
        Prob(<t_50% | logm0), the CDF of the distribution of halo
        formation times t_50% conditioned on mass logm0

    lgmu_infall : float
        Base-10 log of ratio Msub(t_infall)/Mhost(t_infall)
        Set to 0.0 for centrals

    logmhost_infall : float
        Base-10 log of Mhost(t_infall)
        Set to 0.0 for centrals

    gyr_since_infall : float
        Time since infall in Gyr
        Set to -100.0 for centrals

    ran_key : jax.random.PRNGKey
        Single instance of a jax random seed

    tarr : ndarray, shape (nt, )

    lgt0 : float, optional
        Base-10 log of the z=0 age of the Universe in Gyr
        Default is set in diffstar.defaults
        This variable should be self-consistently set with cosmology

    fb : float, optional
        Cosmic baryon fraction Ob0/Om0
        Default is set in diffstar.defaults
        This variable should be self-consistently set with cosmology

    Returns
    -------
    diffstar_params : namedtuple
        DiffstarParams = ms_params, q_params
            ms_params and q_params are tuples of floats
            diffstar_params.ms_params = lgmcrit, lgy_at_mcrit, indx_lo, indx_hi, tau_dep
            diffstar_params.q_params = lg_qt, qlglgdt, lg_drop, lg_rejuv

    sfh : ndarray, shape (nt, )
        Star formation rate in units of Msun/yr

    """
    diffstar_params = mc_diffstar_params_singlegal(
        diffstarpop_params,
        mah_params,
        p50,
        lgmu_infall,
        logmhost_infall,
        gyr_since_infall,
        ran_key,
    )
    sfh = calc_sfh_singlegal(diffstar_params, mah_params, tarr, lgt0=lgt0, fb=fb)
    return diffstar_params, sfh


@jjit
def mc_diffstar_params_singlegal(
    diffstarpop_params,
    mah_params,
    p50,
    lgmu_infall,
    logmhost_infall,
    gyr_since_infall,
    ran_key,
):
    """Monte Carlo realization of a single point in Diffstar parameter space.

    Parameters
    ----------
    diffstarpop_params : namedtuple
        See defaults.DEFAULT_DIFFSTARPOP_PARAMS for an example

    mah_params : namedtuple, length 4
        mah_params is a tuple of floats
        DiffmahParams = logmp, logtc, early_index, late_index

    p50 : float
        Prob(<t_50% | logm0), the CDF of the distribution of halo
        formation times t_50% conditioned on mass logm0

    lgmu_infall : float
        Base-10 log of ratio Msub(t_infall)/Mhost(t_infall)
        Set to 0.0 for centrals

    logmhost_infall : float
        Base-10 log of Mhost(t_infall)
        Set to 0.0 for centrals

    gyr_since_infall : float
        Time since infall in Gyr
        Set to -100.0 for centrals

    ran_key : jax.random.PRNGKey
        Single instance of a jax random seed

    Returns
    -------
    diffstar_params : namedtuple
        DiffstarParams = ms_params, q_params
            ms_params and q_params are tuples of floats
            diffstar_params.ms_params = lgmcrit, lgy_at_mcrit, indx_lo, indx_hi, tau_dep
            diffstar_params.q_params = lg_qt, qlglgdt, lg_drop, lg_rejuv

    """
    diffstar_u_params = mc_diffstar_u_params_singlegal(
        diffstarpop_params,
        mah_params,
        p50,
        lgmu_infall,
        logmhost_infall,
        gyr_since_infall,
        ran_key,
    )
    diffstar_params = get_bounded_diffstar_params(diffstar_u_params)
    return DiffstarParams(*diffstar_params)


@jjit
def mc_diffstar_u_params_singlegal(
    diffstarpop_params,
    mah_params,
    p50,
    lgmu_infall,
    logmhost_infall,
    gyr_since_infall,
    ran_key,
):
    """"""
    args = (
        *diffstarpop_params,
        mah_params,
        p50,
        lgmu_infall,
        logmhost_infall,
        gyr_since_infall,
        ran_key,
    )
    diffstar_u_params = mc_diffstar_u_params_singlegal_kernel(*args)
    return diffstar_u_params


_POP = (None, 0, 0, 0, 0, 0, 0)
mc_diffstar_u_params_galpop_kernel = jjit(
    vmap(mc_diffstar_u_params_singlegal, in_axes=_POP)
)


@jjit
def mc_diffstar_u_params_galpop(
    diffstarpop_params,
    mah_params,
    p50,
    lgmu_infall,
    logmhost_infall,
    gyr_since_infall,
    ran_key,
):
    """"""
    ngals = p50.size
    ran_keys = jran.split(ran_key, ngals)
    diffstar_u_params = mc_diffstar_u_params_galpop_kernel(
        diffstarpop_params,
        mah_params,
        p50,
        lgmu_infall,
        logmhost_infall,
        gyr_since_infall,
        ran_keys,
    )
    return diffstar_u_params


get_bounded_diffstar_params_galpop = jjit(vmap(get_bounded_diffstar_params, in_axes=0))


@jjit
def mc_diffstar_params_galpop(
    diffstarpop_params,
    mah_params,
    p50,
    lgmu_infall,
    logmhost_infall,
    gyr_since_infall,
    ran_key,
):
    """Monte Carlo realization of a population of points in Diffstar parameter space.

    Parameters
    ----------
    diffstarpop_params : namedtuple
        See defaults.DEFAULT_DIFFSTARPOP_PARAMS for an example

    mah_params : namedtuple, length 4
        mah_params is a tuple of ndarrays of shape (ngals, )
        DiffmahParams = logmp, logtc, early_index, late_index

    p50 : ndarray of shape (ngals, )
        Prob(<t_50% | logm0), the CDF of the distribution of halo
        formation times t_50% conditioned on mass logm0

    lgmu_infall : ndarray of shape (ngals, )
        Base-10 log of ratio Msub(t_infall)/Mhost(t_infall)
        Set to 0.0 for centrals

    logmhost_infall : ndarray of shape (ngals, )
        Base-10 log of Mhost(t_infall)
        Set to 0.0 for centrals

    gyr_since_infall : ndarray of shape (ngals, )
        Time since infall in Gyr
        Set to -100.0 for centrals

    ran_key : jax.random.PRNGKey
        Single instance of a jax random seed

    Returns
    -------
    diffstar_params : namedtuple
        DiffstarParams = ms_params, q_params
            ms_params and q_params are tuples of floats
            diffstar_params.ms_params = lgmcrit, lgy_at_mcrit, indx_lo, indx_hi, tau_dep
            diffstar_params.q_params = lg_qt, qlglgdt, lg_drop, lg_rejuv

    """
    diffstar_u_params = mc_diffstar_u_params_galpop(
        diffstarpop_params,
        mah_params,
        p50,
        lgmu_infall,
        logmhost_infall,
        gyr_since_infall,
        ran_key,
    )
    diffstar_params = get_bounded_diffstar_params_galpop(diffstar_u_params)
    return diffstar_params


@jjit
def mc_diffstar_sfh_galpop(
    diffstarpop_params,
    mah_params,
    p50,
    lgmu_infall,
    logmhost_infall,
    gyr_since_infall,
    ran_key,
    tarr,
    lgt0=LGT0,
    fb=FB,
):
    """Monte Carlo realization of a single point in Diffstar parameter space,
    along with the computation of SFH for this point.

    Parameters
    ----------
    diffstarpop_params : namedtuple
        See defaults.DEFAULT_DIFFSTARPOP_PARAMS for an example

    mah_params : namedtuple, length 4
        mah_params is a tuple of ndarrays of shape (ngals, )
        DiffmahParams = logmp, logtc, early_index, late_index

    p50 : ndarray of shape (ngals, )
        Prob(<t_50% | logm0), the CDF of the distribution of halo
        formation times t_50% conditioned on mass logm0

    lgmu_infall : ndarray of shape (ngals, )
        Base-10 log of ratio Msub(t_infall)/Mhost(t_infall)
        Set to 0.0 for centrals

    logmhost_infall : ndarray of shape (ngals, )
        Base-10 log of Mhost(t_infall)
        Set to 0.0 for centrals

    gyr_since_infall : ndarray of shape (ngals, )
        Time since infall in Gyr
        Set to -100.0 for centrals

    ran_key : jax.random.PRNGKey
        Single instance of a jax random seed

    tarr : ndarray, shape (nt, )

    lgt0 : float, optional
        Base-10 log of the z=0 age of the Universe in Gyr
        Default is set in diffstar.defaults
        This variable should be self-consistently set with cosmology

    fb : float, optional
        Cosmic baryon fraction Ob0/Om0
        Default is set in diffstar.defaults
        This variable should be self-consistently set with cosmology

    Returns
    -------
    diffstar_params : namedtuple
        DiffstarParams = ms_params, q_params
            ms_params and q_params are tuples of ndarrays of shape (ngals, )
            diffstar_params.ms_params = lgmcrit, lgy_at_mcrit, indx_lo, indx_hi, tau_dep
            diffstar_params.q_params = lg_qt, qlglgdt, lg_drop, lg_rejuv

    sfh : ndarray, shape (ngals, nt)
        Star formation rate in units of Msun/yr

    """
    diffstar_params = mc_diffstar_params_galpop(
        diffstarpop_params,
        mah_params,
        p50,
        lgmu_infall,
        logmhost_infall,
        gyr_since_infall,
        ran_key,
    )
    sfh = calc_sfh_galpop(diffstar_params, mah_params, tarr, lgt0=lgt0, fb=fb)
    return diffstar_params, sfh
