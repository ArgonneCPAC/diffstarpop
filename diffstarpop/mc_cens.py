"""This module implements kernels for Monte Carlo generating Diffstar SFHs
"""

from diffstar import DiffstarParams, calc_sfh_singlegal, get_bounded_diffstar_params
from diffstar.defaults import FB, LGT0
from jax import jit as jjit

from .kernels.kernel_wrapper_cens import mc_diffstar_u_params_singlecen_kernel


@jjit
def mc_diffstar_sfh_singlecen(
    diffstarpop_params, mah_params, p50, ran_key, tarr, lgt0=LGT0, fb=FB
):
    diffstar_params_q, diffstar_params_ms, frac_q = mc_diffstar_params_singlecen(
        diffstarpop_params,
        mah_params,
        p50,
        ran_key,
    )
    sfh_q = calc_sfh_singlegal(diffstar_params_q, mah_params, tarr, lgt0=lgt0, fb=fb)
    sfh_ms = calc_sfh_singlegal(diffstar_params_ms, mah_params, tarr, lgt0=lgt0, fb=fb)
    return diffstar_params_q, diffstar_params_ms, sfh_q, sfh_ms, frac_q


@jjit
def mc_diffstar_params_singlecen(diffstarpop_params, mah_params, p50, ran_key):
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
    diffstar_params_q : namedtuple
        Diffstar params for quenched galaxy
        DiffstarParams = ms_params, q_params
            ms_params and q_params are tuples of floats
            diffstar_params.ms_params = lgmcrit, lgy_at_mcrit, indx_lo, indx_hi, tau_dep
            diffstar_params.q_params = lg_qt, qlglgdt, lg_drop, lg_rejuv

    diffstar_params_ms : namedtuple
        Diffstar params for main sequence galaxy
        DiffstarParams = ms_params, q_params
            ms_params and q_params are tuples of floats
            diffstar_params.ms_params = lgmcrit, lgy_at_mcrit, indx_lo, indx_hi, tau_dep
            diffstar_params.q_params = lg_qt, qlglgdt, lg_drop, lg_rejuv

    frac_q : float
        Quenched fraction.

    """
    _res = mc_diffstar_u_params_singlecen_kernel(
        diffstarpop_params.sfh_pdf_mainseq_params,
        diffstarpop_params.sfh_pdf_quench_params,
        diffstarpop_params.assembias_mainseq_params,
        diffstarpop_params.assembias_quench_params,
        mah_params,
        p50,
        ran_key,
    )
    diffstar_u_params_q, diffstar_u_params_ms, frac_q = _res
    diffstar_params_q = get_bounded_diffstar_params(diffstar_u_params_q)
    diffstar_params_ms = get_bounded_diffstar_params(diffstar_u_params_ms)
    diffstar_params_q = DiffstarParams(*diffstar_params_q)
    diffstar_params_ms = DiffstarParams(*diffstar_params_ms)
    return diffstar_params_q, diffstar_params_ms, frac_q
