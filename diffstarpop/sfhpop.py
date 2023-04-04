"""
"""
import numpy as np
from collections import OrderedDict
from jax import jit as jjit
from jax import numpy as jnp
from jax import random as jran
from jax import vmap
from functools import partial
from .pdf_mainseq import _get_mean_smah_params_mainseq
from .pdf_mainseq import _get_cov_params_mainseq
from .pdf_mainseq import _get_cov_scalar
from diffstar.stars import DEFAULT_SFR_PARAMS as DEFAULT_SFR_PARAMS_DICT
from diffstar.stars import _get_unbounded_sfr_params
from diffstar.quenching import DEFAULT_Q_PARAMS as DEFAULT_Q_PARAMS_DICT
from diffstar.quenching import _get_unbounded_q_params
from diffstar.main_sequence import get_ms_sfh_from_mah_kern
from diffstar.utils import _jax_get_dt_array
from dsps.sed.stellar_age_weights import _calc_age_weights_from_logsm_table
from dsps.experimental.diffburst import _burst_age_weights_pop


get_ms_sfh_scan_tobs_lgm0 = get_ms_sfh_from_mah_kern(
    tobs_loop="scan", galpop_loop="vmap"
)

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


SFR_MIN = 1e-12


@jjit
def _integrate_sfr(sfr, dt):
    """Calculate the cumulative stellar mass history."""
    return jnp.cumsum(sfr * dt) * 1e9


_integrate_sfrpop = jjit(vmap(_integrate_sfr, in_axes=[0, None]))


_A = (None, 0, None, None)
_get_age_weights_from_tables_pop = jjit(
    vmap(_calc_age_weights_from_logsm_table, in_axes=_A)
)


@jjit
def _get_stellar_age_distributions(log_age_gyr, lgtarr_gyr, tobs_gyr, logsmh_pop):
    age_distributions = _get_age_weights_from_tables_pop(
        lgtarr_gyr, logsmh_pop, log_age_gyr, tobs_gyr
    )[1]

    return age_distributions


@jjit
def _get_cov_scalar2(cov_params):
    return _get_cov_scalar(*cov_params)


_get_cov_vmap = jjit(vmap(_get_cov_scalar2))


@jjit
def _ms_means_and_covs_lgm0(lgm0):
    means_ms = jnp.array(_get_mean_smah_params_mainseq(lgm0))

    cov_params_ms = _get_cov_params_mainseq(lgm0)
    cov_ms = _get_cov_scalar(*cov_params_ms)

    return means_ms, cov_ms


@jjit
def _ms_means_and_covs_lgmpop(lgmpop):
    means_ms_pop = jnp.array(_get_mean_smah_params_mainseq(lgmpop)).T

    cov_params_ms = _get_cov_params_mainseq(lgmpop)
    cov_ms = _get_cov_vmap(cov_params_ms)

    return means_ms_pop, cov_ms


@partial(jjit, static_argnames=["n_gals"])
def _mc_ms_u_params_lgm0(ran_key, means_ms, cov_ms, n_gals):
    ms_params = jran.multivariate_normal(ran_key, means_ms, cov_ms, shape=(n_gals,))
    ulgm = ms_params[:, 0]
    ulgy = ms_params[:, 1]
    ul = ms_params[:, 2]
    utau = ms_params[:, 3]

    zz = jnp.zeros(n_gals)
    uh = jnp.zeros(n_gals) + UH

    ms_u_params = jnp.array((ulgm, ulgy, ul, uh, utau)).T
    q_u_params = jnp.array([zz + p for p in DEFAULT_UNBOUND_Q_PARAMS_MAIN_SEQ]).T

    return ms_u_params, q_u_params


@jjit
def _mc_ms_u_params_lgmpop(ran_key, means_ms_pop, cov_ms_pop):
    ms_params = jran.multivariate_normal(ran_key, means_ms_pop, cov_ms_pop)
    ulgm = ms_params[:, 0]
    ulgy = ms_params[:, 1]
    ul = ms_params[:, 2]
    utau = ms_params[:, 3]

    zz = 0.0 * ulgm
    uh = zz + UH

    ms_u_params = jnp.array((ulgm, ulgy, ul, uh, utau)).T
    q_u_params = jnp.array([zz + p for p in DEFAULT_UNBOUND_Q_PARAMS_MAIN_SEQ]).T

    return ms_u_params, q_u_params


@partial(jjit, static_argnames=["n_gals"])
def mc_galhalo_ms_lgm0(ran_key, mah_params_pop, tarr, n_gals):
    lgm0 = mah_params_pop[0, 0]
    means_ms, cov_ms = _ms_means_and_covs_lgm0(lgm0)

    _res = _mc_ms_u_params_lgm0(ran_key, means_ms, cov_ms, n_gals)
    ms_u_params_pop, q_u_params_pop = _res

    ms_sfh_pop = get_ms_sfh_scan_tobs_lgm0(tarr, mah_params_pop, ms_u_params_pop)
    ms_sfh_pop = jnp.where(ms_sfh_pop < SFR_MIN, SFR_MIN, ms_sfh_pop)

    dtarr = _jax_get_dt_array(tarr)
    ms_smh_pop = _integrate_sfrpop(ms_sfh_pop, dtarr)
    ms_logsmh_pop = jnp.log10(ms_smh_pop)

    return ms_u_params_pop, q_u_params_pop, ms_sfh_pop, ms_logsmh_pop


@jjit
def mc_galhalo_ms_lgmpop(ran_key, mah_params_pop, tarr):
    lgmpop = mah_params_pop[:, 0]
    means_ms_pop, cov_ms_pop = _ms_means_and_covs_lgmpop(lgmpop)

    _res = _mc_ms_u_params_lgmpop(ran_key, means_ms_pop, cov_ms_pop)
    ms_u_params_pop, q_u_params_pop = _res

    ms_sfh_pop = get_ms_sfh_scan_tobs_lgm0(tarr, mah_params_pop, ms_u_params_pop)
    ms_sfh_pop = jnp.where(ms_sfh_pop < SFR_MIN, SFR_MIN, ms_sfh_pop)

    dtarr = _jax_get_dt_array(tarr)
    ms_smh_pop = _integrate_sfrpop(ms_sfh_pop, dtarr)
    ms_logsmh_pop = jnp.log10(ms_smh_pop)

    return ms_u_params_pop, q_u_params_pop, ms_sfh_pop, ms_logsmh_pop


@jjit
def mc_age_weights_ms_lgmpop(ran_key, mah_params_pop, tarr_gyr, log_age_gyr, tobs_gyr):
    _res = mc_galhalo_ms_lgmpop(ran_key, mah_params_pop, tarr_gyr)
    ms_u_params_pop, q_u_params_pop, ms_sfh_pop, ms_logsmh_pop = _res

    lgtarr_gyr = jnp.log10(tarr_gyr)
    age_pdfs_pop = _get_stellar_age_distributions(
        log_age_gyr, lgtarr_gyr, tobs_gyr, ms_logsmh_pop
    )
    return ms_u_params_pop, q_u_params_pop, ms_sfh_pop, ms_logsmh_pop, age_pdfs_pop


@partial(jjit, static_argnames=["n_gals"])
def mc_age_weights_burstpop_lgm0(ran_key, burstpop_params, n_gals, log_age_gyr):
    lgfburst_min, lgfburst_max, dburst_min, dburst_max = burstpop_params
    lgf_key, dburst_key = jran.split(ran_key, 2)
    fburst_pop = 10 ** jran.uniform(
        lgf_key, minval=lgfburst_min, maxval=lgfburst_max, shape=(n_gals,)
    )
    dburst_pop = jran.uniform(
        dburst_key, minval=dburst_min, maxval=dburst_max, shape=(n_gals,)
    )
    log_age_yr = log_age_gyr + 9.0
    burst_age_pdf_pop = _burst_age_weights_pop(log_age_yr, dburst_pop)
    return fburst_pop, dburst_pop, burst_age_pdf_pop
