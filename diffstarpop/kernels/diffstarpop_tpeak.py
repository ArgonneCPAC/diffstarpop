"""
"""

from diffstar import DiffstarUParams, MSUParams, QUParams
from diffstar.defaults import DEFAULT_DIFFSTAR_U_PARAMS, DEFAULT_Q_U_PARAMS_UNQUENCHED
from jax import jit as jjit
from jax import numpy as jnp
from jax import random as jran

from .satquenchpop_model import get_qprob_sat
from .sfh_pdf_tpeak import _sfh_pdf_scalar_kernel


@jjit
def mc_diffstar_u_params_singlegal_kernel(
    diffstarpop_params,
    mah_params,
    t_peak,
    lgmu_infall,
    logmhost_infall,
    gyr_since_infall,
    ran_key,
):
    means_covs = _diffstarpop_means_covs(
        diffstarpop_params,
        mah_params,
        lgmu_infall,
        logmhost_infall,
        gyr_since_infall,
    )

    u_indx_hi = DEFAULT_DIFFSTAR_U_PARAMS.u_ms_params.u_indx_hi

    frac_quench = means_covs[0]
    mu_mseq = means_covs[1]
    q_key_ms_block, q_key_q_block, frac_q_key = jran.split(ran_key, 3)
    mu_qseq_ms_block, cov_qseq_ms_block = means_covs[2:4]
    mu_qseq_q_block, cov_qseq_q_block = means_covs[4:]

    u_params_qseq_ms_block = jran.multivariate_normal(
        q_key_ms_block, jnp.array(mu_qseq_ms_block), cov_qseq_ms_block, shape=()
    )
    u_params_qseq_q_block = jran.multivariate_normal(
        q_key_q_block, jnp.array(mu_qseq_q_block), cov_qseq_q_block, shape=()
    )
    u_params_q = jnp.array(
        (
            *u_params_qseq_ms_block[:3],
            u_indx_hi,
            u_params_qseq_ms_block[3],
            *u_params_qseq_q_block,
        )
    )
    u_params_q = DiffstarUParams(MSUParams(*u_params_q[:5]), QUParams(*u_params_q[5:]))

    # Do not use a new MC realization for the main sequence
    # Instead, shift the MC draws of the quenched sequence to reduce noise
    delta_mu_mseq = jnp.array([x - y for x, y in zip(mu_mseq, mu_qseq_ms_block)])
    u_params_ms_ms_block = u_params_qseq_ms_block + delta_mu_mseq
    u_params_ms = jnp.array(
        (
            *u_params_ms_ms_block[:3],
            u_indx_hi,
            u_params_ms_ms_block[3],
            *DEFAULT_Q_U_PARAMS_UNQUENCHED,
        )
    )
    u_params_ms = DiffstarUParams(
        MSUParams(*u_params_ms[:5]), QUParams(*u_params_ms[5:])
    )

    uran = jran.uniform(frac_q_key, minval=0, maxval=1, shape=())
    mc_is_quenched_sequence = uran < frac_quench

    return u_params_ms, u_params_q, frac_quench, mc_is_quenched_sequence


@jjit
def _diffstarpop_means_covs(
    diffstarpop_params,
    mah_params,
    lgmu_infall,
    logmhost_infall,
    gyr_since_infall,
):
    lgm0 = mah_params[0]
    means_covs = _sfh_pdf_scalar_kernel(diffstarpop_params.sfh_pdf_cens_params, lgm0)

    # Modify frac_q for satellites
    frac_q = means_covs[0]
    frac_q = get_qprob_sat(
        diffstarpop_params.satquench_params,
        lgmu_infall,
        logmhost_infall,
        gyr_since_infall,
        frac_q,
    )
    means_covs = (frac_q, *means_covs[1:])
    return means_covs
