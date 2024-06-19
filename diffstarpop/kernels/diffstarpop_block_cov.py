"""
"""

from diffstar import DiffstarUParams, MSUParams, QUParams
from diffstar.defaults import DEFAULT_DIFFSTAR_U_PARAMS, DEFAULT_Q_U_PARAMS_UNQUENCHED
from jax import jit as jjit
from jax import numpy as jnp
from jax import random as jran

from .mainseq_massonly import (
    MainseqMassOnlyParams,
    _get_cov_mainseq,
    _get_mean_u_params_mainseq,
)
from .sfh_pdf_block_cov import _qseq_pdf_scalar_kernel


@jjit
def mc_diffstar_u_params_singlegal_kernel(ms_params, qseq_params, mah_params, ran_key):
    means_covs = _diffstarpop_means_covs(ms_params, qseq_params, mah_params)
    ms_means_covs, qseq_means_covs = means_covs

    # Main sequence
    mu_ms, cov_ms = ms_means_covs
    ms_key, q_key, frac_q_key = jran.split(ran_key, 3)
    u_ms_params_ms_no_indx_hi = jran.multivariate_normal(
        ms_key, jnp.array(mu_ms), cov_ms, shape=()
    )
    u_indx_hi = DEFAULT_DIFFSTAR_U_PARAMS.u_ms_params.u_indx_hi
    u_params_ms = jnp.array(
        (
            *u_ms_params_ms_no_indx_hi[:3],
            u_indx_hi,
            u_ms_params_ms_no_indx_hi[3],
            *DEFAULT_Q_U_PARAMS_UNQUENCHED,
        )
    )

    u_params_ms = DiffstarUParams(
        MSUParams(*u_params_ms[:5]), QUParams(*u_params_ms[5:])
    )

    # Quenched sequence
    frac_quench = qseq_means_covs[0]
    q_key_ms_block, q_key_q_block = jran.split(q_key, 2)
    mu_qseq_ms_block, cov_qseq_ms_block = qseq_means_covs[1:3]
    mu_qseq_q_block, cov_qseq_q_block = qseq_means_covs[3:]

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

    uran = jran.uniform(frac_q_key, minval=0, maxval=1, shape=())
    mc_is_quenched_sequence = uran < frac_quench

    return u_params_ms, u_params_q, frac_quench, mc_is_quenched_sequence


@jjit
def _diffstarpop_means_covs(ms_params, qseq_params, mah_params):
    ms_means_covs = _main_sequence_mu_cov_wrapper(ms_params, mah_params)
    qseq_means_covs = _qseq_pdf_scalar_kernel_wrapper(qseq_params, mah_params)
    return ms_means_covs, qseq_means_covs


@jjit
def _main_sequence_mu_cov_wrapper(ms_mass_params, mah_params):
    ms_mass_params = MainseqMassOnlyParams(*ms_mass_params)
    lgm0 = mah_params[0]
    mu_ms = _get_mean_u_params_mainseq(ms_mass_params, lgm0)
    cov_ms = _get_cov_mainseq(ms_mass_params, lgm0)
    return mu_ms, cov_ms


@jjit
def _qseq_pdf_scalar_kernel_wrapper(qseq_params, mah_params):
    lgm0 = mah_params[0]
    qseq_means_covs = _qseq_pdf_scalar_kernel(qseq_params, lgm0)
    return qseq_means_covs
