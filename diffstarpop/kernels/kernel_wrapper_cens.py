"""
"""

from diffstar import DiffstarUParams, MSUParams, QUParams
from diffstar.defaults import DEFAULT_DIFFSTAR_U_PARAMS, DEFAULT_Q_U_PARAMS_UNQUENCHED
from jax import jit as jjit
from jax import numpy as jnp
from jax import random as jran

from .kernel_wrapper import _diffstarpop_pdf_params_cens


@jjit
def mc_diffstar_u_params_singlecen_kernel(
    ms_mass_params,
    qs_mass_params,
    ab_ms_params,
    ab_qs_params,
    mah_params,
    p50,
    ran_key,
):
    mu_ms, cov_ms, mu_qs, cov_qs, frac_q = _diffstarpop_pdf_params_cens(
        ms_mass_params, qs_mass_params, ab_ms_params, ab_qs_params, mah_params, p50
    )

    ms_key, q_key = jran.split(ran_key, 2)

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

    u_params_q_no_indx_hi = jran.multivariate_normal(
        q_key, jnp.array(mu_qs), cov_qs, shape=()
    )
    u_params_q = jnp.array(
        (*u_params_q_no_indx_hi[:3], u_indx_hi, *u_params_q_no_indx_hi[3:])
    )

    u_params_q = DiffstarUParams(MSUParams(*u_params_q[:5]), QUParams(*u_params_q[5:]))
    u_params_ms = DiffstarUParams(
        MSUParams(*u_params_ms[:5]), QUParams(*u_params_ms[5:])
    )
    return u_params_q, u_params_ms, frac_q
