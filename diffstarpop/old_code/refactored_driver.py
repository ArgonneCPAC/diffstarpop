"""
"""

from diffstar.defaults import DEFAULT_DIFFSTAR_U_PARAMS, DEFAULT_Q_U_PARAMS_UNQUENCHED
from jax import jit as jjit
from jax import numpy as jnp
from jax import random as jran

from ..kernels.assembias_kernels import _get_slopes_mainseq, _get_slopes_qseq
from ..kernels.mainseq_massonly import (
    MainseqMassOnlyParams,
    _get_cov_mainseq,
    _get_mean_u_params_mainseq,
)
from ..kernels.qseq_massonly import (
    QseqMassOnlyParams,
    _frac_quench_vs_lgm0,
    _get_cov_qseq,
    _get_mean_u_params_qseq,
)


@jjit
def mc_diffstar_u_params_singlegal_kernel(
    ms_mass_params, qs_mass_params, ab_ms_params, ab_qs_params, mah_params, p50, ran_key
):
    mu_ms, cov_ms, mu_qs, cov_qs, frac_q = _diffstarpop_pdf_params(
        ms_mass_params, qs_mass_params, ab_ms_params, ab_qs_params, mah_params, p50
    )

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

    u_params_q_no_indx_hi = jran.multivariate_normal(
        q_key, jnp.array(mu_qs), cov_qs, shape=()
    )
    u_params_q = jnp.array(
        (*u_params_q_no_indx_hi[:3], u_indx_hi, *u_params_q_no_indx_hi[3:])
    )

    uran = jran.uniform(frac_q_key, minval=0, maxval=1, shape=())
    u_params = jnp.where(uran < frac_q, u_params_q, u_params_ms)
    return u_params, u_indx_hi


@jjit
def _diffstarpop_pdf_params(
    ms_mass_params, qs_mass_params, ab_ms_params, ab_qs_params, mah_params, p50
):
    mu_ms, cov_ms = main_sequence_mu_cov(ms_mass_params, mah_params)
    mu_qs, cov_qs = quenched_sequence_mu_cov(qs_mass_params, mah_params)

    R_vals_ms = get_assembias_slopes_mainseq(ab_ms_params, mah_params)
    shifts_ms = jnp.array(R_vals_ms) * (p50 - 0.5)

    R_vals_q = get_assembias_slopes_qseq(ab_qs_params, mah_params)
    shifts_q = jnp.array(R_vals_q) * (p50 - 0.5)

    mu_ms = tuple((x + y for x, y in zip(mu_ms, shifts_ms)))
    mu_qs = tuple((x + y for x, y in zip(mu_qs, shifts_q[1:])))

    frac_q = frac_quench_vs_lgm0(qs_mass_params, mah_params)
    frac_q = frac_q + shifts_q[0]

    return mu_ms, cov_ms, mu_qs, cov_qs, frac_q


@jjit
def get_assembias_slopes_mainseq(ab_ms_params, mah_params):
    lgm0 = mah_params[0]
    ab_slopes_ms = _get_slopes_mainseq(ab_ms_params, lgm0)
    return ab_slopes_ms


@jjit
def get_assembias_slopes_qseq(ab_q_params, mah_params):
    lgm0 = mah_params[0]
    ab_slopes_q = _get_slopes_qseq(ab_q_params, lgm0)
    return ab_slopes_q


@jjit
def main_sequence_mu_cov(ms_mass_params, mah_params):
    ms_mass_params = MainseqMassOnlyParams(*ms_mass_params)
    lgm0 = mah_params[0]
    mu_ms = _get_mean_u_params_mainseq(ms_mass_params, lgm0)
    cov_ms = _get_cov_mainseq(ms_mass_params, lgm0)
    return mu_ms, cov_ms


@jjit
def quenched_sequence_mu_cov(qs_mass_params, mah_params):
    qs_mass_params = QseqMassOnlyParams(*qs_mass_params)
    lgm0 = mah_params[0]
    mu_qs = _get_mean_u_params_qseq(qs_mass_params, lgm0)
    cov_qs = _get_cov_qseq(qs_mass_params, lgm0)
    return mu_qs, cov_qs


@jjit
def frac_quench_vs_lgm0(qs_mass_params, mah_params):
    lgm0 = mah_params[0]
    return _frac_quench_vs_lgm0(qs_mass_params, lgm0)
