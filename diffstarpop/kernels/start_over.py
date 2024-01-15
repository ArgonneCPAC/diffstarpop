"""
"""
from collections import OrderedDict

from diffstar import DEFAULT_DIFFSTAR_U_PARAMS
from jax import numpy as jnp
from jax import random as jran

from .pdf_mainseq import DEFAULT_SFH_PDF_MAINSEQ_PDICT, _get_chol_params_mainseq
from .pdf_mainseq import _get_cov_scalar as _get_cov_scalar_ms
from .pdf_mainseq import _get_mean_smah_params_mainseq
from .pdf_model_assembly_bias_shifts import (
    DEFAULT_R_MAINSEQ_PDICT,
    DEFAULT_R_QUENCH_PDICT,
    _get_slopes_mainseq,
    _get_slopes_quench,
)
from .pdf_quenched import DEFAULT_SFH_PDF_QUENCH_PDICT, _get_chol_params_quench
from .pdf_quenched import _get_cov_scalar as _get_cov_scalar_q
from .pdf_quenched import _get_mean_smah_params_quench, frac_quench_vs_lgm0

DEFAULT_Q_U_PARAMS_UNQUENCHED = jnp.ones(4) * 5


def mc_diffstar_u_params_singlegal_kernel(
    mah_params,
    p50,
    ran_key,
    pdf_pdict_Q=DEFAULT_SFH_PDF_QUENCH_PDICT,
    pdf_pdict_MS=DEFAULT_SFH_PDF_MAINSEQ_PDICT,
    R_model_pdict_Q=DEFAULT_R_QUENCH_PDICT,
    R_model_pdict_MS=DEFAULT_R_MAINSEQ_PDICT,
):
    lgm0 = mah_params[0]

    # main sequence
    means_mainseq_pdict = OrderedDict(
        [(key, val) for (key, val) in pdf_pdict_MS.items() if "mean_" in key]
    )
    cov_mainseq_pdict = OrderedDict(
        [(key, val) for (key, val) in pdf_pdict_MS.items() if "chol_" in key]
    )
    mu_ms = _get_mean_smah_params_mainseq(lgm0, **means_mainseq_pdict)
    chol_params_ms = _get_chol_params_mainseq(lgm0, **cov_mainseq_pdict)
    cov_ms = _get_cov_scalar_ms(*chol_params_ms)

    R_model_params_ms = jnp.array(list(R_model_pdict_MS.values()))
    R_vals_ms = _get_slopes_mainseq(lgm0, *R_model_params_ms)
    shifts_ms = jnp.array(R_vals_ms) * (p50 - 0.5)

    # Quenched sequence
    means_q_pdict = OrderedDict(
        [(key, val) for (key, val) in pdf_pdict_Q.items() if "mean_" in key]
    )
    cov_q_pdict = OrderedDict(
        [(key, val) for (key, val) in pdf_pdict_Q.items() if "chol_" in key]
    )

    mu_q = _get_mean_smah_params_quench(lgm0, **means_q_pdict)
    chol_params_q = _get_chol_params_quench(lgm0, **cov_q_pdict)
    cov_q = _get_cov_scalar_q(*chol_params_q)

    R_model_params_q = jnp.array(list(R_model_pdict_Q.values()))
    R_vals_q = _get_slopes_quench(lgm0, *R_model_params_q)
    shifts_q = jnp.array(R_vals_q) * (p50 - 0.5)

    fquench_params = jnp.array(list(pdf_pdict_Q.values()))[:4]
    fquench_x0 = pdf_pdict_Q["frac_quench_x0"] + shifts_q[0]
    frac_quench = frac_quench_vs_lgm0(lgm0, fquench_x0, *fquench_params[1:4])

    # Monte Carlo draws
    ms_key, q_key, frac_q_key = jran.split(ran_key, 3)

    u_params_q = jran.multivariate_normal(q_key, jnp.array(mu_q), cov_q, shape=())
    u_params_ms = jran.multivariate_normal(ms_key, jnp.array(mu_ms), cov_ms, shape=())

    u_params_ms = u_params_ms + shifts_ms
    u_params_ms = jnp.array((*u_params_ms, *DEFAULT_Q_U_PARAMS_UNQUENCHED))
    u_params_q = u_params_q + shifts_q[1:]

    uran = jran.uniform(frac_q_key, minval=0, maxval=1, shape=())
    u_params_no_indx_hi = jnp.where(uran < frac_quench, u_params_q, u_params_ms)

    u_indx_hi = DEFAULT_DIFFSTAR_U_PARAMS.u_ms_params.u_indx_hi

    u_params = (*u_params_no_indx_hi[:3], u_indx_hi, *u_params_no_indx_hi[3:])

    return u_params
