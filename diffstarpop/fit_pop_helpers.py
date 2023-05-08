import numpy as np
from jax import numpy as jnp, jit as jjit, vmap, random as jran, grad
from jax.scipy.stats import multivariate_normal as jnorm
from collections import OrderedDict


# from diffmah.individual_halo_assembly import _calc_halo_history
# from diffmah.individual_halo_assembly import calc_halo_history


from diffstar.stars import (
    DEFAULT_SFR_PARAMS as DEFAULT_SFR_PARAMS_DICT,
    _get_unbounded_sfr_params,
)

from diffstar.utils import _jax_get_dt_array
from diffstar.quenching import DEFAULT_Q_PARAMS


from .pdf_quenched import DEFAULT_SFH_PDF_QUENCH_PARAMS
from .pdf_mainseq import DEFAULT_SFH_PDF_MAINSEQ_PARAMS
from .pdf_model_assembly_bias_shifts import (
    DEFAULT_R_QUENCH_PARAMS,
    DEFAULT_R_MAINSEQ_PARAMS,
)

from .monte_carlo_diff_halo_population import (
    sumstats_sfh_MIX_vmap,
    sumstats_sfh_MIX_p50_vmap,
)
from functools import partial


DEFAULT_UNBOUND_SFR_PARAMS = _get_unbounded_sfr_params(
    *tuple(DEFAULT_SFR_PARAMS_DICT.values())
)
DEFAULT_UNBOUND_SFR_PARAMS_DICT = OrderedDict(
    zip(DEFAULT_SFR_PARAMS_DICT.keys(), DEFAULT_UNBOUND_SFR_PARAMS)
)

UH = DEFAULT_UNBOUND_SFR_PARAMS_DICT["indx_hi"]

SFH_PDF_Q_KEYS = list(DEFAULT_SFH_PDF_QUENCH_PARAMS.keys())
SFH_PDF_Q_VALUES = list(DEFAULT_SFH_PDF_QUENCH_PARAMS.values())

SFH_PDF_MS_KEYS = list(DEFAULT_SFH_PDF_MAINSEQ_PARAMS.keys())
SFH_PDF_MS_VALUES = list(DEFAULT_SFH_PDF_MAINSEQ_PARAMS.values())
# SFH_PDF_DIAG_KEYS = SFH_PDF_Q_KEYS[2 : 2 + 9 * 4]


N_PDF_Q = len(DEFAULT_SFH_PDF_QUENCH_PARAMS)
N_PDF_MS = len(DEFAULT_SFH_PDF_MAINSEQ_PARAMS)
N_R_Q = len(DEFAULT_R_QUENCH_PARAMS)
N_R_MS = len(DEFAULT_R_MAINSEQ_PARAMS)


@partial(jjit, static_argnames=["n_histories"])
def loss(params, loss_data, n_histories):
    (
        t_table,
        logm0_binmids,
        halo_data,
        p50,
        ran_key,
        index_select,
        index_high,
        fstar_tdelay,
        target_data,
        # target_data_weights,
    ) = loss_data
    (
        mean_sm_target,
        variance_sm_target,
        mean_fstar_MS_target,
        mean_fstar_Q_target,
        variance_fstar_MS_target,
        variance_fstar_Q_target,
        quench_frac_target,
    ) = target_data
    """
    (
        mean_sm_weights,
        variance_sm_weights,
        mean_fstar_MS_weights,
        mean_fstar_Q_weights,
        variance_fstar_MS_weights,
        variance_fstar_Q_weights,
        quench_frac_weights,
    ) = target_data_weights
    """
    _npar = 0
    pdf_q_params = params[_npar : _npar + N_PDF_Q]
    _npar += N_PDF_Q
    pdf_ms_params = params[_npar : _npar + N_PDF_MS]
    _npar += N_PDF_MS
    r_q_params = params[_npar : _npar + N_R_Q]
    _npar += N_R_Q
    r_ms_params = params[_npar : _npar + N_R_MS]

    _res = sumstats_sfh_MIX_vmap(
        t_table,
        logm0_binmids,
        halo_data,
        p50,
        n_histories,
        ran_key,
        index_select,
        index_high,
        fstar_tdelay,
        pdf_q_params,
        pdf_ms_params,
        r_q_params,
        r_ms_params,
    )
    (
        mean_sm,
        variance_sm,
        mean_fstar_MS,
        mean_fstar_Q,
        variance_fstar_MS,
        variance_fstar_Q,
        quench_frac,
    ) = _res

    loss = jnp.mean((mean_sm - mean_sm_target) ** 2)
    loss += jnp.mean((variance_sm - variance_sm_target) ** 2)

    loss += jnp.mean((mean_fstar_MS - mean_fstar_MS_target) ** 2)
    loss += jnp.mean((mean_fstar_Q - mean_fstar_Q_target) ** 2)
    loss += jnp.mean(
        (jnp.sqrt(variance_fstar_MS) - jnp.sqrt(variance_fstar_MS_target)) ** 2
    )
    loss += jnp.mean(
        (jnp.sqrt(variance_fstar_Q) - jnp.sqrt(variance_fstar_Q_target)) ** 2
    )
    loss += jnp.mean((quench_frac - quench_frac_target) ** 2)

    """
    loss = jnp.mean(mean_sm_weights * (mean_sm - mean_sm_target) ** 2)
    loss += jnp.mean(variance_sm_weights * (variance_sm - variance_sm_target) ** 2)

    loss += jnp.mean(
        mean_fstar_MS_weights * (mean_fstar_MS - mean_fstar_MS_target) ** 2
    )
    loss += jnp.mean(mean_fstar_Q_weights * (mean_fstar_Q - mean_fstar_Q_target) ** 2)
    loss += jnp.mean(
        variance_fstar_MS_weights * (variance_fstar_MS - variance_fstar_MS_target) ** 2
    )
    loss += jnp.mean(
        variance_fstar_Q_weights * (variance_fstar_Q - variance_fstar_Q_target) ** 2
    )
    loss += jnp.mean(quench_frac_weights * (quench_frac - quench_frac_target) ** 2)
    """

    return loss


loss_deriv = jjit(grad(loss, argnums=(0)), static_argnames=["n_histories"])


def loss_deriv_np(params, data, n_histories):
    return np.array(loss_deriv(params, data, n_histories)).astype(float)


@jjit
def mse(pred, targ):
    return jnp.mean((pred - targ) ** 2)


@jjit
def mse_var(pred, targ):
    return jnp.mean((jnp.log10(jnp.sqrt(pred)) - jnp.log10(jnp.sqrt(targ))) ** 2)


@jjit
def msew(pred, targ, weight):
    return jnp.mean(weight * (pred - targ) ** 2)


@jjit
def msew_var(pred, targ, weight):
    return jnp.mean(
        weight * (jnp.log10(jnp.sqrt(pred)) - jnp.log10(jnp.sqrt(targ))) ** 2
    )


@partial(jjit, static_argnames=["n_histories"])
def loss_p50(params, loss_data, n_histories, ran_key):
    (
        t_table,
        logm0_binmids,
        halo_data,
        p50,
        index_select,
        index_high,
        sfr_tdelay,
        target_data,
        # target_data_weights,
    ) = loss_data
    (
        mean_sm_target,
        variance_sm_target,
        mean_sfr_MS_target,
        mean_sfr_Q_target,
        variance_sfr_MS_target,
        variance_sfr_Q_target,
        quench_frac_target,
        mean_sm_early_target,
        mean_sm_late_target,
        variance_sm_early_target,
        variance_sm_late_target,
        quench_frac_early_target,
        quench_frac_late_target,
    ) = target_data

    _npar = 0
    pdf_q_params = params[_npar : _npar + N_PDF_Q]
    _npar += N_PDF_Q
    pdf_ms_params = params[_npar : _npar + N_PDF_MS]
    _npar += N_PDF_MS
    r_q_params = params[_npar : _npar + N_R_Q]
    _npar += N_R_Q
    r_ms_params = params[_npar : _npar + N_R_MS]

    _res = sumstats_sfh_MIX_p50_vmap(
        t_table,
        logm0_binmids,
        halo_data,
        p50,
        n_histories,
        ran_key,
        index_select,
        index_high,
        sfr_tdelay,
        pdf_q_params,
        pdf_ms_params,
        r_q_params,
        r_ms_params,
    )
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

    wQ = jnp.where(quench_frac_target > 0.01, 1.0, 0.0)
    wSF = jnp.where(quench_frac_target < 0.99, 1.0, 0.0)

    loss = mse(mean_sm, mean_sm_target)
    loss += mse_var(variance_sm, variance_sm_target)
    loss += msew(mean_sfr_MS, mean_sfr_MS_target, wSF)
    loss += msew(mean_sfr_Q, mean_sfr_Q_target, wQ)
    loss += msew_var(variance_sfr_MS, variance_sfr_MS_target, wSF)
    loss += msew_var(variance_sfr_Q, variance_sfr_Q_target, wQ)
    loss += mse(quench_frac, quench_frac_target)
    loss += mse(mean_sm_early, mean_sm_early_target)
    loss += mse(mean_sm_late, mean_sm_late_target)
    loss += mse_var(variance_sm_early, variance_sm_early_target)
    loss += mse_var(variance_sm_late, variance_sm_late_target)
    loss += mse(quench_frac_early, quench_frac_early_target)
    loss += mse(quench_frac_late, quench_frac_late_target)

    return loss


loss_p50_deriv = jjit(grad(loss_p50, argnums=(0)), static_argnames=["n_histories"])


def loss_p50_deriv_np(params, data, n_histories):
    return np.array(loss_p50_deriv(params, data, n_histories)).astype(float)
