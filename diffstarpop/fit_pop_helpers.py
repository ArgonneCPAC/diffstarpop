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

from .monte_carlo_diff_halo_population import sumstats_sfh_MIX_vmap


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


def loss(params, loss_data):
    (
        t_table,
        logm0_binmids,
        halo_data,
        p50,
        n_histories,
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
