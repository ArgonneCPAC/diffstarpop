import numpy as np
from jax import numpy as jnp, jit as jjit, vmap, random as jran, grad, lax

from .pdf_quenched import DEFAULT_SFH_PDF_QUENCH_PARAMS
from .pdf_mainseq import DEFAULT_SFH_PDF_MAINSEQ_PARAMS
from .pdf_model_assembly_bias_shifts import (
    DEFAULT_R_QUENCH_PARAMS,
    DEFAULT_R_MAINSEQ_PARAMS,
)

from .lightcone_colors import (
    sumstats_lightcone_colors_1d,
    DEFAULT_lgfburst_u_params,
    DEFAULT_burstshape_u_params,
    DEFAULT_lgav_dust_u_params,
    DEFAULT_delta_dust_u_params,
    DEFAULT_boris_dust_u_params,
)
from functools import partial


N_PDF_Q = len(DEFAULT_SFH_PDF_QUENCH_PARAMS)
N_PDF_MS = len(DEFAULT_SFH_PDF_MAINSEQ_PARAMS)
N_R_Q = len(DEFAULT_R_QUENCH_PARAMS)
N_R_MS = len(DEFAULT_R_MAINSEQ_PARAMS)

N_BURST_F = len(DEFAULT_lgfburst_u_params)
N_BURST_SHAPE = len(DEFAULT_burstshape_u_params)
N_DUST_LGAV = len(DEFAULT_lgav_dust_u_params)
N_DUST_DELTA = len(DEFAULT_delta_dust_u_params)
N_DUST_BORIS = len(DEFAULT_boris_dust_u_params)


@jjit
def mse(pred, targ):
    return jnp.mean((pred - targ) ** 2)


@partial(jjit, static_argnames=["n_histories"])
def loss(params, loss_data, n_histories, ran_key):
    (
        t_table,
        logmh_arr,
        mah_params_arr,
        p50_arr,
        pm0,
        index_select,
        index_high,
        fstar_tdelay,
        ndsig_mag,
        ndsig_color,
        bins_LO_mag,
        bins_HI_mag,
        bins_LO_color,
        bins_HI_color,
        z_arr,
        dVdz,
        dsps_data,
        ssp_obs_photflux_table_arr,
        target_data,
    ) = loss_data

    counts_target = target_data

    _npar = 0
    pdf_q_params = params[_npar : _npar + N_PDF_Q]
    _npar += N_PDF_Q
    pdf_ms_params = params[_npar : _npar + N_PDF_MS]
    _npar += N_PDF_MS
    r_q_params = params[_npar : _npar + N_R_Q]
    _npar += N_R_Q
    r_ms_params = params[_npar : _npar + N_R_MS]
    _npar += N_R_MS
    lgfburst_params = params[_npar : _npar + N_BURST_F]
    _npar += N_BURST_F
    burstshape_params = params[_npar : _npar + N_BURST_SHAPE]
    _npar += N_BURST_SHAPE
    lgav_dust_params = params[_npar : _npar + N_DUST_LGAV]
    _npar += N_DUST_LGAV
    delta_dust_params = params[_npar : _npar + N_DUST_DELTA]
    _npar += N_DUST_DELTA
    boris_dust_params = params[_npar : _npar + N_DUST_BORIS]
    _npar += N_DUST_BORIS

    counts_pred = sumstats_lightcone_colors_1d(
        t_table,
        logmh_arr,
        mah_params_arr,
        p50_arr,
        pm0,
        n_histories,
        ran_key,
        index_select,
        index_high,
        fstar_tdelay,
        ndsig_mag,
        ndsig_color,
        bins_LO_mag,
        bins_HI_mag,
        bins_LO_color,
        bins_HI_color,
        z_arr,
        dVdz,
        dsps_data,
        ssp_obs_photflux_table_arr,
        pdf_q_params,
        pdf_ms_params,
        r_q_params,
        r_ms_params,
        lgfburst_params,
        burstshape_params,
        lgav_dust_params,
        delta_dust_params,
        boris_dust_params,
    )

    loss = mse(counts_pred, counts_target)

    return loss


loss_deriv = jjit(grad(loss, argnums=(0)), static_argnames=["n_histories"])


def loss_deriv_np(params, data, n_histories):
    return np.array(loss_deriv(params, data, n_histories)).astype(float)


def print_loss_deriv(grads):
    _npar = 0
    pdf_q_grads = grads[_npar : _npar + N_PDF_Q]
    _npar += N_PDF_Q
    pdf_ms_grads = grads[_npar : _npar + N_PDF_MS]
    _npar += N_PDF_MS
    r_q_grads = grads[_npar : _npar + N_R_Q]
    _npar += N_R_Q
    r_ms_grads = grads[_npar : _npar + N_R_MS]
    _npar += N_R_MS
    lgfburst_grads = grads[_npar : _npar + N_BURST_F]
    _npar += N_BURST_F
    burstshape_grads = grads[_npar : _npar + N_BURST_SHAPE]
    _npar += N_BURST_SHAPE
    lgav_dust_grads = grads[_npar : _npar + N_DUST_LGAV]
    _npar += N_DUST_LGAV
    delta_dust_grads = grads[_npar : _npar + N_DUST_DELTA]
    _npar += N_DUST_DELTA
    boris_dust_grads = grads[_npar : _npar + N_DUST_BORIS]
    _npar += N_DUST_BORIS

    print("pdf_q_grads:", pdf_q_grads)
    print("pdf_ms_grads:", pdf_ms_grads)
    print("r_q_grads:", r_q_grads)
    print("r_ms_grads:", r_ms_grads)
    print("lgfburst_grads:", lgfburst_grads)
    print("burstshape_grads:", burstshape_grads)
    print("lgav_dust_grads:", lgav_dust_grads)
    print("delta_dust_grads:", delta_dust_grads)
    print("boris_dust_grads:", boris_dust_grads)
