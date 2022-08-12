import numpy as np
from jax import jit as jjit, numpy as jnp, value_and_grad, random as jran
from .pdfmodel import (
    get_binned_means_and_covs,
    compute_histories_on_grids,
    _get_pdf_weights_kern,
    compute_target_sumstats_from_histories,
    get_param_grids,
)
from diffstar.utils import _get_dt_array


def get_pdf_model_loss_data(
    logm0_bins,
    lgt,
    dt,
    index_select,
    index_high,
    fstar_tdelay,
    dmhdt_grids,
    log_mah_grids,
    sfh_param_grids,
    sm_mean_target,
    sm_var_target,
    sm_loss_weight,
    fstar_mean_MS_target,
    fstar_mean_Q_target,
    fstar_var_MS_target,
    fstar_var_Q_target,
    fstar_loss_MS_weight,
    fstar_loss_Q_weight,
    quench_frac_target,
):
    loss_data = (
        logm0_bins,
        lgt,
        dt,
        index_select,
        index_high,
        fstar_tdelay,
        dmhdt_grids,
        log_mah_grids,
        sfh_param_grids,
        sm_mean_target,
        sm_var_target,
        sm_loss_weight,
        fstar_mean_MS_target,
        fstar_mean_Q_target,
        fstar_var_MS_target,
        fstar_var_Q_target,
        fstar_loss_MS_weight,
        fstar_loss_Q_weight,
        quench_frac_target,
    )
    return loss_data


@jjit
def _mse(pred, target):
    diff = pred - target
    return jnp.mean(diff * diff)


@jjit
def pdf_model_loss(pdf_model_params, loss_data):
    logm0_bins = loss_data[0]
    lgt, dt = loss_data[1:3]
    index_select, index_high, fstar_tdelay = loss_data[3:6]
    dmhdt_grids, log_mah_grids, sfh_param_grids = loss_data[6:9]
    sm_mean_target, sm_var_target, sm_loss_weight = loss_data[9:12]
    (
        fstar_mean_MS_target,
        fstar_mean_Q_target,
        fstar_var_MS_target,
        fstar_var_Q_target,
        fstar_loss_MS_weight,
        fstar_loss_Q_weight,
    ) = loss_data[12:18]
    quench_frac_target = loss_data[18]

    means, covs = get_binned_means_and_covs(pdf_model_params, logm0_bins)
    _res = compute_histories_on_grids(
        lgt,
        dt,
        index_select,
        index_high,
        fstar_tdelay,
        dmhdt_grids,
        log_mah_grids,
        sfh_param_grids,
    )
    mstar_histories, sfr_histories, fstar_histories = _res

    sfstar_histories = fstar_histories / mstar_histories[:, :, :, index_select]

    weights_quench_bin = jnp.where(sfstar_histories > 1e-11, 1.0, 0.0)

    weights_pdf = _get_pdf_weights_kern(sfh_param_grids, means, covs)

    _res = compute_target_sumstats_from_histories(
        weights_pdf, weights_quench_bin, mstar_histories, sfr_histories, fstar_histories
    )

    sm_mean, sm_variance = _res[0:2]
    fstar_mean_MS, fstar_mean_Q = _res[2:4]
    fstar_var_MS, fstar_var_Q = _res[4:6]
    quench_frac = _res[6]

    sm_mean = sm_mean[sm_loss_weight]
    sm_variance = sm_variance[sm_loss_weight]

    sm_mean_target = sm_mean_target[sm_loss_weight]
    sm_var_target = sm_var_target[sm_loss_weight]

    fstar_mean_MS = fstar_mean_MS[fstar_loss_MS_weight]
    fstar_var_MS = fstar_var_MS[fstar_loss_MS_weight]
    fstar_mean_MS_target = fstar_mean_MS_target[fstar_loss_MS_weight]
    fstar_var_MS_target = fstar_var_MS_target[fstar_loss_MS_weight]

    fstar_mean_Q = fstar_mean_Q[fstar_loss_Q_weight]
    fstar_var_Q = fstar_var_Q[fstar_loss_Q_weight]

    fstar_mean_Q_target = fstar_mean_Q_target[fstar_loss_Q_weight]
    fstar_var_Q_target = fstar_var_Q_target[fstar_loss_Q_weight]

    loss = _mse(sm_mean, sm_mean_target)
    loss += _mse(jnp.sqrt(sm_variance), jnp.sqrt(sm_var_target))
    loss += _mse(fstar_mean_MS, fstar_mean_MS_target)
    loss += _mse(fstar_mean_Q, fstar_mean_Q_target)
    loss += _mse(jnp.sqrt(fstar_var_MS), jnp.sqrt(fstar_var_MS_target))
    loss += _mse(jnp.sqrt(fstar_var_Q), jnp.sqrt(fstar_var_Q_target))
    loss += _mse(quench_frac, quench_frac_target)

    return loss


pdf_model_loss_vag = jjit(value_and_grad(pdf_model_loss, argnums=0))


def burnme(
    n_steps,
    state,
    get_pars,
    update,
    sfh_lh_sig,
    t_table,
    n_sfh_param_grid,
    logm0_binmids,
    logm0_bin_widths,
    n_halos_per_bin,
    halo_data,
    fstar_tdelay,
    sm_mean_target,
    sm_var_target,
    sm_loss_weight,
    fstar_mean_MS_target,
    fstar_mean_Q_target,
    fstar_var_MS_target,
    fstar_var_Q_target,
    fstar_loss_MS_weight,
    fstar_loss_Q_weight,
    quench_frac_target,
    jran_seed=0,
):
    """Pseudo-code for the PDF model optimization

    Parameters
    ----------
    loss_func : func
        Loss function to minimize when optimizing the PDF model
    n_steps : int
        Number of gradient descent steps
    state : obj
        object implemented in jax.experimental.optimizers
    get_pars : func
        function implemented in jax.experimental.optimizers
    update : func
        function implemented in jax.experimental.optimizers
    sfh_lh_sig : float or ndarray of shape (n_sfh_params, )
        Number of sigma used to define the latin hypercube length in each SFH dimension
    n_sfh_param_grid : int
        Number of points in the SFH parameter grid at each mass bin

    Returns
    -------
    params : ndarray of shape (n_params, )

    """
    jran_key = jran.PRNGKey(jran_seed)
    t0 = t_table[-1]
    lgt_table = jnp.log10(t_table)
    dt_table = _get_dt_array(t_table)
    logm0_halos, mah_tauc_halos, mah_early_halos, mah_late_halos = halo_data
    index_high = np.searchsorted(t_table, t_table - fstar_tdelay)
    _mask = t_table > fstar_tdelay + fstar_tdelay / 2.0
    index_select = np.arange(len(t_table))[_mask]
    index_high = index_high[_mask]

    loss_history = []
    for istep in range(n_steps):
        pdf_model_params = get_pars(state)
        dmhdt_grids, log_mah_grids, sfh_param_grids = get_param_grids(
            t_table,
            n_halos_per_bin,
            jran_key,
            logm0_binmids,
            logm0_bin_widths,
            logm0_halos,
            mah_tauc_halos,
            mah_early_halos,
            mah_late_halos,
            pdf_model_params,
            sfh_lh_sig,
            n_sfh_param_grid,
            t0,
        )
        loss_data = get_pdf_model_loss_data(
            logm0_binmids,
            lgt_table,
            dt_table,
            index_select,
            index_high,
            fstar_tdelay,
            dmhdt_grids,
            log_mah_grids,
            sfh_param_grids,
            sm_mean_target,
            sm_var_target,
            sm_loss_weight,
            fstar_mean_MS_target,
            fstar_mean_Q_target,
            fstar_var_MS_target,
            fstar_var_Q_target,
            fstar_loss_MS_weight,
            fstar_loss_Q_weight,
            quench_frac_target,
        )
        loss, grads = pdf_model_loss_vag(pdf_model_params, loss_data)
        state = update(istep, grads, state)
        loss_history.append(loss)
    return state, loss_history
