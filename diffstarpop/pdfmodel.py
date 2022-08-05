"""
"""
import numpy as np
from jax import numpy as jnp, jit as jjit, value_and_grad, vmap, random as jran
from jax.scipy.stats import multivariate_normal as jnorm
from collections import OrderedDict


from diffmah.individual_halo_assembly import calc_halo_history

# from diffmah.individual_halo_assembly import _calc_halo_history


from diffstar.stars import calculate_sm_sfr_fstar_history_from_mah, DEFAULT_SFR_PARAMS
from diffstar.utils import _sigmoid, _get_dt_array
from diffstar.quenching import DEFAULT_Q_PARAMS
from .latin_hypercube import latin_hypercube_from_cov


# from .rockstar_pdf_model import _get_smah_means_and_covs
# from .rockstar_pdf_model import DEFAULT_SFH_PDF_PARAMS

from .pdf_quenched import _get_smah_means_and_covs, DEFAULT_SFH_PDF_QUENCH_PARAMS


UH = DEFAULT_SFR_PARAMS["indx_hi"]

SFH_PDF_KEYS = list(DEFAULT_SFH_PDF_QUENCH_PARAMS.keys())
SFH_PDF_DIAG_KEYS = SFH_PDF_KEYS[2 : 2 + 9 * 4]

N_MS_PARAMS = len(DEFAULT_SFR_PARAMS)
N_Q_PARAMS = len(DEFAULT_Q_PARAMS)


@jjit
def _sm_func_UH(
    lgt,
    dt,
    dmhdt,
    log_mah,
    sfr_ms_params,
    q_params,
    index_select,
    index_high,
    fstar_tdelay,
):
    sfr_params = [*sfr_ms_params[0:3], UH, sfr_ms_params[3]]
    return calculate_sm_sfr_fstar_history_from_mah(
        lgt,
        dt,
        dmhdt,
        log_mah,
        sfr_params,
        q_params,
        index_select,
        index_high,
        fstar_tdelay,
    )


_A = (None, None, 0, 0, None, None, None, None, None)
_B = (None, None, None, None, 0, 0, None, None, None)
sm_sfr_history_vmap = jjit(vmap(vmap(_sm_func_UH, in_axes=_A), _B))


@jjit
def _multivariate_normal_pdf_kernel(sfh_grid_params, mu, cov):
    return jnorm.pdf(sfh_grid_params, mu, cov)


_get_pdf_weights_kern = jjit(vmap(_multivariate_normal_pdf_kernel, in_axes=(0, 0, 0)))


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

    sfstar_histories = fstar_histories / 1e9 / mstar_histories[:, :, :, index_select]

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


@jjit
def compute_target_sumstats_from_histories(
    weights_pdf, weights_quench_bin, mstar_histories, sfr_histories, fstar_histories
):
    """
    Parameters
    ----------
    weights : ndarray of shape (n_m0, n_sfh_grid)
        Description.
    weights_quench_bin : ndarray of shape (n_m0, n_sfh_grid, n_per_m0, n_t)
        Description.
    mstar_histories : ndarray of shape (n_m0, n_sfh_grid, n_per_m0, n_t)
        Description.
    sfr_histories : ndarray of shape (n_m0, n_sfh_grid, n_per_m0, n_t)
        Description.
    fstar_histories : ndarray of shape (n_m0, n_sfh_grid, n_per_m0, n_t)
        Description.

    Returns
    -------

    """
    NHALO = mstar_histories.shape[2]
    w_sum = 1.0 / jnp.sum(weights_pdf, axis=1)
    mstar_histories = jnp.log10(mstar_histories)
    # sfr_histories = jnp.log10(sfr_histories)
    # fstar_histories = jnp.log10(fstar_histories)

    fstar_MS = fstar_histories / 1e9 * weights_quench_bin
    fstar_Q = fstar_histories / 1e9 * (1.0 - weights_quench_bin)
    fstar_MS = jnp.where(fstar_MS > 0.0, jnp.log10(fstar_MS), 0.0)
    fstar_Q = jnp.where(fstar_Q > 0.0, jnp.log10(fstar_Q), 0.0)

    NHALO_MS = jnp.einsum("ab,abcd->ad", weights_pdf, weights_quench_bin)
    NHALO_Q = jnp.einsum("ab,abcd->ad", weights_pdf, 1.0 - weights_quench_bin)
    quench_frac = NHALO_Q / (NHALO_Q + NHALO_MS)

    NHALO_MS = jnp.where(NHALO_MS > 0.0, NHALO_MS, 1.0)
    NHALO_Q = jnp.where(NHALO_Q > 0.0, NHALO_Q, 1.0)

    mean_sm = jnp.einsum("ab,a,abcd->ad", weights_pdf, w_sum, mstar_histories) / NHALO
    # mean_sfr = jnp.einsum("ab,abcd->ad", weights_pdf, sfr_histories)
    # mean_fstar = jnp.einsum("ab,a,abcd->ad", weights_pdf, w_sum, fstar_histories) / NHALO
    mean_fstar_MS = jnp.einsum("ab,abcd->ad", weights_pdf, fstar_MS) / NHALO_MS
    mean_fstar_Q = jnp.einsum("ab,abcd->ad", weights_pdf, fstar_Q) / NHALO_Q

    delta_sm = (mstar_histories - mean_sm[:, None, None, :]) ** 2
    # delta_sfr = (sfr_histories - mean_sfr) ** 2
    # delta_fstar = (fstar_histories - mean_fstar[:, None, None, :]) ** 2
    delta_fstar_MS = (fstar_MS - mean_fstar_MS[:, None, None, :]) ** 2
    delta_fstar_Q = (fstar_Q - mean_fstar_Q[:, None, None, :]) ** 2

    # delta_fstar_MS = jnp.where(fstar_MS == 0.0, 0.0, delta_fstar_MS)
    # delta_fstar_Q = jnp.where(fstar_Q == 0.0, 0.0, delta_fstar_Q)

    delta_fstar_MS = delta_fstar_MS * weights_quench_bin
    delta_fstar_Q = delta_fstar_Q * (1.0 - weights_quench_bin)

    variance_sm = jnp.einsum("ab,a,abcd->ad", weights_pdf, w_sum, delta_sm) / NHALO
    # variance_sfr = jnp.sum(delta_sfr * w, axis=(0, 1, 2, 3, 4))
    # variance_fstar = jnp.einsum("ab,a,abcd->ad", weights_pdf, w_sum, delta_fstar) / NHALO
    variance_fstar_MS = (
        jnp.einsum("ab,abcd->ad", weights_pdf, delta_fstar_MS) / NHALO_MS
    )
    variance_fstar_Q = jnp.einsum("ab,abcd->ad", weights_pdf, delta_fstar_Q) / NHALO_Q

    _out = (
        mean_sm,
        variance_sm,
        mean_fstar_MS,
        mean_fstar_Q,
        variance_fstar_MS,
        variance_fstar_Q,
        quench_frac,
    )
    return _out


@jjit
def compute_histories_on_grids(
    lgt,
    dt,
    index_select,
    index_high,
    fstar_tdelay,
    dmhdt_grids,
    log_mah_grids,
    sfh_param_grids,
):
    """Calculate mstar and SFR histories on the input grids

    Parameters
    ----------
    lgt : ndarray of shape (n_t, )
        Description.
    dt : ndarray of shape (n_t, )
        Description.
    dmhdt_grids : ndarray of shape (n_m0, n_per_m0, n_t)
        Description.
    log_mah_grids : ndarray of shape (n_m0, n_per_m0, n_t)
        Description.
    sfh_param_grids : ndarray of shape (n_m0, n_sfh_grid, ndim_sfh_model)
        Description.
    Returns
    -------
    mstar_histories : ndarray of shape (n_m0, n_sfh_grid, n_per_m0, n_t)
        Description.
    sfr_histories : ndarray of shape (n_m0, n_sfh_grid, n_per_m0, n_t)
        Description.
    """

    ms_sfr_param_grids = sfh_param_grids[:, :, 0:5]

    q_u_param_grids = sfh_param_grids[:, :, 5:9]

    gen = zip(dmhdt_grids, log_mah_grids, ms_sfr_param_grids, q_u_param_grids)
    histories_on_grids = [
        sm_sfr_history_vmap(
            lgt,
            dt,
            dmhdt,
            log_mah,
            sfr_u_ps,
            q_u_ps,
            index_select,
            index_high,
            fstar_tdelay,
        )
        for (dmhdt, log_mah, sfr_u_ps, q_u_ps) in gen
    ]

    mstar_histories = jnp.array([vals[0] for vals in histories_on_grids])
    sfr_histories = jnp.array([vals[1] for vals in histories_on_grids])
    fstar_histories = jnp.array([vals[2] for vals in histories_on_grids])
    return mstar_histories, sfr_histories, fstar_histories


def get_binned_means_and_covs(pdf_model_pars, logm0_bins):
    """Calculate the mean and covariance in SFH parameter space for each input M0.

    Parameters
    ----------
    pdf_model_pars : ndarray of shape (n_sfh_params, )
        Description.
    logm0_bins : ndarray of shape (n_m0_bins, )
        Description.
    Returns
    -------
    means : ndarray of shape (n_m0_bins, n_sfh_params)
        Description.
    covs : ndarray of shape (n_m0_bins, n_sfh_params, n_sfh_params)
        Description.
    """
    # n_sfh_params = N_MS_PARAMS + N_Q_PARAMS
    # n_m0_bins = logm0_bins.size
    # means = np.zeros(n_sfh_params * n_m0_bins).reshape((n_m0_bins, n_sfh_params))
    # covs = np.array([np.eye(n_sfh_params) * (n + 1) for n in range(n_m0_bins)])
    subset_dict = OrderedDict(
        [(key, val) for (key, val) in zip(SFH_PDF_DIAG_KEYS, pdf_model_pars)]
    )

    pdf_model_params_dict = DEFAULT_SFH_PDF_PARAMS.copy()
    pdf_model_params_dict.update(subset_dict)

    _res = _get_smah_means_and_covs(logm0_bins, **pdf_model_params_dict)
    frac_quench, means_quench, covs_quench = _res
    return means_quench, covs_quench


def get_diffmah_grid(
    tarr,
    n_halos_per_bin,
    jran_key,
    logm0_binmids,
    logm0_bin_widths,
    logm0_halos,
    mah_tauc_halos,
    mah_early_halos,
    mah_late_halos,
    t0,
):
    """Get a grid of halo histories by downsampling the input diffmah parameters.

    Parameters
    ----------
    tarr : ndarray of shape (n_times, )
        Description.
    n_halos_per_bin : int
        Number of halos to select per bin
    jran_key : obj
        jran.PRNGKey(seed)
    logm0_binmids : ndarray of shape (n_bins, )
        Midpoint of the logarithmic halo mass bins
    logm0_bin_widths : ndarray of shape (n_bins, )
        logarithmic width of the halo mass bin
    logm0_halos : ndarray of shape (n_halos, )
        Description.
    mah_tauc_halos : ndarray of shape (n_halos, )
        Description.
    mah_early_halos : ndarray of shape (n_halos, )
        Description.
    mah_late_halos : ndarray of shape (n_halos, )
        Description.
    Returns
    -------
    dmhdt_grid : ndarray of shape (n_bins*n_halos_per_bin, n_times)
        Description.
    log_mah_grid : ndarray of shape (n_bins*n_halos_per_bin, n_times)
        Description.
    logm0_sample : ndarray of shape (n_bins*n_halos_per_bin, )
        Description.
    lgtc_sample : ndarray of shape (n_bins*n_halos_per_bin, )
        Description.
    early_indx_sample : ndarray of shape (n_bins*n_halos_per_bin, )
        Description.
    late_indx_sample : ndarray of shape (n_bins*n_halos_per_bin, )
        Description.
    """
    diffmah_params_grid = get_binned_halo_sample(
        n_halos_per_bin,
        jran_key,
        logm0_binmids,
        logm0_bin_widths,
        logm0_halos,
        mah_tauc_halos,
        mah_early_halos,
        mah_late_halos,
    )

    dmhdt_grid, log_mah_grid = calc_halo_history(tarr, t0, *diffmah_params_grid)
    return (dmhdt_grid, log_mah_grid, *diffmah_params_grid)


def get_binned_halo_sample(
    n_per_bin, jran_key, logm0_binmids, logm0_bin_widths, logm0_halos, *other_halo_props
):
    """Retrieve a downsampling of halos binned by the input logm0

    Parameters
    ----------
    n_per_bin : int
        Number of halos to select per bin
    jran_key : obj
        jran.PRNGKey(seed)
    logm0_binmids : ndarray of shape (n_bins, )
        Midpoint of the logarithmic halo mass bins
    logm0_bin_widths : ndarray of shape (n_bins, )
        logarithmic width of the halo mass bin
    logm0_halos : ndarray of shape (n_halos, )
        Description.
    other_halo_props : length m sequence of ndarrays of shape (n_halos, )
        Description.
    Returns
    -------
    binned_halo_sample : list of length m+1
        Each element is an ndarray of shape (n_per_bin*n_bins, )
        The first element is the array of logm0
        The remaining m elements are the input halo properties

    Notes
    -----
    The binned_halo_sample is what is used to generate a diffmah grid

    """
    binned_halos = get_binned_halos(
        logm0_binmids, logm0_bin_widths, logm0_halos, *other_halo_props
    )
    n_bins = len(logm0_binmids)
    ran_keys = jran.split(jran_key, n_bins)
    collector = []
    for ran_key_bin, halos_in_bin in zip(ran_keys, binned_halos):
        halo_sample = randomly_select_halos(n_per_bin, ran_key_bin, *halos_in_bin)
        collector.append(halo_sample)

    binned_halo_sample = [np.empty(0) for __ in range(len(collector[0]))]
    for halos_in_bin in collector:
        gen = zip(binned_halo_sample, halos_in_bin)
        binned_halo_sample = [np.concatenate((a, b)) for a, b in gen]
    return binned_halo_sample


def get_binned_halos(logm0_binmids, logm0_bin_widths, logm0_halos, *other_halo_props):
    """Retrieve a collection of halos binned by the input logm0

    Parameters
    ----------
    logm0_binmids : ndarray of shape (n_bins, )
        Midpoint of the logarithmic halo mass bins
    logm0_bin_widths : ndarray of shape (n_bins, )
        logarithmic width of the halo mass bin
    logm0_halos : ndarray of shape (n_halos, )
        Description.
    other_halo_props : length m sequence of ndarrays of shape (n_halos, )
        Description.
    Returns
    -------
    binned_halos : list of length n_bins
        Element i is a tuple of halo properties that fall within the ith bin
        The first element of each tuple is the value of logm0 of the halos
        The remaining m elements are the input halo properties
        So each tuple has length m+1

    """
    msg = "input logm0_binmids and logm0_bin_widths must have same length"
    assert len(logm0_binmids) == len(logm0_bin_widths), msg
    binned_halos = []
    for cen, w in zip(logm0_binmids, logm0_bin_widths):
        msk = np.abs(logm0_halos - cen) < w
        m0_haloprop_data = [h[msk] for h in other_halo_props]
        m0_data = (logm0_halos[msk], *m0_haloprop_data)
        binned_halos.append(m0_data)
    return binned_halos


def randomly_select_halos(n_sample, jran_key, *halo_sample_properties):
    n_halos = halo_sample_properties[0].size
    indx_all = np.arange(n_halos).astype("i8")
    indx_select = jran.choice(jran_key, indx_all, shape=(n_sample,), replace=False)
    return [arr[np.array(indx_select)] for arr in halo_sample_properties]


def get_sfh_param_grid(pdf_model_pars, logm0_bins, sfh_lh_sig, n_sfh_param_grid):
    means_logm0_bins, covs_logm0_bins = get_binned_means_and_covs(
        pdf_model_pars, logm0_bins
    )
    sfh_param_grid = [
        latin_hypercube_from_cov(mu, cov, sfh_lh_sig, n_sfh_param_grid)
        for mu, cov in zip(means_logm0_bins, covs_logm0_bins)
    ]
    return np.array(sfh_param_grid)


def get_param_grids(
    tarr,
    n_halos_per_bin,
    jran_key,
    logm0_binmids,
    logm0_bin_widths,
    logm0_halos,
    mah_tauc_halos,
    mah_early_halos,
    mah_late_halos,
    pdf_model_pars,
    sfh_lh_sig,
    n_sfh_param_grid,
    t0,
):
    dmhdt_grid, log_mah_grid = get_diffmah_grid(
        tarr,
        n_halos_per_bin,
        jran_key,
        logm0_binmids,
        logm0_bin_widths,
        logm0_halos,
        mah_tauc_halos,
        mah_early_halos,
        mah_late_halos,
        t0,
    )[:2]
    n_m0_bins, n_times = logm0_binmids.size, tarr.size
    _s = (n_m0_bins, n_halos_per_bin, n_times)
    log_mah_grids = np.array(log_mah_grid).reshape(_s)
    dmhdt_grids = np.array(dmhdt_grid).reshape(_s)

    sfh_param_grids = get_sfh_param_grid(
        pdf_model_pars, logm0_binmids, sfh_lh_sig, n_sfh_param_grid
    )
    return dmhdt_grids, log_mah_grids, sfh_param_grids


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


def _get_default_pdf_SFH_prediction(
    sfh_lh_sig,
    t_table,
    n_sfh_param_grid,
    logm0_binmids,
    logm0_bin_widths,
    n_halos_per_bin,
    halo_data,
    fstar_tdelay,
    pdf_model_params=None,
):
    if pdf_model_params is None:
        pdf_model_params = list(DEFAULT_SFH_PDF_PARAMS.values())[2 : 2 + 9 * 4]

    index_high = np.searchsorted(t_table, t_table - fstar_tdelay)
    _mask = t_table > fstar_tdelay + fstar_tdelay / 2.0
    index_select = np.arange(len(t_table))[_mask]
    index_high = index_high[_mask]

    jran_key = jran.PRNGKey(0)
    t0 = t_table[-1]
    lgt_table = jnp.log10(t_table)
    dt_table = _get_dt_array(t_table)
    logm0_halos, mah_tauc_halos, mah_early_halos, mah_late_halos = halo_data

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

    means, covs = get_binned_means_and_covs(pdf_model_params, logm0_binmids)
    _res = compute_histories_on_grids(
        lgt_table,
        dt_table,
        index_select,
        index_high,
        fstar_tdelay,
        dmhdt_grids,
        log_mah_grids,
        sfh_param_grids,
    )
    mstar_histories, sfr_histories, fstar_histories = _res

    sfstar_histories = fstar_histories / 1e9 / mstar_histories[:, :, :, index_select]

    weights_quench_bin = jnp.where(sfstar_histories > 1e-11, 1.0, 0.0)

    weights = _get_pdf_weights_kern(sfh_param_grids, means, covs)

    return compute_target_sumstats_from_histories(
        weights, weights_quench_bin, mstar_histories, sfr_histories, fstar_histories
    )
