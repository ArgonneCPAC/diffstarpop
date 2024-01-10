import numpy as np
from diffstar.kernels.main_sequence_kernels import (
    DEFAULT_MS_PARAMS as DEFAULT_SFR_PARAMS_DICT,
    _get_unbounded_sfr_params
)

from diffstar.fitting_helpers.fitting_kernels import (
    calculate_sm_sfr_fstar_history_from_mah,
    _integrate_sfr,
    compute_fstar,
    _ms_sfr_history_from_mah
)

from diffstar.kernels.quenching_kernels import (
    DEFAULT_Q_PARAMS as DEFAULT_Q_PARAMS_DICT,
    _get_unbounded_q_params,
    _quenching_kern_u_params as quenching_function
)
from jax import numpy as jnp, jit as jjit, vmap, lax
from collections import OrderedDict
from diffstar.main_sequence import get_ms_sfh_from_mah_kern


DEFAULT_UNBOUND_SFR_PARAMS = _get_unbounded_sfr_params(
    *tuple(DEFAULT_SFR_PARAMS_DICT.values())
)

DEFAULT_UNBOUND_Q_PARAMS = np.array(
    _get_unbounded_q_params(*tuple(DEFAULT_Q_PARAMS_DICT.values()))
)

DEFAULT_UNBOUND_SFR_PARAMS_DICT = OrderedDict(
    zip(DEFAULT_SFR_PARAMS_DICT.keys(), DEFAULT_UNBOUND_SFR_PARAMS)
)

UH = DEFAULT_UNBOUND_SFR_PARAMS_DICT["indx_hi"]

DEFAULT_UNBOUND_Q_PARAMS_MAIN_SEQ = DEFAULT_UNBOUND_Q_PARAMS.copy()
DEFAULT_UNBOUND_Q_PARAMS_MAIN_SEQ[0] = 1.9

sfh_scan_tobs_kern = get_ms_sfh_from_mah_kern(tobs_loop="scan")

# TAU_DEP_MIN = 0.01
MIN_SFR = 1e-10


@jjit
def sm_sfr_history_diffstar_scan(
    tarr,
    lgt,
    dt,
    mah_params,
    sfr_ms_params,
    q_params,
    index_select,
    index_high,
    fstar_tdelay,
):
    # tau_dep = sfr_ms_params[3]
    # tau_dep = jnp.clip(tau_dep, TAU_DEP_MIN, jnp.inf)
    # sfr_params = [*sfr_ms_params[0:3], UH, tau_dep]
    sfr_params = [*sfr_ms_params[0:3], UH, sfr_ms_params[3]]
    ms_sfr = sfh_scan_tobs_kern(tarr, mah_params, sfr_params)
    qfrac = quenching_function(lgt, *q_params)
    sfr = qfrac * ms_sfr
    sfr = jnp.clip(sfr, MIN_SFR, None)
    mstar = _integrate_sfr(sfr, dt)
    fstar = compute_fstar(10**lgt, mstar, index_select, index_high, fstar_tdelay)
    return mstar, sfr, fstar


@jjit
def sm_sfr_history_diffstar_scan_MS(
    tarr,
    lgt,
    dt,
    mah_params,
    sfr_ms_params,
    index_select,
    index_high,
    fstar_tdelay,
):
    # tau_dep = sfr_ms_params[3]
    # tau_dep = jnp.clip(tau_dep, TAU_DEP_MIN, jnp.inf)
    # sfr_params = [*sfr_ms_params[0:3], UH, tau_dep]
    sfr_params = [*sfr_ms_params[0:3], UH, sfr_ms_params[3]]
    sfr = sfh_scan_tobs_kern(tarr, mah_params, sfr_params)
    sfr = jnp.clip(sfr, MIN_SFR, None)
    mstar = _integrate_sfr(sfr, dt)
    fstar = compute_fstar(10**lgt, mstar, index_select, index_high, fstar_tdelay)
    return mstar, sfr, fstar


@jjit
def sfr_history_diffstar_scan(
    tarr,
    mah_params,
    sfr_ms_params,
    q_params,
):
    sfr_params = [*sfr_ms_params[0:3], UH, sfr_ms_params[3]]
    ms_sfr = sfh_scan_tobs_kern(tarr, mah_params, sfr_params)
    qfrac = quenching_function(jnp.log10(tarr), *q_params)
    sfr = qfrac * ms_sfr
    sfr = jnp.clip(sfr, MIN_SFR, None)
    return sfr


@jjit
def sfr_history_diffstar_scan_MS(
    tarr,
    mah_params,
    sfr_ms_params,
):
    sfr_params = [*sfr_ms_params[0:3], UH, sfr_ms_params[3]]
    sfr = sfh_scan_tobs_kern(tarr, mah_params, sfr_params)
    sfr = jnp.clip(sfr, MIN_SFR, None)
    return sfr


@jjit
def sm_sfr_history_diffstar_vmap(
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


@jjit
def sm_sfr_history_diffstar_vmap_MS(
    lgt,
    dt,
    dmhdt,
    log_mah,
    sfr_ms_params,
    index_select,
    index_high,
    fstar_tdelay,
):
    sfr_params = [*sfr_ms_params[0:3], UH, sfr_ms_params[3]]
    sfr = _ms_sfr_history_from_mah(lgt, dt, dmhdt, log_mah, sfr_params)
    mstar = _integrate_sfr(sfr, dt)
    fstar = compute_fstar(10**lgt, mstar, index_select, index_high, fstar_tdelay)
    return mstar, sfr, fstar


_A = (None, None, 0, 0, None, None, None, None, None)
_B = (None, None, None, None, 0, 0, None, None, None)
sm_sfr_history_diffstar_vmap_Xmah_vmap = jjit(
    vmap(sm_sfr_history_diffstar_vmap, in_axes=_A)
)
sm_sfr_history_diffstar_vmap_Xsfh_vmap_Xmah_vmap = jjit(
    vmap(vmap(sm_sfr_history_diffstar_vmap, in_axes=_A), _B)
)
_A = (None, None, 0, 0, None, None, None, None)
_B = (None, None, None, None, 0, None, None, None)
sm_sfr_history_diffstar_vmap_MS_Xsfh_vmap_Xmah_vmap = jjit(
    vmap(vmap(sm_sfr_history_diffstar_vmap_MS, in_axes=_A), _B)
)

_A = (*[None] * 3, 0, *[None] * 5)
_B = (*[None] * 4, 0, 0, *[None] * 3)
sm_sfr_history_diffstar_scan_Xsfh_vmap_Xmah_vmap = jjit(
    vmap(vmap(sm_sfr_history_diffstar_scan, in_axes=_A), _B)
)

_A = (*[None] * 3, 0, *[None] * 4)
_B = (*[None] * 4, 0, *[None] * 3)
sm_sfr_history_diffstar_scan_MS_Xsfh_vmap_Xmah_vmap = jjit(
    vmap(vmap(sm_sfr_history_diffstar_scan_MS, in_axes=_A), _B)
)

_A = (*[None] * 3, *[0] * 3, *[None] * 3)
sm_sfr_history_diffstar_scan_XsfhXmah_vmap = jjit(
    vmap(sm_sfr_history_diffstar_scan, in_axes=_A)
)

_A = (*[None] * 3, *[0] * 2, *[None] * 3)
sm_sfr_history_diffstar_scan_MS_XsfhXmah_vmap = jjit(
    vmap(sm_sfr_history_diffstar_scan_MS, in_axes=_A)
)


@jjit
def sm_sfr_history_diffstar_vmap_Xsfh_scan_Xmah_vmap(
    lgt,
    dt,
    dmhdt,
    log_mah,
    sfr_ms_params_arr,
    q_params_arr,
    index_select,
    index_high,
    fstar_tdelay,
):
    nmah = len(dmhdt)
    nt = len(lgt)
    ntstar = len(index_high)

    init = (
        jnp.zeros((nmah, nt)),
        jnp.zeros((nmah, nt)),
        jnp.zeros((nmah, ntstar)),
    )

    @jjit
    def _testfun_scan(carry, data):
        sfr_ms_params, q_params = data
        _res = sm_sfr_history_diffstar_vmap_Xmah_vmap(
            lgt,
            dt,
            dmhdt,
            log_mah,
            sfr_ms_params,
            q_params,
            index_select,
            index_high,
            fstar_tdelay,
        )
        return _res, _res

    data = (sfr_ms_params_arr, q_params_arr)
    result = lax.scan(_testfun_scan, init, data)

    return result[1]


@jjit
def sm_sfr_history_diffstar_vmap_Xsfh_scan_Xmah_scan(
    lgt,
    dt,
    dmhdt_arr,
    log_mah_arr,
    sfr_ms_params_arr,
    q_params_arr,
    index_select,
    index_high,
    fstar_tdelay,
):
    nmah = len(dmhdt_arr)
    nfstar = len(index_high)
    nt = len(lgt)

    init1 = (
        jnp.zeros((nmah, nt)),
        jnp.zeros((nmah, nt)),
        jnp.zeros((nmah, nfstar)),
    )

    def _testfun_scan1(carry, data):
        sfr_ms_params, q_params = data

        def _testfun_scan2(carry2, data2):
            dmhdt, log_mah = data2

            _res = sm_sfr_history_diffstar_vmap(
                lgt,
                dt,
                dmhdt,
                log_mah,
                sfr_ms_params,
                q_params,
                index_select,
                index_high,
                fstar_tdelay,
            )
            return _res, _res

        init2 = (jnp.zeros(nt), jnp.zeros(nt), jnp.zeros(nfstar))

        data2 = (dmhdt_arr, log_mah_arr)
        result2 = lax.scan(_testfun_scan2, init2, data2)[1]

        return result2, result2

    data1 = (sfr_ms_params_arr, q_params_arr)
    result = lax.scan(_testfun_scan1, init1, data1)

    return result[1]


@jjit
def sm_sfr_history_diffstar_vmap_MS_Xsfh_scan_Xmah_scan(
    lgt,
    dt,
    dmhdt_arr,
    log_mah_arr,
    sfr_ms_params_arr,
    index_select,
    index_high,
    fstar_tdelay,
):
    nmah = len(dmhdt_arr)
    nfstar = len(index_high)
    nt = len(lgt)

    init1 = (
        jnp.zeros((nmah, nt)),
        jnp.zeros((nmah, nt)),
        jnp.zeros((nmah, nfstar)),
    )

    def _testfun_scan1(carry, data):
        sfr_ms_params = data

        def _testfun_scan2(carry2, data2):
            dmhdt, log_mah = data2

            _res = sm_sfr_history_diffstar_vmap_MS(
                lgt,
                dt,
                dmhdt,
                log_mah,
                sfr_ms_params,
                index_select,
                index_high,
                fstar_tdelay,
            )
            return _res, _res

        init2 = (jnp.zeros(nt), jnp.zeros(nt), jnp.zeros(nfstar))

        data2 = (dmhdt_arr, log_mah_arr)
        result2 = lax.scan(_testfun_scan2, init2, data2)[1]

        return result2, result2

    data1 = sfr_ms_params_arr
    result = lax.scan(_testfun_scan1, init1, data1)

    return result[1]


@jjit
def compute_histories_on_grids_diffstar_vmap_Xsfh_scan_Xmah_scan(
    lgt,
    dt,
    index_select,
    index_high,
    fstar_tdelay,
    dmhdt_grids,
    log_mah_grids,
    sfh_param_grids,
):
    """Calculate SMH, SFH and fstar histories on the input grids
    for a population of quenched galaxies.

    Parameters
    ----------
    t_table : ndarray of shape (n_t, )
        Cosmic time array in Gyr.
    lgt : ndarray of shape (n_t, )
        Array of log10 cosmic times in units of Gyr
    dt : ndarray of shape (n_t, )
        Time step sizes in units of Gyr
    index_select: ndarray of shape (n_times_fstar, )
        Snapshot indices used in fstar computation
    index_high: ndarray of shape (n_times_fstar, )
        Indices of np.searchsorted(t, t - fstar_tdelay)[index_select]
    fstar_tdelay: float
        Time interval in Gyr for fstar definition.
        fstar = (mstar(t) - mstar(t-fstar_tdelay)) / fstar_tdelay[Gyr]
    mah_param_grids : ndarray of shape (n_m0, n_per_m0, ndim_mah_model)
        Description.
    sfh_param_grids : ndarray of shape (n_m0, n_sfh_grid, ndim_sfh_model)
        Description.

    Returns
    -------
    mstar_histories : ndarray of shape (n_m0, n_sfh_grid, n_per_m0, n_t)
        Cumulative stellar mass history in units of Msun assuming h=1.
    sfr_histories : ndarray of shape (n_m0, n_sfh_grid, n_per_m0, n_t)
        Star formation rate history in units of Msun/yr assuming h=1.
    fstar_histories : ndarray of shape (n_m0, n_sfh_grid, n_per_m0, n_t)
        Star formation rate history in units of Msun/yr assuming h=1.

    Notes
    -----
    Details of kernel implementation:
        Diffstar tobs and tcons loops: vmap
        Diffstarpop Xmah loop: scan
        Diffstarpop Xsfh loop: scan
    """

    ms_sfr_param_grids = sfh_param_grids[:, :, 0:4]
    q_u_param_grids = sfh_param_grids[:, :, 4:8]

    gen = zip(dmhdt_grids, log_mah_grids, ms_sfr_param_grids, q_u_param_grids)
    histories_on_grids = [
        sm_sfr_history_diffstar_vmap_Xsfh_scan_Xmah_scan(
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


@jjit
def compute_histories_on_grids_MS_diffstar_vmap_Xsfh_scan_Xmah_scan(
    lgt,
    dt,
    index_select,
    index_high,
    fstar_tdelay,
    dmhdt_grids,
    log_mah_grids,
    sfh_param_grids,
):
    """Calculate SMH, SFH and fstar histories on the input grids
    for a population of main sequence galaxies.

    Parameters
    ----------
    t_table : ndarray of shape (n_t, )
        Cosmic time array in Gyr.
    lgt : ndarray of shape (n_t, )
        Array of log10 cosmic times in units of Gyr
    dt : ndarray of shape (n_t, )
        Time step sizes in units of Gyr
    index_select: ndarray of shape (n_times_fstar, )
        Snapshot indices used in fstar computation
    index_high: ndarray of shape (n_times_fstar, )
        Indices of np.searchsorted(t, t - fstar_tdelay)[index_select]
    fstar_tdelay: float
        Time interval in Gyr for fstar definition.
        fstar = (mstar(t) - mstar(t-fstar_tdelay)) / fstar_tdelay[Gyr]
    mah_param_grids : ndarray of shape (n_m0, n_per_m0, ndim_mah_model)
        Description.
    sfh_param_grids : ndarray of shape (n_m0, n_sfh_grid, ndim_sfh_model)
        Description.

    Returns
    -------
    mstar_histories : ndarray of shape (n_m0, n_sfh_grid, n_per_m0, n_t)
        Cumulative stellar mass history in units of Msun assuming h=1.
    sfr_histories : ndarray of shape (n_m0, n_sfh_grid, n_per_m0, n_t)
        Star formation rate history in units of Msun/yr assuming h=1.
    fstar_histories : ndarray of shape (n_m0, n_sfh_grid, n_per_m0, n_t)
        Star formation rate history in units of Msun/yr assuming h=1.

    Notes
    -----
    Details of kernel implementation:
        Diffstar tobs and tcons loops: scan
        Diffstarpop Xmah loop: scan
        Diffstarpop Xsfh loop: scan
    """
    ms_sfr_param_grids = sfh_param_grids[:, :, 0:4]

    gen = zip(dmhdt_grids, log_mah_grids, ms_sfr_param_grids)
    histories_on_grids = [
        sm_sfr_history_diffstar_vmap_MS_Xsfh_scan_Xmah_scan(
            lgt,
            dt,
            dmhdt,
            log_mah,
            sfr_u_ps,
            index_select,
            index_high,
            fstar_tdelay,
        )
        for (dmhdt, log_mah, sfr_u_ps) in gen
    ]

    mstar_histories = jnp.array([vals[0] for vals in histories_on_grids])
    sfr_histories = jnp.array([vals[1] for vals in histories_on_grids])
    fstar_histories = jnp.array([vals[2] for vals in histories_on_grids])
    return mstar_histories, sfr_histories, fstar_histories


@jjit
def compute_histories_on_grids_diffstar_vmap_Xsfh_vmap_Xmah_vmap(
    lgt,
    dt,
    index_select,
    index_high,
    fstar_tdelay,
    dmhdt_grids,
    log_mah_grids,
    sfh_param_grids,
):
    """Calculate SMH, SFH and fstar histories on the input grids
    for a population of quenched galaxies.

    Parameters
    ----------
    t_table : ndarray of shape (n_t, )
        Cosmic time array in Gyr.
    lgt : ndarray of shape (n_t, )
        Array of log10 cosmic times in units of Gyr
    dt : ndarray of shape (n_t, )
        Time step sizes in units of Gyr
    index_select: ndarray of shape (n_times_fstar, )
        Snapshot indices used in fstar computation
    index_high: ndarray of shape (n_times_fstar, )
        Indices of np.searchsorted(t, t - fstar_tdelay)[index_select]
    fstar_tdelay: float
        Time interval in Gyr for fstar definition.
        fstar = (mstar(t) - mstar(t-fstar_tdelay)) / fstar_tdelay[Gyr]
    mah_param_grids : ndarray of shape (n_m0, n_per_m0, ndim_mah_model)
        Description.
    sfh_param_grids : ndarray of shape (n_m0, n_sfh_grid, ndim_sfh_model)
        Description.

    Returns
    -------
    mstar_histories : ndarray of shape (n_m0, n_sfh_grid, n_per_m0, n_t)
        Cumulative stellar mass history in units of Msun assuming h=1.
    sfr_histories : ndarray of shape (n_m0, n_sfh_grid, n_per_m0, n_t)
        Star formation rate history in units of Msun/yr assuming h=1.
    fstar_histories : ndarray of shape (n_m0, n_sfh_grid, n_per_m0, n_t)
        Star formation rate history in units of Msun/yr assuming h=1.

    Notes
    -----
    Details of kernel implementation:
        Diffstar tobs and tcons loops: vmap
        Diffstarpop Xmah loop: vmap
        Diffstarpop Xsfh loop: vmap
    """

    ms_sfr_param_grids = sfh_param_grids[:, :, 0:4]
    q_u_param_grids = sfh_param_grids[:, :, 4:8]

    gen = zip(dmhdt_grids, log_mah_grids, ms_sfr_param_grids, q_u_param_grids)
    histories_on_grids = [
        sm_sfr_history_diffstar_vmap_Xsfh_vmap_Xmah_vmap(
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


@jjit
def compute_histories_on_grids_MS_diffstar_vmap_Xsfh_vmap_Xmah_vmap(
    lgt,
    dt,
    index_select,
    index_high,
    fstar_tdelay,
    dmhdt_grids,
    log_mah_grids,
    sfh_param_grids,
):
    """Calculate SMH, SFH and fstar histories on the input grids
    for a population of main sequence galaxies.

    Parameters
    ----------
    t_table : ndarray of shape (n_t, )
        Cosmic time array in Gyr.
    lgt : ndarray of shape (n_t, )
        Array of log10 cosmic times in units of Gyr
    dt : ndarray of shape (n_t, )
        Time step sizes in units of Gyr
    index_select: ndarray of shape (n_times_fstar, )
        Snapshot indices used in fstar computation
    index_high: ndarray of shape (n_times_fstar, )
        Indices of np.searchsorted(t, t - fstar_tdelay)[index_select]
    fstar_tdelay: float
        Time interval in Gyr for fstar definition.
        fstar = (mstar(t) - mstar(t-fstar_tdelay)) / fstar_tdelay[Gyr]
    mah_param_grids : ndarray of shape (n_m0, n_per_m0, ndim_mah_model)
        Description.
    sfh_param_grids : ndarray of shape (n_m0, n_sfh_grid, ndim_sfh_model)
        Description.

    Returns
    -------
    mstar_histories : ndarray of shape (n_m0, n_sfh_grid, n_per_m0, n_t)
        Cumulative stellar mass history in units of Msun assuming h=1.
    sfr_histories : ndarray of shape (n_m0, n_sfh_grid, n_per_m0, n_t)
        Star formation rate history in units of Msun/yr assuming h=1.
    fstar_histories : ndarray of shape (n_m0, n_sfh_grid, n_per_m0, n_t)
        Star formation rate history in units of Msun/yr assuming h=1.

    Notes
    -----
    Details of kernel implementation:
        Diffstar tobs and tcons loops: vmap
        Diffstarpop Xmah loop: vmap
        Diffstarpop Xsfh loop: vmap
    """
    ms_sfr_param_grids = sfh_param_grids[:, :, 0:4]

    gen = zip(dmhdt_grids, log_mah_grids, ms_sfr_param_grids)
    histories_on_grids = [
        sm_sfr_history_diffstar_vmap_MS_Xsfh_vmap_Xmah_vmap(
            lgt,
            dt,
            dmhdt,
            log_mah,
            sfr_u_ps,
            index_select,
            index_high,
            fstar_tdelay,
        )
        for (dmhdt, log_mah, sfr_u_ps) in gen
    ]

    mstar_histories = jnp.array([vals[0] for vals in histories_on_grids])
    sfr_histories = jnp.array([vals[1] for vals in histories_on_grids])
    fstar_histories = jnp.array([vals[2] for vals in histories_on_grids])
    return mstar_histories, sfr_histories, fstar_histories


@jjit
def sm_sfr_history_diffstar_scan_Xsfh_scan_Xmah_scan(
    tarr,
    lgt,
    dt,
    mah_params_arr,
    sfr_ms_params_arr,
    q_params_arr,
    index_select,
    index_high,
    fstar_tdelay,
):
    nmah = len(mah_params_arr)
    nfstar = len(index_high)
    nt = len(lgt)

    init1 = (
        jnp.zeros((nmah, nt)),
        jnp.zeros((nmah, nt)),
        jnp.zeros((nmah, nfstar)),
    )

    def _testfun_scan1(carry, data):
        sfr_ms_params, q_params = data

        def _testfun_scan2(carry2, data2):
            mah_params = data2

            _res = sm_sfr_history_diffstar_scan(
                tarr,
                lgt,
                dt,
                mah_params,
                sfr_ms_params,
                q_params,
                index_select,
                index_high,
                fstar_tdelay,
            )
            return _res, _res

        init2 = (jnp.zeros(nt), jnp.zeros(nt), jnp.zeros(nfstar))

        data2 = mah_params_arr
        result2 = lax.scan(_testfun_scan2, init2, data2)[1]

        return result2, result2

    data1 = (sfr_ms_params_arr, q_params_arr)
    result = lax.scan(_testfun_scan1, init1, data1)

    return result[1]


@jjit
def sm_sfr_history_diffstar_scan_MS_Xsfh_scan_Xmah_scan(
    tarr,
    lgt,
    dt,
    mah_params_arr,
    sfr_ms_params_arr,
    index_select,
    index_high,
    fstar_tdelay,
):
    nmah = len(mah_params_arr)
    nfstar = len(index_high)
    nt = len(lgt)

    init1 = (
        jnp.zeros((nmah, nt)),
        jnp.zeros((nmah, nt)),
        jnp.zeros((nmah, nfstar)),
    )

    def _testfun_scan1(carry, data):
        sfr_ms_params = data

        def _testfun_scan2(carry2, data2):
            mah_params = data2

            _res = sm_sfr_history_diffstar_scan_MS(
                tarr,
                lgt,
                dt,
                mah_params,
                sfr_ms_params,
                index_select,
                index_high,
                fstar_tdelay,
            )
            return _res, _res

        init2 = (jnp.zeros(nt), jnp.zeros(nt), jnp.zeros(nfstar))

        data2 = mah_params_arr
        result2 = lax.scan(_testfun_scan2, init2, data2)[1]

        return result2, result2

    data1 = sfr_ms_params_arr
    result = lax.scan(_testfun_scan1, init1, data1)

    return result[1]


@jjit
def compute_histories_on_grids_diffstar_scan_Xsfh_scan_Xmah_scan(
    tarr,
    lgt,
    dt,
    index_select,
    index_high,
    fstar_tdelay,
    mah_param_grids,
    sfh_param_grids,
):
    """Calculate SMH, SFH and fstar histories on the input grids
    for a population of quenched galaxies.

    Parameters
    ----------
    t_table : ndarray of shape (n_t, )
        Cosmic time array in Gyr.
    lgt : ndarray of shape (n_t, )
        Array of log10 cosmic times in units of Gyr
    dt : ndarray of shape (n_t, )
        Time step sizes in units of Gyr
    index_select: ndarray of shape (n_times_fstar, )
        Snapshot indices used in fstar computation
    index_high: ndarray of shape (n_times_fstar, )
        Indices of np.searchsorted(t, t - fstar_tdelay)[index_select]
    fstar_tdelay: float
        Time interval in Gyr for fstar definition.
        fstar = (mstar(t) - mstar(t-fstar_tdelay)) / fstar_tdelay[Gyr]
    mah_param_grids : ndarray of shape (n_m0, n_per_m0, ndim_mah_model)
        Description.
    sfh_param_grids : ndarray of shape (n_m0, n_sfh_grid, ndim_sfh_model)
        Description.

    Returns
    -------
    mstar_histories : ndarray of shape (n_m0, n_sfh_grid, n_per_m0, n_t)
        Cumulative stellar mass history in units of Msun assuming h=1.
    sfr_histories : ndarray of shape (n_m0, n_sfh_grid, n_per_m0, n_t)
        Star formation rate history in units of Msun/yr assuming h=1.
    fstar_histories : ndarray of shape (n_m0, n_sfh_grid, n_per_m0, n_t)
        Star formation rate history in units of Msun/yr assuming h=1.

    Notes
    -----
    Details of kernel implementation:
        Diffstar tobs and tcons loops: scan
        Diffstarpop Xmah loop: scan
        Diffstarpop Xsfh loop: scan
    """
    ms_sfr_param_grids = sfh_param_grids[:, :, 0:4]
    q_u_param_grids = sfh_param_grids[:, :, 4:8]

    gen = zip(mah_param_grids, ms_sfr_param_grids, q_u_param_grids)
    histories_on_grids = [
        sm_sfr_history_diffstar_scan_Xsfh_scan_Xmah_scan(
            tarr,
            lgt,
            dt,
            mah_params,
            sfr_u_ps,
            q_u_ps,
            index_select,
            index_high,
            fstar_tdelay,
        )
        for (mah_params, sfr_u_ps, q_u_ps) in gen
    ]

    mstar_histories = jnp.array([vals[0] for vals in histories_on_grids])
    sfr_histories = jnp.array([vals[1] for vals in histories_on_grids])
    fstar_histories = jnp.array([vals[2] for vals in histories_on_grids])
    return mstar_histories, sfr_histories, fstar_histories


@jjit
def compute_histories_on_grids_diffstar_scan_Xsfh_vmap_Xmah_vmap(
    tarr,
    lgt,
    dt,
    index_select,
    index_high,
    fstar_tdelay,
    mah_param_grids,
    sfh_param_grids,
):
    """Calculate SMH, SFH and fstar histories on the input grids
    for a population of quenched galaxies.

    Parameters
    ----------
    t_table : ndarray of shape (n_t, )
        Cosmic time array in Gyr.
    lgt : ndarray of shape (n_t, )
        Array of log10 cosmic times in units of Gyr
    dt : ndarray of shape (n_t, )
        Time step sizes in units of Gyr
    index_select: ndarray of shape (n_times_fstar, )
        Snapshot indices used in fstar computation
    index_high: ndarray of shape (n_times_fstar, )
        Indices of np.searchsorted(t, t - fstar_tdelay)[index_select]
    fstar_tdelay: float
        Time interval in Gyr for fstar definition.
        fstar = (mstar(t) - mstar(t-fstar_tdelay)) / fstar_tdelay[Gyr]
    mah_param_grids : ndarray of shape (n_m0, n_per_m0, ndim_mah_model)
        Description.
    sfh_param_grids : ndarray of shape (n_m0, n_sfh_grid, ndim_sfh_model)
        Description.

    Returns
    -------
    mstar_histories : ndarray of shape (n_m0, n_sfh_grid, n_per_m0, n_t)
        Cumulative stellar mass history in units of Msun assuming h=1.
    sfr_histories : ndarray of shape (n_m0, n_sfh_grid, n_per_m0, n_t)
        Star formation rate history in units of Msun/yr assuming h=1.
    fstar_histories : ndarray of shape (n_m0, n_sfh_grid, n_per_m0, n_t)
        Star formation rate history in units of Msun/yr assuming h=1.

    Notes
    -----
    Details of kernel implementation:
        Diffstar tobs and tcons loops: scan
        Diffstarpop Xmah loop: vmap
        Diffstarpop Xsfh loop: vmap
    """
    ms_sfr_param_grids = sfh_param_grids[:, :, 0:4]
    q_u_param_grids = sfh_param_grids[:, :, 4:8]

    gen = zip(mah_param_grids, ms_sfr_param_grids, q_u_param_grids)
    histories_on_grids = [
        sm_sfr_history_diffstar_scan_Xsfh_vmap_Xmah_vmap(
            tarr,
            lgt,
            dt,
            mah_params,
            sfr_u_ps,
            q_u_ps,
            index_select,
            index_high,
            fstar_tdelay,
        )
        for (mah_params, sfr_u_ps, q_u_ps) in gen
    ]

    mstar_histories = jnp.array([vals[0] for vals in histories_on_grids])
    sfr_histories = jnp.array([vals[1] for vals in histories_on_grids])
    fstar_histories = jnp.array([vals[2] for vals in histories_on_grids])
    return mstar_histories, sfr_histories, fstar_histories


@jjit
def compute_histories_on_grids_MS_diffstar_scan_Xsfh_scan_Xmah_scan(
    tarr,
    lgt,
    dt,
    index_select,
    index_high,
    fstar_tdelay,
    mah_param_grids,
    sfh_param_grids,
):
    """Calculate SMH, SFH and fstar histories on the input grids
    for a population of main sequence galaxies.

    Parameters
    ----------
    t_table : ndarray of shape (n_t, )
        Cosmic time array in Gyr.
    lgt : ndarray of shape (n_t, )
        Array of log10 cosmic times in units of Gyr
    dt : ndarray of shape (n_t, )
        Time step sizes in units of Gyr
    index_select: ndarray of shape (n_times_fstar, )
        Snapshot indices used in fstar computation
    index_high: ndarray of shape (n_times_fstar, )
        Indices of np.searchsorted(t, t - fstar_tdelay)[index_select]
    fstar_tdelay: float
        Time interval in Gyr for fstar definition.
        fstar = (mstar(t) - mstar(t-fstar_tdelay)) / fstar_tdelay[Gyr]
    mah_param_grids : ndarray of shape (n_m0, n_per_m0, ndim_mah_model)
        Description.
    sfh_param_grids : ndarray of shape (n_m0, n_sfh_grid, ndim_sfh_model)
        Description.

    Returns
    -------
    mstar_histories : ndarray of shape (n_m0, n_sfh_grid, n_per_m0, n_t)
        Cumulative stellar mass history in units of Msun assuming h=1.
    sfr_histories : ndarray of shape (n_m0, n_sfh_grid, n_per_m0, n_t)
        Star formation rate history in units of Msun/yr assuming h=1.
    fstar_histories : ndarray of shape (n_m0, n_sfh_grid, n_per_m0, n_t)
        Star formation rate history in units of Msun/yr assuming h=1.

    Notes
    -----
    Details of kernel implementation:
        Diffstar tobs and tcons loops: scan
        Diffstarpop Xmah loop: scan
        Diffstarpop Xsfh loop: scan
    """
    ms_sfr_param_grids = sfh_param_grids[:, :, 0:4]

    gen = zip(mah_param_grids, ms_sfr_param_grids)
    histories_on_grids = [
        sm_sfr_history_diffstar_scan_MS_Xsfh_scan_Xmah_scan(
            tarr,
            lgt,
            dt,
            mah_params,
            sfr_u_ps,
            index_select,
            index_high,
            fstar_tdelay,
        )
        for (mah_params, sfr_u_ps) in gen
    ]

    mstar_histories = jnp.array([vals[0] for vals in histories_on_grids])
    sfr_histories = jnp.array([vals[1] for vals in histories_on_grids])
    fstar_histories = jnp.array([vals[2] for vals in histories_on_grids])
    return mstar_histories, sfr_histories, fstar_histories


@jjit
def compute_histories_on_grids_MS_diffstar_scan_Xsfh_vmap_Xmah_vmap(
    tarr,
    lgt,
    dt,
    index_select,
    index_high,
    fstar_tdelay,
    mah_param_grids,
    sfh_param_grids,
):
    """Calculate SMH, SFH and fstar histories on the input grids
    for a population of main sequence galaxies.

    Parameters
    ----------
    t_table : ndarray of shape (n_t, )
        Cosmic time array in Gyr.
    lgt : ndarray of shape (n_t, )
        Array of log10 cosmic times in units of Gyr
    dt : ndarray of shape (n_t, )
        Time step sizes in units of Gyr
    index_select: ndarray of shape (n_times_fstar, )
        Snapshot indices used in fstar computation
    index_high: ndarray of shape (n_times_fstar, )
        Indices of np.searchsorted(t, t - fstar_tdelay)[index_select]
    fstar_tdelay: float
        Time interval in Gyr for fstar definition.
        fstar = (mstar(t) - mstar(t-fstar_tdelay)) / fstar_tdelay[Gyr]
    mah_param_grids : ndarray of shape (n_m0, n_per_m0, ndim_mah_model)
        Description.
    sfh_param_grids : ndarray of shape (n_m0, n_sfh_grid, ndim_sfh_model)
        Description.

    Returns
    -------
    mstar_histories : ndarray of shape (n_m0, n_sfh_grid, n_per_m0, n_t)
        Cumulative stellar mass history in units of Msun assuming h=1.
    sfr_histories : ndarray of shape (n_m0, n_sfh_grid, n_per_m0, n_t)
        Star formation rate history in units of Msun/yr assuming h=1.
    fstar_histories : ndarray of shape (n_m0, n_sfh_grid, n_per_m0, n_t)
        Star formation rate history in units of Msun/yr assuming h=1.

    Notes
    -----
    Details of kernel implementation:
        Diffstar tobs and tcons loops: scan
        Diffstarpop Xmah loop: vmap
        Diffstarpop Xsfh loop: vmap
    """
    ms_sfr_param_grids = sfh_param_grids[:, :, 0:4]

    gen = zip(mah_param_grids, ms_sfr_param_grids)
    histories_on_grids = [
        sm_sfr_history_diffstar_scan_MS_Xsfh_vmap_Xmah_vmap(
            tarr,
            lgt,
            dt,
            mah_params,
            sfr_u_ps,
            index_select,
            index_high,
            fstar_tdelay,
        )
        for (mah_params, sfr_u_ps) in gen
    ]

    mstar_histories = jnp.array([vals[0] for vals in histories_on_grids])
    sfr_histories = jnp.array([vals[1] for vals in histories_on_grids])
    fstar_histories = jnp.array([vals[2] for vals in histories_on_grids])
    return mstar_histories, sfr_histories, fstar_histories
