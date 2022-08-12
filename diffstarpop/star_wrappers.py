from diffstar.stars import (
    calculate_sm_sfr_fstar_history_from_mah,
    DEFAULT_SFR_PARAMS as DEFAULT_SFR_PARAMS_DICT,
    _get_unbounded_sfr_params,
)
from jax import numpy as jnp, jit as jjit, vmap, lax
from collections import OrderedDict


DEFAULT_UNBOUND_SFR_PARAMS = _get_unbounded_sfr_params(
    *tuple(DEFAULT_SFR_PARAMS_DICT.values())
)
DEFAULT_UNBOUND_SFR_PARAMS_DICT = OrderedDict(
    zip(DEFAULT_SFR_PARAMS_DICT.keys(), DEFAULT_UNBOUND_SFR_PARAMS)
)

UH = DEFAULT_UNBOUND_SFR_PARAMS_DICT["indx_hi"]


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
sm_sfr_history_vmap_halos = jjit(vmap(_sm_func_UH, in_axes=_A))
sm_sfr_history_vmap = jjit(vmap(vmap(_sm_func_UH, in_axes=_A), _B))


@jjit
def sm_sfr_history_scan(
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
        _res = sm_sfr_history_vmap_halos(
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
def sm_sfr_history_scan_full(
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

            _res = _sm_func_UH(
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

    ms_sfr_param_grids = sfh_param_grids[:, :, 0:4]

    q_u_param_grids = sfh_param_grids[:, :, 4:8]

    gen = zip(dmhdt_grids, log_mah_grids, ms_sfr_param_grids, q_u_param_grids)
    histories_on_grids = [
        sm_sfr_history_scan_full(
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
