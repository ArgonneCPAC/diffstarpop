"""
"""
import numpy as np
from jax import numpy as jnp
from jax import random as jran
from diffmah.monte_carlo_halo_population import mc_halo_population
from ..diffburstpop import _get_bursty_age_weights, _get_lgfburst
from ..diffburstpop import LGFBURST_BOUNDS_PDICT, DEFAULT_LGFBURST_PDICT
from ..diffburstpop import LGFBURST_BOUNDS
from ..diffburstpop import _get_bounded_lgfburst_params, _get_unbounded_lgfburst_params
from .. import sfhpop


def test_get_bursty_age_weights():
    T_MIN = 0.1
    T0 = 13.8
    N_T = 100

    n_gals = 50
    zz = np.zeros(n_gals)

    tarr_gyr = np.linspace(T_MIN, T0, N_T)
    lgtarr_gyr = np.log10(tarr_gyr)

    logm0 = 12.0

    halopop = mc_halo_population(tarr_gyr, T0, logm0 + zz)
    mah_params_pop = (logm0 + zz, halopop.lgtc, halopop.early_index, halopop.late_index)
    mah_params_pop = jnp.array(mah_params_pop).T

    log_age_gyr = np.arange(-4, 1.35, 0.05)
    n_ages = log_age_gyr.size

    ran_key = jran.PRNGKey(0)
    ms_key, ran_key = jran.split(ran_key, 2)

    _res = sfhpop.mc_age_weights_ms_lgmpop(
        ms_key, mah_params_pop, tarr_gyr, log_age_gyr, T0
    )
    ms_u_params_pop, q_u_params_pop, ms_sfh_pop, ms_logsmh_pop, age_pdfs_mspop = _res

    dburst_pop = np.random.uniform(2, 2.5, n_gals)
    lgfburst_pop = np.random.uniform(-3, -2, n_gals)

    t_obs = 13.0
    _res = _get_bursty_age_weights(
        log_age_gyr,
        lgtarr_gyr,
        ms_logsmh_pop,
        dburst_pop,
        lgfburst_pop,
        log_age_gyr,
        t_obs,
    )
    age_weights, age_weights_smooth, age_weights_burst = _res

    assert age_weights.shape == (n_gals, n_ages)
    assert np.allclose(np.sum(age_weights, axis=1), 1.0, rtol=1e-3)
    assert np.all(np.isfinite(age_weights))


def test_fburst_params_is_bounded():
    for key, bound in LGFBURST_BOUNDS_PDICT.items():
        default_val = DEFAULT_LGFBURST_PDICT[key]
        assert bound[0] < default_val < bound[1]


def test_fburst_params_bounding_functions():
    n_tests = 20
    nparams = len(LGFBURST_BOUNDS_PDICT)
    for __ in range(n_tests):
        uran = np.random.uniform(0, 1, nparams)
        gen = zip(uran, LGFBURST_BOUNDS)
        pran = np.array([lo + u * (hi - lo) for u, (lo, hi) in gen])
        u_p = _get_unbounded_lgfburst_params(pran)
        pran2 = _get_bounded_lgfburst_params(u_p)
        assert np.allclose(pran, pran2, atol=1e-4)


def test_get_fburst_is_sensible():
    ngals = 50_000
    logsm_obs = np.linspace(5, 15, ngals)
    logssfr_obs = np.random.uniform(-15, -5, ngals)

    fburst_params = np.array(list(DEFAULT_LGFBURST_PDICT.values()))
    lgfburst = _get_lgfburst(logsm_obs, logssfr_obs, fburst_params)
    assert np.all(np.isfinite(lgfburst))

    assert np.all(lgfburst < 0)
    assert np.all(lgfburst > -12)
