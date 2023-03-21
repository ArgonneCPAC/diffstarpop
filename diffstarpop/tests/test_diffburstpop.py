"""
"""
import numpy as np
from jax import numpy as jnp
from jax import random as jran
from diffmah.monte_carlo_halo_population import mc_halo_population
from ..diffburstpop import _get_bursty_age_weights
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

    _res = _get_bursty_age_weights(
        log_age_gyr, lgtarr_gyr, ms_logsmh_pop, dburst_pop, lgfburst_pop
    )
    age_weights, age_weights_smooth, age_weights_burst = _res

    assert age_weights.shape == (n_gals, n_ages)
    assert np.allclose(np.sum(age_weights, axis=1), 1.0, rtol=1e-3)
    assert np.all(np.isfinite(age_weights))
