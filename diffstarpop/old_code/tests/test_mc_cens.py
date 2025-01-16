"""
"""

import numpy as np
from diffmah.defaults import DEFAULT_MAH_PARAMS, DiffmahParams
from diffsky.mass_functions.mc_diffmah_tpeak import mc_subhalos
from diffstar import DEFAULT_DIFFSTAR_PARAMS
from diffstar.defaults import T_TABLE_MIN, TODAY
from jax import random as jran

from ...defaults import DEFAULT_DIFFSTARPOP_PARAMS
from ..mc_cens import mc_diffstar_sfh_cenpop, mc_diffstar_sfh_singlecen
from ..mc_diffstarpop import mc_diffstar_sfh_galpop


def test_mc_diffstar_params_galpop_evaluates():
    ran_key = jran.PRNGKey(0)

    p50 = 0.25

    tarr = np.linspace(T_TABLE_MIN, TODAY)
    args = (DEFAULT_DIFFSTARPOP_PARAMS, DEFAULT_MAH_PARAMS, p50, ran_key, tarr)
    params_q, params_ms, sfh_q, sfh_ms, frac_q = mc_diffstar_sfh_singlecen(*args)
    assert len(params_q.ms_params) == len(DEFAULT_DIFFSTAR_PARAMS.ms_params)
    assert len(params_q.q_params) == len(DEFAULT_DIFFSTAR_PARAMS.q_params)
    for p in params_q.ms_params:
        assert p.shape == ()
    for p in params_q.q_params:
        assert p.shape == ()

    assert len(params_ms.ms_params) == len(DEFAULT_DIFFSTAR_PARAMS.ms_params)
    assert len(params_ms.q_params) == len(DEFAULT_DIFFSTAR_PARAMS.q_params)
    for p in params_ms.ms_params:
        assert p.shape == ()
    for p in params_ms.q_params:
        assert p.shape == ()

    assert frac_q.shape == ()


def test_mc_diffstar_sfh_cenpop_evaluates():
    ran_key = jran.PRNGKey(0)
    p50 = 0.8
    nt = 50
    tarr = np.linspace(0.1, 13.7, nt)
    ngals = 150
    zz = np.zeros(ngals)
    mah_params_galpop = DiffmahParams(*[zz + x for x in DEFAULT_MAH_PARAMS])
    args = (DEFAULT_DIFFSTARPOP_PARAMS, mah_params_galpop, p50 + zz, ran_key, tarr)
    params_q, params_ms, sfh_q, sfh_ms, frac_q = mc_diffstar_sfh_cenpop(*args)
    assert sfh_q.shape == (ngals, nt)
    assert np.all(np.isfinite(sfh_q))
    assert np.all(sfh_q > 0)
    assert np.all(sfh_q < 1e5)

    assert sfh_ms.shape == (ngals, nt)
    assert np.all(np.isfinite(sfh_ms))
    assert np.all(sfh_ms > 0)
    assert np.all(sfh_ms < 1e5)


def test_mc_diffstar_sfh_cenpop_agrees_with_galpop():
    ran_key = jran.PRNGKey(0)

    lgmp_min = 11.0
    z_obs = 0.5
    volume_com = 50.0**3
    ran_key, sub_key, gal_key = jran.split(ran_key, 3)
    subcat = mc_subhalos(sub_key, lgmp_min, z_obs, volume_com)
    ngals = subcat.logmp_pen_inf.size

    p50 = np.random.uniform(0, 1, ngals)
    nt = 50
    tarr = np.linspace(0.1, 13.7, nt)
    zz = np.zeros(ngals)
    args = (DEFAULT_DIFFSTARPOP_PARAMS, subcat.mah_params, p50, ran_key, tarr)
    params_q, params_ms, sfh_q, sfh_ms, frac_q = mc_diffstar_sfh_cenpop(*args)

    lgmu_infall = zz
    logmhost_infall = zz + 12.0
    gyr_since_infall = 0.01 + zz
    args2 = (
        DEFAULT_DIFFSTARPOP_PARAMS,
        subcat.mah_params,
        p50,
        lgmu_infall,
        logmhost_infall,
        gyr_since_infall,
        ran_key,
        tarr,
    )
    params_q2, params_ms2, sfh_q2, sfh_ms2, frac_q2 = mc_diffstar_sfh_galpop(*args2)
    for p, p2 in zip(params_ms, params_ms2):
        assert np.allclose(p, p2, rtol=0.001)
    for p, p2 in zip(params_q, params_q2):
        assert np.allclose(p, p2, rtol=0.01)
    assert np.allclose(sfh_ms, sfh_ms2, rtol=0.01)
    assert np.allclose(sfh_q, sfh_q2, rtol=0.01)
    assert np.allclose(frac_q, frac_q2, rtol=0.01)

    lgmu_infall = -1.0 + zz
    logmhost_infall = 20.0 + zz
    gyr_since_infall = 100 + zz
    args3 = (*args2[:3], lgmu_infall, logmhost_infall, gyr_since_infall, ran_key, tarr)
    params_q3, params_ms3, sfh_q3, sfh_ms3, frac_q3 = mc_diffstar_sfh_galpop(*args3)
    assert np.allclose(sfh_ms3, sfh_ms2, rtol=0.01)
    assert np.allclose(sfh_q3, sfh_q2, rtol=0.01)
    assert not np.allclose(frac_q3, frac_q2, rtol=0.01)
