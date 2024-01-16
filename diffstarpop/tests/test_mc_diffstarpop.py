"""
"""
import numpy as np
from diffmah.defaults import DEFAULT_MAH_PARAMS, DiffmahParams
from diffstar.defaults import DEFAULT_DIFFSTAR_PARAMS
from jax import random as jran

from ..defaults import DEFAULT_DIFFSTARPOP_PARAMS
from ..kernels.start_over import mc_diffstar_u_params_singlegal_kernel
from ..mc_diffstarpop import (
    mc_diffstar_params_galpop,
    mc_diffstar_sfh_galpop,
    mc_diffstar_sfh_singlegal,
    mc_diffstar_u_params_galpop,
    mc_diffstar_u_params_singlegal,
)


def test_mc_diffstar_params_singlegal_consistent_with_mc_diffstar_u_params_singlegal():
    ran_key = jran.PRNGKey(0)

    p50 = 0.25
    args = DEFAULT_MAH_PARAMS, p50, ran_key
    u_params = mc_diffstar_u_params_singlegal_kernel(*args)

    args2 = (DEFAULT_DIFFSTARPOP_PARAMS, DEFAULT_MAH_PARAMS, p50, ran_key)
    u_params2 = mc_diffstar_u_params_singlegal(*args2)

    assert np.allclose(u_params[:5], u_params2.u_ms_params)
    assert np.allclose(u_params[5:], u_params2.u_q_params)


def test_mc_diffstar_sfh_singlegal_evaluates():
    ran_key = jran.PRNGKey(0)
    p50 = 0.8
    nt = 50
    tarr = np.linspace(0.1, 13.7, nt)
    diffstar_params, sfh = mc_diffstar_sfh_singlegal(
        DEFAULT_DIFFSTARPOP_PARAMS, DEFAULT_MAH_PARAMS, p50, ran_key, tarr
    )
    assert sfh.shape == (nt,)
    assert np.all(np.isfinite(sfh))
    assert np.all(sfh > 0)
    assert np.all(sfh < 1e5)

    assert len(diffstar_params.ms_params) == 5
    assert len(diffstar_params.q_params) == 4
    assert np.all(np.isfinite(diffstar_params.ms_params))
    assert np.all(np.isfinite(diffstar_params.q_params))


def test_mc_diffstar_u_params_galpop_evaluates():
    ran_key = jran.PRNGKey(0)

    p50 = 0.25

    ngals = 150
    zz = np.zeros(ngals)
    nmah = len(DEFAULT_MAH_PARAMS)
    mah_params_galpop = np.repeat(DEFAULT_MAH_PARAMS, ngals).reshape((ngals, nmah))
    args = (DEFAULT_DIFFSTARPOP_PARAMS, mah_params_galpop, p50 + zz, ran_key)
    u_params = mc_diffstar_u_params_galpop(*args)
    assert len(u_params.u_ms_params) == len(DEFAULT_DIFFSTAR_PARAMS.ms_params)
    assert len(u_params.u_q_params) == len(DEFAULT_DIFFSTAR_PARAMS.q_params)
    for u_p in u_params.u_ms_params:
        assert u_p.shape == (ngals,)
    for u_p in u_params.u_q_params:
        assert u_p.shape == (ngals,)


def test_mc_diffstar_params_galpop_evaluates():
    ran_key = jran.PRNGKey(0)

    p50 = 0.25

    ngals = 15
    zz = np.zeros(ngals)
    nmah = len(DEFAULT_MAH_PARAMS)
    mah_params_galpop = np.repeat(DEFAULT_MAH_PARAMS, ngals).reshape((ngals, nmah))
    args = (DEFAULT_DIFFSTARPOP_PARAMS, mah_params_galpop, p50 + zz, ran_key)
    params = mc_diffstar_params_galpop(*args)
    assert len(params.ms_params) == len(DEFAULT_DIFFSTAR_PARAMS.ms_params)
    assert len(params.q_params) == len(DEFAULT_DIFFSTAR_PARAMS.q_params)
    for p in params.ms_params:
        assert p.shape == (ngals,)
    for p in params.q_params:
        assert p.shape == (ngals,)


def test_mc_diffstar_sfh_galpop_evaluates():
    ran_key = jran.PRNGKey(0)
    p50 = 0.8
    nt = 50
    tarr = np.linspace(0.1, 13.7, nt)
    ngals = 150
    nmah = len(DEFAULT_MAH_PARAMS)
    zz = np.zeros(ngals)
    mah_params_galpop = np.repeat(DEFAULT_MAH_PARAMS, ngals).reshape((ngals, nmah))
    args = (DEFAULT_DIFFSTARPOP_PARAMS, mah_params_galpop, p50 + zz, ran_key, tarr)
    diffstar_params, sfh = mc_diffstar_sfh_galpop(*args)
    assert sfh.shape == (ngals, nt)
    assert np.all(np.isfinite(sfh))
    assert np.all(sfh > 0)
    assert np.all(sfh < 1e5)

    # assert len(diffstar_params.ms_params) == 5
    # assert len(diffstar_params.q_params) == 4
    # assert np.all(np.isfinite(diffstar_params.ms_params))
    # assert np.all(np.isfinite(diffstar_params.q_params))
