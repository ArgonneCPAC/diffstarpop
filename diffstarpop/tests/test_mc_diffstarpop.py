"""
"""
import os
import pickle

import numpy as np
from diffmah.defaults import DEFAULT_MAH_PARAMS, DiffmahParams
from diffstar import DEFAULT_DIFFSTAR_PARAMS
from jax import jit as jjit
from jax import numpy as jnp
from jax import random as jran
from jax import value_and_grad

from ..defaults import DEFAULT_DIFFSTARPOP_PARAMS, DiffstarPopParams
from ..mc_diffstarpop import (
    mc_diffstar_params_galpop,
    mc_diffstar_sfh_galpop,
    mc_diffstar_u_params_galpop,
    mc_diffstar_u_params_singlegal,
)

_THIS_DRNAME = os.path.dirname(os.path.abspath(__file__))


def test_satquench_params_have_some_effect():
    ran_key = jran.PRNGKey(0)

    p50 = 0.25

    ngals = 10
    zz = np.zeros(ngals)
    nmah = len(DEFAULT_MAH_PARAMS)
    mah_params_galpop = np.repeat(DEFAULT_MAH_PARAMS, ngals).reshape((ngals, nmah))
    args = (
        DEFAULT_DIFFSTARPOP_PARAMS,
        mah_params_galpop,
        p50 + zz,
        -1 + zz,
        13 + zz,
        10 + zz,
        ran_key,
    )
    u_params = mc_diffstar_u_params_galpop(*args)

    args = (
        DEFAULT_DIFFSTARPOP_PARAMS,
        mah_params_galpop,
        p50 + zz,
        -3 + zz,
        15.0 + zz,
        10 + zz,
        ran_key,
    )
    u_params_satquench = mc_diffstar_u_params_galpop(*args)

    assert not np.allclose(u_params.u_ms_params, u_params_satquench.u_ms_params)
    assert not np.allclose(u_params.u_q_params, u_params_satquench.u_q_params)


def get_random_dpp_params(ran_key):
    collector = []
    ran_keys = jran.split(ran_key, len(DEFAULT_DIFFSTARPOP_PARAMS))
    for key, params in zip(ran_keys, DEFAULT_DIFFSTARPOP_PARAMS):
        u = jran.uniform(key, minval=-1, maxval=1, shape=(len(params),))
        ran_params = np.array(params) + u
        collector.append(params._make(ran_params))

    return DiffstarPopParams(*collector)


def test_mc_diffstar_u_params_singlegal_agrees_with_old_implementation_on_defaults():
    """this test was used in early development stages of the
    OrderedDict-->NamedTuple overhaul of DiffstarPop. This test may fail in future,
    for example of the default DiffstarPop parameters are changed.
    It may be safe to delete this test once the NamedTuple-based DiffstarPop
    implementation is sufficiently far along.

    """
    data_drn = os.path.join(_THIS_DRNAME, "testing_data")
    fname = os.path.join(data_drn, "diffstar_u_params_unit_test.pickle")
    with open(fname, "rb") as handle:
        u_params_old = pickle.load(handle)

    ran_key = jran.PRNGKey(0)

    p50 = 0.25
    lgmu_infall = -0.5
    logmhost_infall = 15.0
    gyr_since_infall = -30.0
    args = (
        DEFAULT_DIFFSTARPOP_PARAMS,
        DEFAULT_MAH_PARAMS,
        p50,
        lgmu_infall,
        logmhost_infall,
        gyr_since_infall,
        ran_key,
    )
    u_params = mc_diffstar_u_params_singlegal(*args)

    assert np.allclose(u_params_old.u_ms_params, u_params.u_ms_params, rtol=1e-3)
    assert np.allclose(u_params_old.u_q_params, u_params.u_q_params, rtol=1e-3)


def test_mc_diffstar_u_params_singlegal_evaluates_finite_on_random_u_params():
    n_tests = 100
    ran_key = jran.PRNGKey(0)
    for __ in range(n_tests):
        params_key, p50_key, test_key, ran_key, sat_key = jran.split(ran_key, 5)
        p50 = jran.uniform(p50_key, minval=0, maxval=1, shape=())
        umu, ulgmh, u_gyr = jran.uniform(sat_key, shape=(3,))
        lgmu_infall = -5 + 4 * umu
        logmhost_infall = 10 + 5 * ulgmh
        gyr_since_infall = -5 + 10 * u_gyr
        dpp_params = get_random_dpp_params(params_key)
        args = (
            dpp_params,
            DEFAULT_MAH_PARAMS,
            p50,
            lgmu_infall,
            logmhost_infall,
            gyr_since_infall,
            test_key,
        )

        u_params = mc_diffstar_u_params_singlegal(*args)
        for up in u_params:
            assert np.all(np.isfinite(up))


def test_mc_diffstar_u_params_galpop_evaluates():
    ran_key = jran.PRNGKey(0)

    p50 = 0.25
    lgmu_infall = -1.5
    logmhost_infall = 14.0
    gyr_since_infall = 3.0

    ngals = 150
    zz = np.zeros(ngals)
    nmah = len(DEFAULT_MAH_PARAMS)
    mah_params_galpop = np.repeat(DEFAULT_MAH_PARAMS, ngals).reshape((ngals, nmah))
    args = (
        DEFAULT_DIFFSTARPOP_PARAMS,
        mah_params_galpop,
        p50 + zz,
        lgmu_infall + zz,
        logmhost_infall + zz,
        gyr_since_infall + zz,
        ran_key,
    )
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
    lgmu_infall = -1.5
    logmhost_infall = 14.0
    gyr_since_infall = 3.0

    ngals = 15
    zz = np.zeros(ngals)
    mah_params_galpop = DiffmahParams(*[zz + x for x in DEFAULT_MAH_PARAMS])
    args = (
        DEFAULT_DIFFSTARPOP_PARAMS,
        mah_params_galpop,
        p50 + zz,
        lgmu_infall + zz,
        logmhost_infall + zz,
        gyr_since_infall + zz,
        ran_key,
    )
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
    lgmu_infall = -1.5
    logmhost_infall = 14.0
    gyr_since_infall = 3.0
    nt = 50
    tarr = np.linspace(0.1, 13.7, nt)
    ngals = 150
    zz = np.zeros(ngals)
    mah_params_galpop = DiffmahParams(*[zz + x for x in DEFAULT_MAH_PARAMS])
    args = (
        DEFAULT_DIFFSTARPOP_PARAMS,
        mah_params_galpop,
        p50 + zz,
        lgmu_infall + zz,
        logmhost_infall + zz,
        gyr_since_infall + zz,
        ran_key,
        tarr,
    )
    diffstar_params, sfh = mc_diffstar_sfh_galpop(*args)
    assert sfh.shape == (ngals, nt)
    assert np.all(np.isfinite(sfh))
    assert np.all(sfh > 0)
    assert np.all(sfh < 1e5)


def test_grad_mc_diffstar_params_singlegal_evaluates():
    ran_key = jran.PRNGKey(0)

    ngals = 200
    zz = np.zeros(ngals)
    mah_params_galpop = [zz + x for x in DEFAULT_MAH_PARAMS]
    p50 = np.linspace(0.01, 0.99, ngals)
    lgmu_infall = -1.5 + zz
    logmhost_infall = 14.0 + zz
    gyr_since_infall = 3.0 + zz
    target_sfh_params = mc_diffstar_params_galpop(
        DEFAULT_DIFFSTARPOP_PARAMS,
        mah_params_galpop,
        p50,
        lgmu_infall,
        logmhost_infall,
        gyr_since_infall,
        ran_key,
    )

    @jjit
    def _mse(pred, target):
        diff_ms = jnp.array(pred.ms_params) - jnp.array(target.ms_params)
        diff_q = jnp.array(pred.q_params) - jnp.array(target.q_params)
        return jnp.mean(diff_ms**2) + jnp.mean(diff_q**2)

    @jjit
    def _loss(
        diffstarpop_params,
        mah_params,
        p50,
        lgmu_infall,
        logmhost_infall,
        gyr_since_infall,
        ran_key,
    ):
        sfh_params = mc_diffstar_params_galpop(
            diffstarpop_params,
            mah_params,
            p50,
            lgmu_infall,
            logmhost_infall,
            gyr_since_infall,
            ran_key,
        )
        return _mse(sfh_params, target_sfh_params)

    gfunc = jjit(value_and_grad(_loss, argnums=0))

    pars = DEFAULT_DIFFSTARPOP_PARAMS.sfh_pdf_mainseq_params
    alt_params = [p + 0.1 for p in pars]
    sfh_pdf_mainseq_params_alt = pars._make(alt_params)

    pars = DEFAULT_DIFFSTARPOP_PARAMS.sfh_pdf_quench_params
    alt_params = [p + 0.1 for p in pars]
    sfh_pdf_quench_params_alt = pars._make(alt_params)

    init_diffstar_params = DiffstarPopParams(
        sfh_pdf_mainseq_params_alt,
        sfh_pdf_quench_params_alt,
        *DEFAULT_DIFFSTARPOP_PARAMS[2:],
    )

    args = (
        init_diffstar_params,
        mah_params_galpop,
        p50,
        lgmu_infall,
        logmhost_infall,
        gyr_since_infall,
        ran_key,
    )
    loss, grads = gfunc(*args)
    assert np.isfinite(loss)
    assert 0 < loss < 1

    assert np.all(np.isfinite(grads.sfh_pdf_mainseq_params))
    assert np.all(np.isfinite(grads.sfh_pdf_quench_params))

    assert np.any(grads.sfh_pdf_mainseq_params != 0)
    assert np.any(grads.sfh_pdf_quench_params != 0)
