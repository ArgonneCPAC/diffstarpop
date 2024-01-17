"""
"""
from collections import namedtuple

import numpy as np
from diffmah.defaults import DEFAULT_MAH_PARAMS, DiffmahParams
from diffstar.defaults import DEFAULT_DIFFSTAR_PARAMS
from jax import jit as jjit
from jax import numpy as jnp
from jax import random as jran
from jax import value_and_grad

from ..defaults import (
    DEFAULT_DIFFSTARPOP_PARAMS,
    DEFAULT_SFH_PDF_MAINSEQ_PDICT,
    DEFAULT_SFH_PDF_QUENCH_PDICT,
    DiffstarPopParams,
)
from ..kernels.legacy_wrapper import mc_diffstar_u_params_singlegal_kernel
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
    mah_params_galpop = DiffmahParams(*[zz + x for x in DEFAULT_MAH_PARAMS])
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
    zz = np.zeros(ngals)
    mah_params_galpop = DiffmahParams(*[zz + x for x in DEFAULT_MAH_PARAMS])
    args = (DEFAULT_DIFFSTARPOP_PARAMS, mah_params_galpop, p50 + zz, ran_key, tarr)
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
    target_sfh_params = mc_diffstar_params_galpop(
        DEFAULT_DIFFSTARPOP_PARAMS, mah_params_galpop, p50, ran_key
    )

    @jjit
    def _mse(pred, target):
        diff_ms = jnp.array(pred.ms_params) - jnp.array(target.ms_params)
        diff_q = jnp.array(pred.q_params) - jnp.array(target.q_params)
        return jnp.mean(diff_ms**2) + jnp.mean(diff_q**2)

    @jjit
    def _loss(diffstarpop_params, mah_params, p50, ran_key):
        sfh_params = mc_diffstar_params_galpop(
            diffstarpop_params, mah_params, p50, ran_key
        )
        return _mse(sfh_params, target_sfh_params)

    gfunc = jjit(value_and_grad(_loss, argnums=0))

    Params = namedtuple("Params", list(DEFAULT_SFH_PDF_MAINSEQ_PDICT.keys()))
    Params2 = namedtuple("Params2", list(DEFAULT_SFH_PDF_QUENCH_PDICT.keys()))
    sfh_pdf_mainseq_params_alt = Params(
        *[x + 0.1 for x in DEFAULT_DIFFSTARPOP_PARAMS.sfh_pdf_mainseq_params]
    )
    sfh_pdf_quench_params_alt = Params2(
        *[x + 0.1 for x in DEFAULT_DIFFSTARPOP_PARAMS.sfh_pdf_quench_params]
    )

    init_diffstar_params = DiffstarPopParams(
        sfh_pdf_mainseq_params_alt,
        sfh_pdf_quench_params_alt,
        *DEFAULT_DIFFSTARPOP_PARAMS[2:],
    )

    args = (init_diffstar_params, mah_params_galpop, p50, ran_key)
    loss, grads = gfunc(*args)
    assert np.isfinite(loss)
    assert 0 < loss < 1

    assert np.all(np.isfinite(grads.sfh_pdf_mainseq_params))
    assert np.all(np.isfinite(grads.sfh_pdf_quench_params))

    assert np.any(grads.sfh_pdf_mainseq_params != 0)
    assert np.any(grads.sfh_pdf_quench_params != 0)
