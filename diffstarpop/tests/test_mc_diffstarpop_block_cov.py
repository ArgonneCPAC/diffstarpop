"""
"""

import numpy as np
from diffmah.defaults import DEFAULT_MAH_PARAMS
from jax import random as jran

from .. import mc_diffstarpop_block_cov as mcdsp
from ..kernels.defaults_block_cov import DEFAULT_DIFFSTARPOP_PARAMS


def test_mc_diffstar_params_singlegal_evaluates():
    ran_key = jran.PRNGKey(0)
    args = (DEFAULT_DIFFSTARPOP_PARAMS, DEFAULT_MAH_PARAMS, ran_key)
    _res = mcdsp.mc_diffstar_params_singlegal(*args)
    params_ms, params_qseq, frac_q, mc_is_q = _res
    assert np.all(frac_q >= 0)
    assert np.all(frac_q <= 1)
    assert np.all(np.isfinite(params_ms.ms_params))
    assert np.all(np.isfinite(params_ms.q_params))
    assert np.all(np.isfinite(params_qseq.ms_params))
    assert np.all(np.isfinite(params_qseq.q_params))
    assert mc_is_q in (False, True)


def test_mc_diffstar_sfh_singlegal_evaluates():
    ran_key = jran.PRNGKey(0)
    n_times = 30
    tarr = np.linspace(0.1, 13.8, n_times)
    args = (DEFAULT_DIFFSTARPOP_PARAMS, DEFAULT_MAH_PARAMS, ran_key, tarr)
    _res = mcdsp.mc_diffstar_sfh_singlegal(*args)
    params_q, params_ms, sfh_q, sfh_ms, frac_q, mc_is_q = _res
    assert np.all(frac_q >= 0)
    assert np.all(frac_q <= 1)
    assert np.all(np.isfinite(params_ms.ms_params))
    assert np.all(np.isfinite(params_ms.q_params))
    assert np.all(np.isfinite(params_q.ms_params))
    assert np.all(np.isfinite(params_q.q_params))
    assert mc_is_q in (False, True)
    assert sfh_q.shape == (n_times,)
    assert sfh_ms.shape == (n_times,)
    assert np.all(np.isfinite(sfh_q))
    assert np.all(np.isfinite(sfh_ms))
    assert np.all(sfh_ms > 0)
    assert np.all(sfh_q > 0)


def test_mc_diffstar_u_params_galpop():
    ngals = 50
    zz = np.zeros(ngals)
    ran_key = jran.key(0)
    mah_params = DEFAULT_MAH_PARAMS._make([zz + p for p in DEFAULT_MAH_PARAMS])
    _res = mcdsp.mc_diffstar_u_params_galpop(
        DEFAULT_DIFFSTARPOP_PARAMS, mah_params, ran_key
    )
    diffstar_u_params_q, diffstar_u_params_ms, frac_q, mc_is_q = _res


def test_mc_diffstar_params_galpop():
    ngals = 50
    zz = np.zeros(ngals)
    ran_key = jran.key(0)
    mah_params = DEFAULT_MAH_PARAMS._make([zz + p for p in DEFAULT_MAH_PARAMS])
    _res = mcdsp.mc_diffstar_params_galpop(
        DEFAULT_DIFFSTARPOP_PARAMS, mah_params, ran_key
    )
    diffstar_params_q, diffstar_params_ms, frac_q, mc_is_q = _res
