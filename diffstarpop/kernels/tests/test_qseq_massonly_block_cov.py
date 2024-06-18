"""
"""

import numpy as np
import pytest
from diffmah.utils import get_cholesky_from_params
from jax import jit as jjit
from jax import random as jran
from jax import vmap

from .. import qseq_massonly_block_cov as qseq

get_cholesky_from_params_vmap = jjit(vmap(get_cholesky_from_params, in_axes=(0,)))
_get_cov_qseq_ms_vmap = jjit(vmap(qseq._get_cov_qseq_ms_block, in_axes=(None, 0)))
_get_cov_qseq_q_vmap = jjit(vmap(qseq._get_cov_qseq_q_block, in_axes=(None, 0)))

EPSILON = 1e-5


def _enforce_is_cov(matrix):
    det = np.linalg.det(matrix)
    assert det.shape == ()
    assert det > 0
    covinv = np.linalg.inv(matrix)
    assert np.all(np.isfinite(covinv))
    assert np.allclose(matrix, matrix.T)
    evals, evecs = np.linalg.eigh(matrix)
    assert np.all(evals > 0)


def test_param_u_param_names_propagate_properly():
    gen = zip(
        qseq.DEFAULT_SFH_PDF_QUENCH_BLOCK_U_PARAMS._fields,
        qseq.DEFAULT_SFH_PDF_QUENCH_BLOCK_PARAMS._fields,
    )
    for u_key, key in gen:
        assert u_key[:2] == "u_"
        assert u_key[2:] == key

    inferred_default_params = qseq.get_bounded_qseq_massonly_params(
        qseq.DEFAULT_SFH_PDF_QUENCH_BLOCK_U_PARAMS
    )
    assert set(inferred_default_params._fields) == set(
        qseq.DEFAULT_SFH_PDF_QUENCH_BLOCK_PARAMS._fields
    )

    inferred_default_u_params = qseq.get_unbounded_qseq_massonly_params(
        qseq.DEFAULT_SFH_PDF_QUENCH_BLOCK_PARAMS
    )
    assert set(inferred_default_u_params._fields) == set(
        qseq.DEFAULT_SFH_PDF_QUENCH_BLOCK_U_PARAMS._fields
    )


def test_get_bounded_params_fails_when_passing_params():
    try:
        qseq.get_bounded_qseq_massonly_params(qseq.DEFAULT_SFH_PDF_QUENCH_BLOCK_PARAMS)
        raise NameError("get_bounded_qseq_massonly_params should not accept u_params")
    except AttributeError:
        pass


def test_get_unbounded_params_fails_when_passing_u_params():
    try:
        qseq.get_unbounded_qseq_massonly_params(
            qseq.DEFAULT_SFH_PDF_QUENCH_BLOCK_U_PARAMS
        )
        raise NameError("get_unbounded_qseq_massonly_params should not accept u_params")
    except AttributeError:
        pass


def test_get_qseq_means_and_covs_vmap_fails_when_passed_u_params():
    lgmarr = np.linspace(10, 15, 20)

    try:
        qseq._get_qseq_means_and_covs_vmap(
            qseq.DEFAULT_SFH_PDF_QUENCH_BLOCK_U_PARAMS, lgmarr
        )
        raise NameError("_get_qseq_means_and_covs_vmap should not accept u_params")
    except AttributeError:
        pass


def test_param_u_param_inversion():
    ran_key = jran.key(0)
    n_tests = 100
    for itest in range(n_tests):
        ran_key, test_key = jran.split(ran_key, 2)
        n_p = len(qseq.DEFAULT_SFH_PDF_QUENCH_BLOCK_PARAMS)
        u_p = jran.uniform(test_key, minval=-10, maxval=10, shape=(n_p,))
        u_p = qseq.QseqMassOnlyBlockUParams(*u_p)
        p = qseq.get_bounded_qseq_massonly_params(u_p)
        u_p2 = qseq.get_unbounded_qseq_massonly_params(p)
        for x, y in zip(u_p, u_p2):
            assert np.allclose(x, y, rtol=0.0001)


@pytest.mark.xfail
def test_covs_are_always_covs():
    nhalos = 30
    lgmarr = np.linspace(5, 20, nhalos)
    ran_key = jran.key(0)
    n_tests = 2_000
    for itest in range(n_tests):
        ran_key, test_key = jran.split(ran_key, 2)
        n_p = len(qseq.DEFAULT_SFH_PDF_QUENCH_BLOCK_PARAMS)
        u_p = jran.uniform(test_key, minval=-100, maxval=100, shape=(n_p,))
        u_p = qseq.QseqMassOnlyBlockUParams(*u_p)
        p = qseq.get_bounded_qseq_massonly_params(u_p)
        _res = qseq._get_qseq_means_and_covs_vmap(p, lgmarr)
        for x in _res:
            assert np.all(np.isfinite(x))

        mu_ms, cov_ms, mu_q, cov_q, frac_q = _res
        assert mu_ms.shape == (nhalos, 4)
        assert cov_ms.shape == (nhalos, 4, 4)
        assert mu_q.shape == (nhalos, 4)
        assert cov_q.shape == (nhalos, 4, 4)
        assert frac_q.shape == (nhalos,)

        assert np.all(frac_q >= -EPSILON)
        assert np.all(frac_q <= 1 + EPSILON)

        for matrix in cov_ms:
            _enforce_is_cov(matrix)
        for matrix in cov_q:
            _enforce_is_cov(matrix)


def test_frac_quench_vs_lgm0():
    lgmarr = np.linspace(1, 20, 100)
    fqarr = qseq._frac_quench_vs_lgm0(qseq.DEFAULT_SFH_PDF_QUENCH_BLOCK_PARAMS, lgmarr)
    assert np.all(fqarr >= 0.0)
    assert np.all(fqarr <= 1.0)


def test_default_frac_quench_params_are_in_bounds():
    for key in qseq.FRAC_Q_PNAMES:
        bounds = qseq.FRAC_Q_BOUNDS_PDICT[key]
        val = qseq.DEFAULT_SFH_PDF_QUENCH_BLOCK_PDICT[key]
        assert bounds[0] < val < bounds[1], key


def test_params_u_params_inverts():
    qseq_massonly_u_params = qseq.get_unbounded_qseq_massonly_params(
        qseq.DEFAULT_SFH_PDF_QUENCH_BLOCK_PARAMS
    )
    qseq_massonly_params = qseq.get_bounded_qseq_massonly_params(qseq_massonly_u_params)
    assert np.allclose(qseq.DEFAULT_SFH_PDF_QUENCH_BLOCK_PARAMS, qseq_massonly_params)


def test_get_mean_u_params_qseq():
    lgmarr = np.linspace(11, 15, 100)
    params = qseq.DEFAULT_SFH_PDF_QUENCH_BLOCK_PARAMS
    _means = qseq._get_mean_u_params_qseq(params, lgmarr)
    ulgm, ulgy, ul, utau, uqt, uqs, udrop, urej = _means
    for x in _means:
        assert np.all(np.isfinite(x))


def test_get_chol_u_params_qseq_ms_block():
    ngals = 100
    lgmarr = np.linspace(11, 15, ngals)
    params = qseq.DEFAULT_SFH_PDF_QUENCH_BLOCK_PARAMS
    _chols = qseq._get_chol_u_params_qseq_ms_block(params, lgmarr)
    assert len(_chols) == 10
    for x in _chols:
        assert np.all(np.isfinite(x))
        assert x.shape == (ngals,)

    chol_params = np.array(_chols).T
    assert chol_params.shape == (ngals, 10)

    chols0 = get_cholesky_from_params(chol_params[0, :])
    chols = get_cholesky_from_params_vmap(chol_params)
    assert np.allclose(chols0, chols[0, :])
    assert chols.shape == (ngals, 4, 4)


def test_get_chol_u_params_qseq_q_block():
    ngals = 100
    lgmarr = np.linspace(11, 15, ngals)
    params = qseq.DEFAULT_SFH_PDF_QUENCH_BLOCK_PARAMS
    _chols = qseq._get_chol_u_params_qseq_q_block(params, lgmarr)
    assert len(_chols) == 10
    for x in _chols:
        assert np.all(np.isfinite(x))
        assert x.shape == (ngals,)

    chol_params = np.array(_chols).T
    assert chol_params.shape == (ngals, 10)

    chols0 = get_cholesky_from_params(chol_params[0, :])
    chols = get_cholesky_from_params_vmap(chol_params)
    assert np.allclose(chols0, chols[0, :])
    assert chols.shape == (ngals, 4, 4)


def test_get_cov_qseq_ms_block():
    ngals = 100
    lgmarr = np.linspace(11, 15, ngals)
    params = qseq.DEFAULT_SFH_PDF_QUENCH_BLOCK_PARAMS
    cov_qseq0 = qseq._get_cov_qseq_ms_block(params, lgmarr[0])
    covs_qseq = _get_cov_qseq_ms_vmap(params, lgmarr)
    assert np.allclose(cov_qseq0, covs_qseq[0, :, :])
    assert covs_qseq.shape == (ngals, 4, 4)
    assert np.all(np.isfinite(covs_qseq))
    for matrix in covs_qseq:
        _enforce_is_cov(matrix)


def test_sub_block_param_dicts_have_expected_dimension():
    ms_block_dict = qseq.DEFAULT_SFH_PDF_QUENCH_MS_BLOCK_PDICT
    q_block_dict = qseq.DEFAULT_SFH_PDF_QUENCH_Q_BLOCK_PDICT
    n_entries_chol_44 = 4 * (4 + 1) / 2
    n_params_chol_44 = 2 * n_entries_chol_44
    n_params_mean_44 = 2 * 4
    n_params_block_44 = n_params_chol_44 + n_params_mean_44
    assert len(ms_block_dict) == len(q_block_dict) == n_params_block_44
    assert set(ms_block_dict.keys()) & set(q_block_dict.keys()) == set()


def test_get_chol_u_params_qseq_block():
    n_gals = 50
    lgmarr = np.linspace(10, 15, n_gals)
    params = qseq.DEFAULT_SFH_PDF_QUENCH_BLOCK_PARAMS
    _chol_pars_q_block = qseq._get_chol_u_params_qseq_q_block(params, lgmarr)
    assert len(_chol_pars_q_block) == 4 * (4 + 1) / 2
    for x in _chol_pars_q_block:
        assert x.shape == (n_gals,)
        assert np.all(np.isfinite(x))

    _chol_pars_ms_block = qseq._get_chol_u_params_qseq_ms_block(params, lgmarr)
    assert len(_chol_pars_ms_block) == 4 * (4 + 1) / 2
    for x in _chol_pars_ms_block:
        assert x.shape == (n_gals,)
        assert np.all(np.isfinite(x))


def test_get_mean_u_params_qseq_block():
    n_gals = 50
    lgmarr = np.linspace(10, 15, n_gals)
    params = qseq.SFH_PDF_QUENCH_PARAMS
    _mean_pars_ms_block = qseq._get_mean_u_params_qseq_ms_block(params, lgmarr)
    assert len(_mean_pars_ms_block) == 4
    for x in _mean_pars_ms_block:
        assert x.shape == (n_gals,)
        assert np.all(np.isfinite(x))
    _mean_pars_q_block = qseq._get_mean_u_params_qseq_q_block(params, lgmarr)
    assert len(_mean_pars_q_block) == 4
    for x in _mean_pars_q_block:
        assert x.shape == (n_gals,)
        assert np.all(np.isfinite(x))


def test_get_mean_u_params_qseq_ms_block():
    lgm = 13.0
    _res = qseq._get_mean_u_params_qseq_ms_block(qseq.SFH_PDF_QUENCH_PARAMS, lgm)
    ulgm, ulgy, ul, utau = _res
    for x in _res:
        assert np.all(np.isfinite(x))


def test_get_mean_u_params_qseq_q_block():
    lgm = 13.0
    _res = qseq._get_mean_u_params_qseq_q_block(qseq.SFH_PDF_QUENCH_PARAMS, lgm)
    uqt, uqs, udrop, urej = _res
    for x in _res:
        assert np.all(np.isfinite(x))


def test_get_cov_params_qseq_ms_block():
    lgm = 13.0
    _res = qseq._get_cov_params_qseq_ms_block(qseq.SFH_PDF_QUENCH_PARAMS, lgm)
    for x in _res:
        assert np.all(np.isfinite(x))
