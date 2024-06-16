"""
"""

import numpy as np
from diffmah.utils import get_cholesky_from_params
from jax import jit as jjit
from jax import vmap

from .. import qseq_massonly_block_cov as qseq

get_cholesky_from_params_vmap = jjit(vmap(get_cholesky_from_params, in_axes=(0,)))
_get_cov_qseq_ms_vmap = jjit(vmap(qseq._get_cov_qseq_ms_block, in_axes=(None, 0)))
_get_cov_qseq_q_vmap = jjit(vmap(qseq._get_cov_qseq_q_block, in_axes=(None, 0)))


def test_frac_quench_vs_lgm0():
    lgm = 13.0
    fq = qseq._frac_quench_vs_lgm0(qseq.DEFAULT_SFH_PDF_QUENCH_BLOCK_PARAMS, lgm)
    assert 0 <= fq <= 1


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
    for cov in covs_qseq:
        det = np.linalg.det(cov)
        assert det.shape == ()
        assert det > 0
        covinv = np.linalg.inv(cov)
        assert np.all(np.isfinite(covinv))
        assert np.allclose(cov, cov.T)
        evals, evecs = np.linalg.eigh(cov)
        assert np.all(evals > 0)


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
    params = qseq.DEFAULT_SFH_PDF_QUENCH_BLOCK_PARAMS
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
