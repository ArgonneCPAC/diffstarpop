"""
"""

import numpy as np
from diffmah.utils import get_cholesky_from_params
from jax import jit as jjit
from jax import vmap

from .. import qseq_massonly_block_cov as qseq

get_cholesky_from_params_vmap = jjit(vmap(get_cholesky_from_params, in_axes=(0,)))


def test_frac_quench_vs_lgm0():
    lgm = 13.0
    fq = qseq._frac_quench_vs_lgm0(qseq.DEFAULT_SFH_PDF_QUENCH_PARAMS, lgm)
    assert 0 <= fq <= 1


def test_params_u_params_inverts():
    qseq_massonly_u_params = qseq.get_unbounded_qseq_massonly_params(
        qseq.DEFAULT_SFH_PDF_QUENCH_PARAMS
    )
    qseq_massonly_params = qseq.get_bounded_qseq_massonly_params(qseq_massonly_u_params)
    assert np.allclose(qseq.DEFAULT_SFH_PDF_QUENCH_PARAMS, qseq_massonly_params)


def test_get_mean_u_params_qseq():
    lgmarr = np.linspace(11, 15, 100)
    params = qseq.DEFAULT_SFH_PDF_QUENCH_PARAMS
    _means = qseq._get_mean_u_params_qseq(params, lgmarr)
    ulgm, ulgy, ul, utau, uqt, uqs, udrop, urej = _means
    for x in _means:
        assert np.all(np.isfinite(x))


def test_get_chol_u_params_qseq():
    ngals = 100
    lgmarr = np.linspace(11, 15, ngals)
    params = qseq.DEFAULT_SFH_PDF_QUENCH_PARAMS
    _chols = qseq._get_chol_u_params_qseq(params, lgmarr)
    assert len(_chols) == 36
    for x in _chols:
        assert np.all(np.isfinite(x))
        assert x.shape == (ngals,)

    chol_params = np.array(_chols).T
    assert chol_params.shape == (ngals, 36)

    chols0 = get_cholesky_from_params(chol_params[0, :])
    chols = get_cholesky_from_params_vmap(chol_params)
    assert np.allclose(chols0, chols[0, :])
    assert chols.shape == (ngals, 8, 8)
