"""
"""
from collections import OrderedDict

import numpy as np

from ..pdf_quenched import _get_chol_params_quench as _old_get_chol_params_quench
from ..pdf_quenched import (
    _get_mean_smah_params_quench as _old_get_mean_smah_params_quench,
)
from ..qseq_massonly import (
    DEFAULT_SFH_PDF_QUENCH_PARAMS,
    _get_chol_u_params_qseq,
    _get_mean_u_params_qseq,
)


def test_get_mean_u_params_qseq_agrees_with_legacy():
    lgm = 12.0
    qs_mu = _get_mean_u_params_qseq(lgm, DEFAULT_SFH_PDF_QUENCH_PARAMS)
    assert len(qs_mu) == 8
    assert np.all(np.isfinite(qs_mu))

    gen = zip(DEFAULT_SFH_PDF_QUENCH_PARAMS._fields, DEFAULT_SFH_PDF_QUENCH_PARAMS)
    qs_mu_pdict = OrderedDict([(key, val) for key, val in gen if "mean_" in key])

    qs_mu_old = _old_get_mean_smah_params_quench(lgm, **qs_mu_pdict)

    assert np.allclose(qs_mu, qs_mu_old)


def test_get_cov_mainseq_agrees_with_legacy():
    lgm = 12.0
    qs_chol_u_params = _get_chol_u_params_qseq(lgm, DEFAULT_SFH_PDF_QUENCH_PARAMS)

    gen = zip(DEFAULT_SFH_PDF_QUENCH_PARAMS._fields, DEFAULT_SFH_PDF_QUENCH_PARAMS)
    qs_cov_pdict = OrderedDict([(key, val) for key, val in gen if "chol_" in key])
    qs_chol_u_params_old = _old_get_chol_params_quench(lgm, **qs_cov_pdict)

    assert np.allclose(qs_chol_u_params, qs_chol_u_params_old)
