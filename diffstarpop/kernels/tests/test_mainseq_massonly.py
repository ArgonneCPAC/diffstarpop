"""
"""
from collections import OrderedDict

import numpy as np

from ..mainseq_massonly import (
    DEFAULT_SFH_PDF_MAINSEQ_PARAMS,
    _get_mean_smah_params_mainseq,
)
from ..pdf_mainseq import (
    _get_mean_smah_params_mainseq as _old_get_mean_smah_params_mainseq,
)


def test_get_mean_smah_params_mainseq_agrees_with_legacy():
    lgm = 12.0
    mu_ms_new = _get_mean_smah_params_mainseq(lgm, DEFAULT_SFH_PDF_MAINSEQ_PARAMS)
    gen = zip(DEFAULT_SFH_PDF_MAINSEQ_PARAMS._fields, DEFAULT_SFH_PDF_MAINSEQ_PARAMS)

    ms_mu_pdict = OrderedDict([(key, val) for key, val in gen if "mean_" in key])
    mu_ms_orig = _old_get_mean_smah_params_mainseq(lgm, *ms_mu_pdict.values())

    assert np.allclose(mu_ms_new, mu_ms_orig)


def test_get_mean_smah_params_mainseq_agrees_with_legacy():
    lgm = 12.0
    gen = zip(DEFAULT_SFH_PDF_MAINSEQ_PARAMS._fields, DEFAULT_SFH_PDF_MAINSEQ_PARAMS)
    ms_cov_pdict = OrderedDict([(key, val) for key, val in gen if "chol_" in key])
