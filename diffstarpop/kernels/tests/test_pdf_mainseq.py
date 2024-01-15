"""
"""
from ..pdf_mainseq import _get_mean_smah_params_mainseq


def test_get_mean_smah_params_mainseq():
    lgmh = 12.0
    ulgm, ulgy, ul, utau = _get_mean_smah_params_mainseq(lgmh)
