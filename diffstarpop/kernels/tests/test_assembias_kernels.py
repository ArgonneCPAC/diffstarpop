"""
"""
import numpy as np

from ..assembias_kernels import (
    DEFAULT_AB_MAINSEQ_PARAMS,
    DEFAULT_AB_QSEQ_PARAMS,
    _get_slopes_mainseq,
    _get_slopes_qseq,
)
from ..pdf_model_assembly_bias_shifts import (
    _get_slopes_mainseq as _old_get_slopes_mainseq,
)
from ..pdf_model_assembly_bias_shifts import (
    _get_slopes_quench as _old_get_slopes_quench,
)


def test_get_slopes_qseq():
    lgm = 12.0
    slopes_qseq = _get_slopes_qseq(DEFAULT_AB_QSEQ_PARAMS, lgm)
    slopes_qseq_old = _old_get_slopes_quench(lgm)
    assert np.allclose(slopes_qseq, slopes_qseq_old)


def test_get_slopes_mainseq():
    lgm = 12.0
    slopes_mainseq = _get_slopes_mainseq(DEFAULT_AB_MAINSEQ_PARAMS, lgm)
    slopes_mainseq_old = _old_get_slopes_mainseq(lgm)
    assert np.allclose(slopes_mainseq, slopes_mainseq_old)
