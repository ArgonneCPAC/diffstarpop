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
    _get_slopes_quench as _old_get_slopes_quench,
)


def test_get_slopes_mainseq():
    lgm = 12.0
    slopes_qseq = _get_slopes_qseq(lgm, *DEFAULT_AB_QSEQ_PARAMS)
    slopes_qseq_old = _old_get_slopes_quench(lgm)
    assert np.allclose(slopes_qseq, slopes_qseq_old)
