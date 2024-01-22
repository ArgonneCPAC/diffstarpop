"""
"""

import numpy as np

from ..qseq_massonly import (
    DEFAULT_SFH_PDF_QUENCH_PARAMS,
    _frac_quench_vs_lgm0,
    get_bounded_qseq_massonly_params,
    get_unbounded_qseq_massonly_params,
)


def test_frac_quench_vs_lgm0():
    lgm = 13.0
    fq = _frac_quench_vs_lgm0(DEFAULT_SFH_PDF_QUENCH_PARAMS, lgm)
    assert 0 <= fq <= 1


def test_params_u_params_inverts():
    qseq_massonly_u_params = get_unbounded_qseq_massonly_params(
        DEFAULT_SFH_PDF_QUENCH_PARAMS
    )
    qseq_massonly_params = get_bounded_qseq_massonly_params(qseq_massonly_u_params)
    assert np.allclose(DEFAULT_SFH_PDF_QUENCH_PARAMS, qseq_massonly_params)
