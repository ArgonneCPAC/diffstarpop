"""
"""

import numpy as np

from .. import qseq_massonly_block_cov as qseq


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
