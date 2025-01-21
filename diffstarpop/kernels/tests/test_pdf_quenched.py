"""
"""

import numpy as np

from ..pdf_quenched import frac_quench_vs_lgm0


def test_frac_quench_vs_lgm0():
    lgmh = 12.0
    fq = frac_quench_vs_lgm0(lgmh)
    assert np.all(np.isfinite(fq))
    assert fq.shape == ()
    assert 0 <= float(fq) <= 1
