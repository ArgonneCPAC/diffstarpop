"""
"""

import os

import numpy as np
import pytest

from ..old_get_loss_data import get_loss_p50_data

MSG = "get_loss_p50_data can only be tested if DIFFSTARPOP_DRN env var is set"


@pytest.mark.skipif(os.environ.get("DIFFSTARPOP_DRN", None) is None, reason=MSG)
def test_get_loss_p50_data():
    path = os.environ["DIFFSTARPOP_DRN"]
    _res = get_loss_p50_data(path)
    for x in _res:
        assert np.all(np.isfinite(x))
    t_table, logm0_binmids, halo_data_MC, p50_data, MC_res_target = _res
