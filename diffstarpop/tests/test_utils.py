"""
"""
import numpy as np
from jax import random as jran

from ..utils import get_t50_p50


def test_get_t50_p50_evaluates():
    nt = 50
    t_table = np.linspace(0.1, 13.7, nt)
    ngals = 150
    ran_key = jran.PRNGKey(0)
    histories = jran.uniform(ran_key, minval=0, maxval=10, shape=(ngals, nt))
    threshold = 0.5
    logmpeak = histories[:, -1]
    t50, p50 = get_t50_p50(t_table, 10**histories, threshold, logmpeak)
    assert t50.shape == (ngals,)
    assert p50.shape == (ngals,)
