"""
"""
import pytest
from jax import random as jran
import numpy as np
from ..mc_galpop import mc_sfh_population


def test_mc_sfh_population_has_correct_shape():
    """Test a large number of halos so that there are always both quenched and
    star-forming galaxies
    """
    ran_key = jran.PRNGKey(0)
    n_halos, n_times = 1000, 50
    cosmic_time = np.linspace(0.1, 13.8, n_times)
    logmh = np.linspace(10, 15, n_halos)
    galpop = mc_sfh_population(ran_key, cosmic_time, logmh=logmh)
    smh, sfh, log_mah, msk_is_quenched = galpop
    assert smh.shape == (n_halos, n_times)
    assert np.allclose(log_mah[:, -1], logmh, atol=1e-4)
    assert np.all(sfh >= 0)


def test_mc_sfh_population_has_correct_shape2():
    """Test a very small number of halos so that sometimes galaxies are either
    all quenched or all star-forming
    """
    n_tests = 10
    n_halos, n_times = 2, 50
    cosmic_time = np.linspace(0.1, 13.8, n_times)
    logmh = np.linspace(10, 15, n_halos)
    for itest in range(n_tests):
        ran_key = jran.PRNGKey(itest)
        galpop = mc_sfh_population(ran_key, cosmic_time, logmh=logmh)
        smh, sfh, log_mah, msk_is_quenched = galpop
        assert smh.shape == (n_halos, n_times)
        assert np.allclose(log_mah[:, -1], logmh, atol=1e-4)
        assert np.all(sfh >= 0)


def test_mc_sfh_population_has_reasonable_ms_vs_q_behavior():
    """Enforce that for a population of galaxies in halos of the same mass,
    the quenched population has a smaller mean SFR at late times relative to
    the main-sequence population
    """
    ran_key = jran.PRNGKey(0)
    n_halos, n_times = 1000, 50
    cosmic_time = np.linspace(0.1, 13.8, n_times)
    logmh = np.zeros(n_halos) + 12
    galpop = mc_sfh_population(ran_key, cosmic_time, logmh=logmh)
    smh, sfh, log_mah, msk_is_quenched = galpop

    sfh_q = sfh[msk_is_quenched]
    sfh_ms = sfh[~msk_is_quenched]

    mean_sfh_q = np.mean(sfh_q, axis=0)
    mean_sfh_ms = np.mean(sfh_ms, axis=0)
    assert np.all(mean_sfh_q[-10:] <= mean_sfh_ms[-10:])
