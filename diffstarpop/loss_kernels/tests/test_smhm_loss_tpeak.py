"""
"""

from collections import OrderedDict, namedtuple

import numpy as np
from diffmah.diffmah_kernels import DEFAULT_MAH_PARAMS
from jax import random as jran

from ...kernels.defaults_tpeak import (
    DEFAULT_DIFFSTARPOP_PARAMS,
    DEFAULT_DIFFSTARPOP_U_PARAMS,
)
from ...mc_diffstarpop_tpeak import mc_diffstar_sfh_galpop
from .. import smhm_loss_tpeak as smhm_loss
from ..namedtuple_utils_tpeak import (
    register_tuple_new_diffstarpop_tpeak,
    tuple_to_array,
)


def get_test_loss_data():
    n_halos = 100
    n_targets = 50
    ZZ = np.zeros((n_targets, n_halos))

    ran_key = jran.PRNGKey(np.random.randint(2**32))
    lgmu_infall_data = -1.0 + ZZ
    logmhost_infall_data = 13.0 + ZZ
    gyr_since_infall_data = 2.0 + ZZ

    ran_key_data = jran.split(ran_key, n_targets)
    t_obs_targets = np.random.uniform(low=3.0, high=13.8, size=n_targets)
    smhm_targets = np.random.uniform(low=9.0, high=11.0, size=n_targets)
    logmp0_data = np.random.uniform(low=11.0, high=15.0, size=(n_targets, n_halos))

    mah_params = DEFAULT_MAH_PARAMS._make([ZZ + x for x in DEFAULT_MAH_PARAMS])
    logm0 = np.random.uniform(low=11.0, high=15.0, size=(n_targets, n_halos))
    mah_params_data = mah_params._replace(logm0=logm0)
    mah_params_data = np.array(mah_params_data)
    mah_params_data = np.transpose(mah_params_data, axes=(1, 0, 2))

    loss_data = (
        mah_params_data,
        logmp0_data,
        lgmu_infall_data,
        logmhost_infall_data,
        gyr_since_infall_data,
        ran_key_data,
        t_obs_targets,
        smhm_targets,
    )
    return loss_data


def test_smhm_loss_and_grads_are_finite():

    loss_data = get_test_loss_data()

    # Register params ---------------------------------------------

    unbound_params_dict = OrderedDict(diffstarpop_u_params=DEFAULT_DIFFSTARPOP_U_PARAMS)
    UnboundParams = namedtuple("UnboundParams", list(unbound_params_dict.keys()))
    register_tuple_new_diffstarpop_tpeak(UnboundParams)
    all_u_params = UnboundParams(*list(unbound_params_dict.values()))

    flat_all_u_params = tuple_to_array(all_u_params)

    loss, grads = smhm_loss.mean_smhm_loss_kern_tobs_wrapper(
        flat_all_u_params, loss_data
    )
    assert loss > 0, "Loss is not positive!"
    assert np.isfinite(grads).all(), "Some gradients are not finite!"
    assert np.any(~(grads == np.zeros_like(grads))), "Some gradients are zero!"


def test_mean_smhm_loss_kern():

    n_halos = 100
    ZZ = np.zeros(n_halos)

    ran_key = jran.PRNGKey(np.random.randint(2**32))
    lgmu_infall = -1.0 + ZZ
    logmhost_infall = 13.0 + ZZ
    gyr_since_infall = 2.0 + ZZ

    t_table = np.linspace(1.0, 13.8, 100)

    mah_params = DEFAULT_MAH_PARAMS._make([ZZ + x for x in DEFAULT_MAH_PARAMS])
    logm0 = np.random.uniform(low=11.0, high=15.0, size=(n_halos))
    mean_logsm_target = np.random.uniform(low=9.0, high=1.0, size=(n_halos))
    mah_params = mah_params._replace(logm0=logm0)
    mah_params = np.array(mah_params)

    loss_data = (
        mah_params,
        logm0,
        lgmu_infall,
        logmhost_infall,
        gyr_since_infall,
        ran_key,
        t_table,
        mean_logsm_target,
    )

    loss = smhm_loss.mean_smhm_loss_kern(DEFAULT_DIFFSTARPOP_PARAMS, loss_data)
    assert loss > 0, "Loss is not positive!"


def test_cumulative_mstar_formed_halopop():

    n_halos = 100
    ZZ = np.zeros(n_halos)

    ran_key = jran.PRNGKey(np.random.randint(2**32))
    lgmu_infall = -1.0 + ZZ
    logmhost_infall = 13.0 + ZZ
    gyr_since_infall = 2.0 + ZZ

    t_table = np.linspace(1.0, 13.8, 100)

    mah_params = DEFAULT_MAH_PARAMS._make([ZZ + x for x in DEFAULT_MAH_PARAMS])
    logm0 = np.random.uniform(low=11.0, high=15.0, size=(n_halos))
    mah_params = mah_params._replace(logm0=logm0)
    mah_params = np.array(mah_params)

    _res = mc_diffstar_sfh_galpop(
        DEFAULT_DIFFSTARPOP_PARAMS,
        mah_params,
        logm0,
        lgmu_infall,
        logmhost_infall,
        gyr_since_infall,
        ran_key,
        t_table,
    )
    diffstar_params_ms, diffstar_params_q, sfh_q, sfh_ms, frac_q, mc_is_q = _res

    smh_ms = smhm_loss.cumulative_mstar_formed_halopop(t_table, sfh_ms)
    smh_q = smhm_loss.cumulative_mstar_formed_halopop(t_table, sfh_q)

    assert np.isfinite(smh_ms).all()
    assert np.isfinite(smh_q).all()

    assert (smh_ms >= 0.0).all()
    assert (smh_q >= 0.0).all()
