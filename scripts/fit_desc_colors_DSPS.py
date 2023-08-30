import numpy as np
from jax import numpy as jnp
from jax import jit as jjit
from jax import vmap
from jax import grad
from jax import random as jran
import time
from functools import partial
import h5py
import os
import pandas as pd
from astropy.cosmology import Planck18

from diffstarpop.desc_colors_DSPS import (
    loss_COSMOS,
    loss_COSMOS_deriv,
    loss_HSC,
    loss_HSC_deriv,
    loss_DEEP2,
    loss_DEEP2_deriv,
    loss_SDSS,
    loss_SDSS_deriv,
    get_loss_data_COSMOS,
    get_loss_data_HSC,
)

from diffstarpop.lightcone_colors import (
    DEFAULT_lgfburst_u_params,
    DEFAULT_burstshape_u_params,
    DEFAULT_lgav_dust_u_params,
    DEFAULT_delta_dust_u_params,
    DEFAULT_boris_dust_u_params,
)


lightcone_path = "/lcrc/project/galsampler/SMDPL/DR1/lightcones/"
fn = "survey_z0.02-2.00_x60.00_y60.00_0.h5"

with h5py.File(os.path.join(lightcone_path, fn), "r") as f:
    print(f.keys())

    redshift_obs = f["redshift_obs"][...]
    # redshift_snapshot = f["redshift_snapshot"][...]
    # redshift_true = f["redshift_true"][...]
    # mpeak_lightcone = f["mpeak"][...]
    # halo_id_lightcone = f["halo_id"][...]

fn = "sfh_crossmatch/survey_z0.02-2.00_x60.00_y60.00_0.sfh_crossmatched.h5"
with h5py.File(os.path.join(lightcone_path, fn), "r") as f:
    print(f.keys())

    # halo_id_crossmatch = f["halo_id"][...]
    # mpeak_crossmatch = f["mpeak"][...]
    sfr_history_all_prog = f["sfr_history_all_prog"][...]
    # sfr_history_main_prog = f["sfr_history_main_prog"][...]



nines = np.sum(sfr_history_all_prog == -999.0, axis=1)
last_sfr_value = sfr_history_all_prog[np.arange(len(nines)), -nines-1]
last_sfr_value_arr = np.ones_like(sfr_history_all_prog) * last_sfr_value[:, None]
gal_sfr_arr = np.where(sfr_history_all_prog==-999.0, last_sfr_value_arr, sfr_history_all_prog)
assert np.all(gal_sfr_arr >=0.0)

SMDPL_a = np.load("/lcrc/project/halotools/alarcon/data/scale_list_SMDPL.npy")
SMDPL_z = 1.0 / SMDPL_a - 1.0
SMDPL_t = Planck18.age(SMDPL_z).value

area_norm_HSC = 1.0 # area in sq.deg

init_params = np.concatenate((
    DEFAULT_lgfburst_u_params,
    DEFAULT_burstshape_u_params,
    DEFAULT_lgav_dust_u_params,
    DEFAULT_delta_dust_u_params,
    DEFAULT_boris_dust_u_params,
))

select = np.random.choice(len(redshift_obs), int(1e5), replace=False)
"""
print("Get loss_data COSMOS")
loss_data_COSMOS = get_loss_data_COSMOS(
    SMDPL_t,
    gal_sfr_arr[select],
    redshift_obs[select],
)
"""
print("Get loss_data HSC")
loss_data_HSC = get_loss_data_HSC(
    SMDPL_t,
    gal_sfr_arr[select],
    redshift_obs[select],
    area_norm_HSC,
)

init_params = np.concatenate((
    DEFAULT_lgfburst_u_params,
    DEFAULT_burstshape_u_params,
    DEFAULT_lgav_dust_u_params,
    DEFAULT_delta_dust_u_params,
    DEFAULT_boris_dust_u_params,
))

ran_key = jran.PRNGKey(np.random.randint(2**32))
"""
print("Get loss COSMOS")
loss_cosmos_val = loss_COSMOS(init_params, loss_data_COSMOS, ran_key)
loss_deriv_cosmos_val = loss_COSMOS_deriv(init_params, loss_data_COSMOS, ran_key)
"""
print("Get loss HSC")
loss_hsc_val = loss_HSC(init_params, loss_data_HSC, ran_key)
loss_deriv_hsc_val = loss_HSC_deriv(init_params, loss_data_HSC, ran_key)

assert False
N_BURST_F = len(DEFAULT_lgfburst_u_params)
N_BURST_SHAPE = len(DEFAULT_burstshape_u_params)
N_DUST_LGAV = len(DEFAULT_lgav_dust_u_params)
N_DUST_DELTA = len(DEFAULT_delta_dust_u_params)
N_DUST_BORIS = len(DEFAULT_boris_dust_u_params)



npar = 0
print("BURST_F grads:", loss_deriv_cosmos_val[npar:npar+N_BURST_F])
npar += N_BURST_F
print("BURST_SHAPE grads:", loss_deriv_cosmos_val[npar:npar+N_BURST_SHAPE])
npar += N_BURST_SHAPE
print("DUST_LGAV grads:", loss_deriv_cosmos_val[npar:npar+N_DUST_LGAV])
npar += N_DUST_LGAV
print("DUST_DELTA grads:", loss_deriv_cosmos_val[npar:npar+N_DUST_DELTA])
npar += N_DUST_DELTA
print("DUST_BORIS grads:", loss_deriv_cosmos_val[npar:npar+N_DUST_BORIS])
npar += N_DUST_BORIS

from jax import config
config.update("jax_debug_nans", True)
loss_deriv_cosmos_val = loss_COSMOS_deriv(init_params, loss_data_COSMOS, ran_key)
