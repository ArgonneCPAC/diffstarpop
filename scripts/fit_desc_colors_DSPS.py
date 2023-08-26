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

from diffstarpop.desc_colors_DSPS import (
    loss_COSMOS,
    loss_COSMOS_deriv,
    loss_DEEP2,
    loss_DEEP2_deriv,
    loss_SDSS,
    loss_SDSS_deriv,
    get_loss_data_COSMOS,
)

lightcone_path = "/lcrc/project/galsampler/SMDPL/DR1/lightcones/"
fn = "survey_z0.02-2.00_x60.00_y60.00_0.h5"

with h5py.File(os.path.join(lightcone_path, fn), "r") as f:
    print(f.keys())

    redshift_obs = f["redshift_obs"][...]
    redshift_snapshot = f["redshift_snapshot"][...]
    redshift_true = f["redshift_true"][...]
    mpeak_lightcone = f["mpeak"][...]
    halo_id_lightcone = f["halo_id"][...]

fn = "sfh_crossmatch/survey_z0.02-2.00_x60.00_y60.00_0.sfh_crossmatched.h5"
with h5py.File(os.path.join(lightcone_path, fn), "r") as f:
    print(f.keys())

    redshift_obs = f["redshift_obs"][...]
    halo_id_crossmatch = f["halo_id"][...]
    mpeak_crossmatch = f["mpeak"][...]
    sfr_history_all_prog = f["sfr_history_all_prog"][...]
    sfr_history_main_prog = f["sfr_history_main_prog"][...]


assert False

get_loss_data_COSMOS(
    t_table,
    gal_sfr_arr,
    gal_z_arr,
    area_norm_HSC,
)
