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
    get_loss_data_COSMOS,
    get_loss_data_HSC,
    get_loss_data_DEEP2,
    get_loss_data_SDSS,
    predict_COSMOS,
    predict_HSC,
    predict_DEEP2,
    predict_HSC_magi,
    predict_COSMOS_mags,
)

from diffstarpop.lightcone_colors import (
    DEFAULT_lgfburst_u_params,
    DEFAULT_burstshape_u_params,
    DEFAULT_lgav_dust_u_params,
    DEFAULT_delta_dust_u_params,
    DEFAULT_boris_dust_u_params,
)


lightcone_path = "/lcrc/project/galsampler/SMDPL/DR1/lightcones/"

redshift_obs = []
sfr_history_all_prog = []
for light_id in range(10):
    fn = f"survey_z0.02-2.00_x60.00_y60.00_{light_id}.h5"
    with h5py.File(os.path.join(lightcone_path, fn), "r") as f:
        print(f.keys())
        _redshift_obs = f["redshift_obs"][...]
        # redshift_snapshot = f["redshift_snapshot"][...]
        # redshift_true = f["redshift_true"][...]
        # mpeak_lightcone = f["mpeak"][...]
        # halo_id_lightcone = f["halo_id"][...]

    fn = f"sfh_crossmatch/survey_z0.02-2.00_x60.00_y60.00_{light_id}.sfh_crossmatched.h5"
    with h5py.File(os.path.join(lightcone_path, fn), "r") as f:
        print(f.keys())

        # halo_id_crossmatch = f["halo_id"][...]
        # mpeak_crossmatch = f["mpeak"][...]
        _logmp_crossmatch = f["logmp"][...]
        _sfr_history_all_prog = f["sfr_history_all_prog"][...]
        # sfr_history_main_prog = f["sfr_history_main_prog"][...]

    mask = _logmp_crossmatch > 11.0
    redshift_obs.append(_redshift_obs[mask])
    sfr_history_all_prog.append(_sfr_history_all_prog[mask])

sfr_history_all_prog = np.concatenate(sfr_history_all_prog)
redshift_obs = np.concatenate(redshift_obs)


nines = np.sum(sfr_history_all_prog == -999.0, axis=1)
last_sfr_value = sfr_history_all_prog[np.arange(len(nines)), -nines-1]
last_sfr_value_arr = np.ones_like(sfr_history_all_prog) * last_sfr_value[:, None]
gal_sfr_arr = np.where(sfr_history_all_prog==-999.0, last_sfr_value_arr, sfr_history_all_prog)
assert np.all(gal_sfr_arr >=0.0)

SMDPL_a = np.load("/lcrc/project/halotools/alarcon/data/scale_list_SMDPL.npy")
SMDPL_z = 1.0 / SMDPL_a - 1.0
SMDPL_t = Planck18.age(SMDPL_z).value

area_norm_HSC = 10.0 # area in sq.deg

init_params = np.concatenate((
    DEFAULT_lgfburst_u_params,
    DEFAULT_burstshape_u_params,
    DEFAULT_lgav_dust_u_params,
    DEFAULT_delta_dust_u_params,
    DEFAULT_boris_dust_u_params,
))
output_filename = "/lcrc/project/halotools/alarcon/results/DESC_mocks/vary_DSPS/test_combloss_mean_230831_test3.npz"
# output_filename = "/lcrc/project/halotools/alarcon/results/DESC_mocks/vary_DSPS/test_COSMOSloss_230901_test1.npz"
bestfit_params_file = np.load(output_filename)
init_params = bestfit_params_file["best_fit_params"]
# select = np.random.choice(len(redshift_obs), int(1e5), replace=False)                                                                                         
select = np.random.choice(len(redshift_obs), int(5e5), replace=False)
undersampling_factor = len(select) / len(redshift_obs)  

print("Get loss_data COSMOS")
loss_data_COSMOS = get_loss_data_COSMOS(
    SMDPL_t,
    gal_sfr_arr[select],
    redshift_obs[select],
)

print("Get loss_data HSC")
loss_data_HSC = get_loss_data_HSC(
    SMDPL_t,
    gal_sfr_arr[select],
    redshift_obs[select],
    area_norm_HSC * undersampling_factor,
)

print("Get loss_data DEEP2")
loss_data_DEEP2 = get_loss_data_DEEP2(
    SMDPL_t,
    gal_sfr_arr[select],
    redshift_obs[select],
)

ran_key = jran.PRNGKey(np.random.randint(2**32))

print("Predict COSMOS")
pred_cosmos = predict_COSMOS(init_params, loss_data_COSMOS, ran_key)

print("Predict HSC")
pred_hsc = predict_HSC(init_params, loss_data_HSC, ran_key)

print("Predict DEEP2")
pred_deep2 = predict_DEEP2(init_params, loss_data_DEEP2, ran_key)

# print("Predict mag_i HSC")
# mag_i = predict_HSC_magi(init_params, loss_data_HSC, ran_key)

# print("Predict COSMOS mags")
# pred_cosmos_mags = predict_COSMOS_mags(init_params, loss_data_COSMOS, ran_key)

out_path = "/lcrc/project/halotools/alarcon/results/DESC_mocks/vary_DSPS/"
# fn = "pred_init_params.h5"
fn = "pred_bestfit_params.h5"
# fn = "pred_bestfit_params_cosmos.h5"
with h5py.File(os.path.join(out_path, fn), "w") as f:
    f["pred_cosmos_diff_i_counts"] = pred_cosmos[:7]
    f["pred_cosmos_diff_color_counts"] = pred_cosmos[7:]
    f["pred_hsc_cumcounts_imag_tri"] = pred_hsc
    f["pred_deep2_dNdz_rmag"] = pred_deep2[:4]
    f["pred_deep2_dNdz_imag"] =pred_deep2[4:]

assert False

(
    gal_mags_z01_03,
    gal_mags_z03_05,
    gal_mags_z05_07,
    gal_mags_z07_09,
    gal_mags_z09_11,
    gal_mags_z11_13,
    gal_mags_z13_15,
) = pred_cosmos_mags

fn = "pred_cosmos_mags.h5"
with h5py.File(os.path.join(out_path, fn), "w") as f:
    f["gal_mags_z01_03"] = gal_mags_z01_03
    f["gal_mags_z03_05"] = gal_mags_z03_05
    f["gal_mags_z05_07"] = gal_mags_z05_07
    f["gal_mags_z07_09"] = gal_mags_z07_09
    f["gal_mags_z09_11"] = gal_mags_z09_11
    f["gal_mags_z11_13"] = gal_mags_z11_13
    f["gal_mags_z13_15"] = gal_mags_z13_15


assert False
out_path = "/lcrc/project/halotools/alarcon/results/DESC_mocks/vary_DSPS/"
np.savez(out_path+"objs_magi.npz", mag_i = mag_i, redshift_obs = redshift_obs[select], logmp=logmp_crossmatch[select])


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
