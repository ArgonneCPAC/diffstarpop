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
from collections import OrderedDict
from halotools.utils import crossmatch

from diffstarpop.desc_colors_DSPS import (
    loss_COSMOS,
    loss_COSMOS_deriv,
    loss_HSC,
    loss_HSC_deriv,
    loss_DEEP2,
    loss_DEEP2_deriv,
    loss_SDSS,
    loss_SDSS_deriv,
    loss_combined,
    loss_combined_deriv,
    get_loss_data_COSMOS,
    get_loss_data_HSC,
    get_loss_data_DEEP2,
    get_loss_data_SDSS,
)

from diffstarpop.lightcone_colors import (
    DEFAULT_lgfburst_u_params,
    DEFAULT_burstshape_u_params,
    DEFAULT_lgav_dust_u_params,
    DEFAULT_delta_dust_u_params,
    DEFAULT_boris_dust_u_params,
)

from fit_adam_helpers import jax_adam_wrapper
from jax.example_libraries import optimizers as jax_opt


lightcone_path = "/lcrc/project/galsampler/SMDPL/DR1/lightcones/"
root_dir = "/lcrc/project/galsampler/SMDPL/dr1_no_merging_upidh/"
diffmah_path = root_dir + "sfh_binary_catalogs/diffmah_fits/subvol_%d/diffmah_fits.h5"


def get_diffmah_params(halo_ids):
    for subvol in range(0, 576):
        df = []
        diffmah_params = OrderedDict()
        fn = diffmah_path % subvol
        with h5py.File(fn, "r") as hdf:
            # print(hdf.keys())
            # print(len(hdf["halo_id"]))
            for key in hdf.keys():
                diffmah_params[key] = hdf[key][...]
            subdf = pd.DataFrame(diffmah_params)
            subdf = subdf[subdf.halo_id.isin(halo_ids)]
            df.append(subdf)
        df = pd.concat(df, ignore_index=True)
        idx_reorder, idx_df = crossmatch(halo_ids, df.halo_id.values)
    return df.iloc[idx_df], idx_reorder


redshift_obs = []
sfr_history_exsitu = []
halo_id_crossmatch = []
for light_id in range(3):
    fn = f"survey_z0.02-2.00_x60.00_y60.00_{light_id}.h5"
    with h5py.File(os.path.join(lightcone_path, fn), "r") as f:
        print(f.keys())
        _redshift_obs = f["redshift_obs"][...]
        # redshift_snapshot = f["redshift_snapshot"][...]
        # redshift_true = f["redshift_true"][...]
        # mpeak_lightcone = f["mpeak"][...]
        # halo_id_lightcone = f["halo_id"][...]

    fn = (
        f"sfh_crossmatch/survey_z0.02-2.00_x60.00_y60.00_{light_id}.sfh_crossmatched.h5"
    )
    with h5py.File(os.path.join(lightcone_path, fn), "r") as f:
        print(f.keys())

        _halo_id_crossmatch = f["halo_id"][...]
        # mpeak_crossmatch = f["mpeak"][...]
        _logmp_crossmatch = f["logmp"][...]
        _sfr_history_all_prog = f["sfr_history_all_prog"][...]
        _sfr_history_main_prog = f["sfr_history_main_prog"][...]
        _sfr_history_exsitu = np.where(
            _sfr_history_all_prog == -999.0,
            -999.0,
            _sfr_history_all_prog - _sfr_history_main_prog,
        )

    mask = _logmp_crossmatch > 11.0
    redshift_obs.append(_redshift_obs[mask])
    sfr_history_exsitu.append(_sfr_history_exsitu[mask])
    halo_id_crossmatch.append(_halo_id_crossmatch[mask])

sfr_history_all_prog = np.concatenate(sfr_history_exsitu)
redshift_obs = np.concatenate(redshift_obs)
halo_id_crossmatch = np.concatenate(halo_id_crossmatch)

diffmah_df, idx_reorder = get_diffmah_params(halo_id_crossmatch)

assert False

halo_mah_params_arr = np.array(
    [
        diffmah_df,
    ]
)

sfr_history_exsitu = sfr_history_exsitu[idx_reorder]
redshift_obs = redshift_obs[idx_reorder]
halo_id_crossmatch = halo_id_crossmatch[idx_reorder]

nines = np.sum(sfr_history_exsitu == -999.0, axis=1)
last_sfr_value = sfr_history_exsitu[np.arange(len(nines)), -nines - 1]
last_sfr_value_arr = np.ones_like(sfr_history_exsitu) * last_sfr_value[:, None]
gal_sfr_exsitu_arr = np.where(
    sfr_history_exsitu == -999.0, last_sfr_value_arr, sfr_history_exsitu
)
assert np.all(gal_sfr_exsitu_arr >= 0.0)

SMDPL_a = np.load("/lcrc/project/halotools/alarcon/data/scale_list_SMDPL.npy")
SMDPL_z = 1.0 / SMDPL_a - 1.0
SMDPL_t = Planck18.age(SMDPL_z).value

area_norm_HSC = 10.0  # area in sq.deg

# select = np.random.choice(len(redshift_obs), int(1e5), replace=False)
# select = logmp_crossmatch > 11.0
# gal_sfr_exsitu_arr = gal_sfr_exsitu_arr[select]
# redshift_obs = redshift_obs[select]

select = np.random.choice(len(redshift_obs), int(1e5), replace=False)
undersampling_factor = len(select) / len(redshift_obs)

# assert False
print("Get loss_data COSMOS")
loss_data_COSMOS = get_loss_data_COSMOS(
    SMDPL_t,
    gal_sfr_exsitu_arr[select],
    redshift_obs[select],
    halo_mah_params_arr[select],
)
"""
print("Get loss_data HSC")
loss_data_HSC = get_loss_data_HSC(
    SMDPL_t,
    gal_sfr_exsitu_arr[select],
    redshift_obs[select],
    halo_mah_params_arr[select], 
    area_norm_HSC * undersampling_factor,
)

print("Get loss_data DEEP2")
loss_data_DEEP2 = get_loss_data_DEEP2(
    SMDPL_t,
    gal_sfr_exsitu_arr[select],
    redshift_obs[select],
    halo_mah_params_arr[select], 
)
"""
init_params = np.concatenate(
    (
        DEFAULT_lgfburst_u_params,
        DEFAULT_burstshape_u_params,
        DEFAULT_lgav_dust_u_params,
        DEFAULT_delta_dust_u_params,
        DEFAULT_boris_dust_u_params,
    )
)
# output_filename = "/lcrc/project/halotools/alarcon/results/DESC_mocks/vary_DSPS/test_combloss_mean_230831_test3.npz"
# output_filename = "/lcrc/project/halotools/alarcon/results/DESC_mocks/vary_DSPS/test_COSMOSloss_230901_test1.npz"
bestfit_params_file = np.load(output_filename)
init_params = bestfit_params_file["best_fit_params"]

ran_key = jran.PRNGKey(np.random.randint(2**32))
"""
print("Get loss COSMOS")
loss_cosmos_val = loss_COSMOS(init_params, loss_data_COSMOS, ran_key)
loss_deriv_cosmos_val = loss_COSMOS_deriv(init_params, loss_data_COSMOS, ran_key)
print("Loss COSMOS: %.5f"%loss_cosmos_val) 
print("Loss grads COSMOS:", loss_deriv_cosmos_val)

print("Get loss HSC")
loss_hsc_val = loss_HSC(init_params, loss_data_HSC, ran_key)
loss_deriv_hsc_val = loss_HSC_deriv(init_params, loss_data_HSC, ran_key)
print("Loss HSC: %.5f"%loss_hsc_val)
print("Loss grads HSC:", loss_deriv_hsc_val)

print("Get loss DEEP2")
loss_deep2_val = loss_DEEP2(init_params, loss_data_DEEP2, ran_key)
loss_deriv_deep2_val = loss_DEEP2_deriv(init_params, loss_data_DEEP2, ran_key)
print("Loss DEEP2: %.5f"%loss_deep2_val)
print("Loss grads DEEP2:", loss_deriv_deep2_val)
"""
"""
print("Get loss Combined")
loss_data_comb = (
    loss_data_COSMOS,
    loss_data_HSC,
    loss_data_DEEP2,
)
t0 = time.time()
loss_comb_val = loss_combined(init_params, loss_data_comb, ran_key)
loss_deriv_comb_val = loss_combined_deriv(init_params, loss_data_comb, ran_key)
t1 = time.time()
print(t1-t0)
print("Loss Comb: %.5f"%loss_comb_val)
print("Loss grads Comb:", loss_deriv_comb_val)
"""
output_filename = "/lcrc/project/halotools/alarcon/results/DESC_mocks/vary_DSPS/test_COSMOSloss_230901_test1.npz"


n_step = int(1e4)
step_size = 0.01

opt_init, opt_update, get_params = jax_opt.adam(step_size)
opt_state = opt_init(init_params)
jax_optimizer = (opt_state, opt_update, get_params)

print(f"Running grad descend")
"""
_res0 = jax_adam_wrapper(
    init_params,
    loss_data_comb,
    loss_combined,
    loss_combined_deriv,
    n_step,
    ran_key,
    output_filename,
    jax_optimizer,

)
"""
_res0 = jax_adam_wrapper(
    init_params,
    loss_data_COSMOS,
    loss_COSMOS,
    loss_COSMOS_deriv,
    n_step,
    ran_key,
    output_filename,
    jax_optimizer,
)
best_fit_params = _res0[0]

assert False
N_BURST_F = len(DEFAULT_lgfburst_u_params)
N_BURST_SHAPE = len(DEFAULT_burstshape_u_params)
N_DUST_LGAV = len(DEFAULT_lgav_dust_u_params)
N_DUST_DELTA = len(DEFAULT_delta_dust_u_params)
N_DUST_BORIS = len(DEFAULT_boris_dust_u_params)


npar = 0
print("BURST_F grads:", loss_deriv_cosmos_val[npar : npar + N_BURST_F])
npar += N_BURST_F
print("BURST_SHAPE grads:", loss_deriv_cosmos_val[npar : npar + N_BURST_SHAPE])
npar += N_BURST_SHAPE
print("DUST_LGAV grads:", loss_deriv_cosmos_val[npar : npar + N_DUST_LGAV])
npar += N_DUST_LGAV
print("DUST_DELTA grads:", loss_deriv_cosmos_val[npar : npar + N_DUST_DELTA])
npar += N_DUST_DELTA
print("DUST_BORIS grads:", loss_deriv_cosmos_val[npar : npar + N_DUST_BORIS])
npar += N_DUST_BORIS

from jax import config

config.update("jax_debug_nans", True)
loss_deriv_cosmos_val = loss_COSMOS_deriv(init_params, loss_data_COSMOS, ran_key)
