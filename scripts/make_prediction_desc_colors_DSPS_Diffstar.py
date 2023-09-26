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
from diffstarpop.json_utils import load_params

from diffstarpop.desc_colors_DSPS_Diffstar import (
    get_loss_data_COSMOS,
    get_loss_data_HSC,
    get_loss_data_DEEP2,
    get_loss_data_SDSS,
    predict_COSMOS,
    predict_HSC,
    predict_DEEP2,
)

from diffstarpop.lightcone_colors import (
    DEFAULT_lgfburst_u_params,
    DEFAULT_burstshape_u_params,
    DEFAULT_lgav_dust_u_params,
    DEFAULT_delta_dust_u_params,
    DEFAULT_boris_dust_u_params,
)

from diffstarpop.pdf_quenched import DEFAULT_SFH_PDF_QUENCH_PARAMS
from diffstarpop.pdf_mainseq import DEFAULT_SFH_PDF_MAINSEQ_PARAMS
from diffstarpop.pdf_model_assembly_bias_shifts import (
    DEFAULT_R_QUENCH_PARAMS,
    DEFAULT_R_MAINSEQ_PARAMS,
)

from fit_adam_helpers import jax_adam_wrapper
from jax.example_libraries import optimizers as jax_opt

N_PDF_Q = len(DEFAULT_SFH_PDF_QUENCH_PARAMS)
N_PDF_MS = len(DEFAULT_SFH_PDF_MAINSEQ_PARAMS)
N_R_Q = len(DEFAULT_R_QUENCH_PARAMS)
N_R_MS = len(DEFAULT_R_MAINSEQ_PARAMS)
N_BURST_F = len(DEFAULT_lgfburst_u_params)
N_BURST_SHAPE = len(DEFAULT_burstshape_u_params)
N_DUST_LGAV = len(DEFAULT_lgav_dust_u_params)
N_DUST_DELTA = len(DEFAULT_delta_dust_u_params)
N_DUST_BORIS = len(DEFAULT_boris_dust_u_params)

lightcone_path = "/lcrc/project/galsampler/SMDPL/DR1/lightcones/"
root_dir = "/lcrc/project/galsampler/SMDPL/dr1_no_merging_upidh/"
diffmah_path = root_dir + "sfh_binary_catalogs/diffmah_fits/subvol_%d/diffmah_fits.h5"
crossmatch_dir = "/lcrc/project/halotools/alarcon/data/"


def get_diffmah_params(halo_ids):
    df = []
    for subvol in range(0, 576):
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
    breakpoint()
    unique_idx = np.unique(df.halo_id.values, return_index=True)[1]
    df = df.iloc[unique_idx]
    idx_reorder, idx_df = crossmatch(halo_ids, df.halo_id.values)
    return df.iloc[idx_df], idx_reorder

def get_diffmah_paramsb_by_index(index_arr, subvol_arr):
    mah_cols = ['t0','logmp_fit', 'mah_logtc', 'mah_k', 'early_index', 'late_index']
    df = np.zeros((len(index_arr), len(mah_cols))) + np.nan
    for subvol in range(0, 576):
        diffmah_params = OrderedDict()
        fn = diffmah_path % subvol
        with h5py.File(fn, "r") as hdf:
            for key in hdf.keys():
                diffmah_params[key] = hdf[key][...]
        subdf = pd.DataFrame(diffmah_params)
        mask = subvol_arr==subvol
        df[mask] = subdf.loc[index_arr[mask], mah_cols].values
    df = pd.DataFrame(df, columns=mah_cols)
    return df


redshift_obs = []
sfr_history_exsitu = []
halo_id_at_z0 = []
halo_binary_index = []
halo_binary_subvol = []
count_all = 0
for light_id in range(10):
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

    fn = (
        crossmatch_dir + f"survey_z0.02-2.00_x60.00_y60.00_{light_id}.haloid_indices_v1.h5"
    )
    with h5py.File(fn, "r") as hdf:
        _halo_id_at_snapshot = hdf["halo_id_at_snapshot"][...]
        _halo_id_at_z0 = hdf["halo_id_at_z0"][...]
        _halo_binary_index = hdf["binary_snapshot_index"][...]
        _halo_binary_subvol = hdf["binary_subvol_index"][...]
    mask = _logmp_crossmatch > 11.0
    count_all += mask.sum()
    _redshift_obs = _redshift_obs[mask]
    _sfr_history_exsitu = _sfr_history_exsitu[mask]
    _halo_id_crossmatch = _halo_id_crossmatch[mask]
    # assert False
    
    unique_indices = np.unique(_halo_id_at_snapshot, return_index=True)[1]
    # df_unique_crossmatch = pd.DataFrame({"halo_id_at_snapshot":_halo_id_at_snapshot[unique_indices], "halo_id_at_z0":_halo_id_at_z0[unique_indices]})
    # df_lightcone = pd.DataFrame({"halo_id_crossmatch":_halo_id_crossmatch})
    _halo_id_at_snapshot = _halo_id_at_snapshot[unique_indices]
    _halo_id_at_z0 = _halo_id_at_z0[unique_indices]
    _halo_binary_index = _halo_binary_index[unique_indices]
    _halo_binary_subvol = _halo_binary_subvol[unique_indices]
    idx_x, idx_y = crossmatch(_halo_id_crossmatch, _halo_id_at_snapshot)

    _halo_id_crossmatch = _halo_id_crossmatch[idx_x]
    _redshift_obs = _redshift_obs[idx_x]
    _sfr_history_exsitu = _sfr_history_exsitu[idx_x]

    _halo_id_at_snapshot = _halo_id_at_snapshot[idx_y]
    _halo_id_at_z0 = _halo_id_at_z0[idx_y]
    _halo_binary_index = _halo_binary_index[idx_y]
    _halo_binary_subvol = _halo_binary_subvol[idx_y]
    assert np.allclose(_halo_id_crossmatch, _halo_id_at_snapshot)

    redshift_obs.append(_redshift_obs)
    sfr_history_exsitu.append(_sfr_history_exsitu)
    halo_id_at_z0.append(_halo_id_at_z0)
    halo_binary_index.append(_halo_binary_index)
    halo_binary_subvol.append(_halo_binary_subvol)


sfr_history_exsitu = np.concatenate(sfr_history_exsitu)
redshift_obs = np.concatenate(redshift_obs)
halo_id_at_z0 = np.concatenate(halo_id_at_z0)
halo_binary_index = np.concatenate(halo_binary_index)
halo_binary_subvol = np.concatenate(halo_binary_subvol)

count_crossmatch = len(halo_id_at_z0)
crossmatch_downsampling = count_crossmatch / count_all

print("Getting diffmah parameters")
diffmah_df = get_diffmah_paramsb_by_index(halo_binary_index, halo_binary_subvol)
# diffmah_df, idx_reorder = get_diffmah_params(halo_id_at_z0)

# assert False
mah_cols = ['t0','logmp_fit', 'mah_logtc', 'mah_k', 'early_index', 'late_index']
halo_mah_params_arr = diffmah_df.loc[:,mah_cols].values
halo_mah_params_arr[:,0] = np.log10(halo_mah_params_arr[:,0])

# sfr_history_exsitu = sfr_history_exsitu[idx_reorder]
# redshift_obs = redshift_obs[idx_reorder]

sfr_history_exsitu = np.where((sfr_history_exsitu!=-999.0)&(sfr_history_exsitu < 0.0), 0.0, sfr_history_exsitu)

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

select = np.random.choice(len(redshift_obs), int(5e5), replace=False)
undersampling_factor = len(select) / len(redshift_obs) * crossmatch_downsampling

# assert False
print("Get loss_data COSMOS")
loss_data_COSMOS = get_loss_data_COSMOS(
    SMDPL_t,
    gal_sfr_exsitu_arr[select],
    redshift_obs[select],
    halo_mah_params_arr[select],
)
# assert False

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
path_json = "/home/aalarcongonzalez/source/diffstarpop/diffstarpop/bestfit_diffstarpop_params_UM_hists_v4.json"

outputs = load_params(path_json)
init_pdf_q_params = outputs[0][0:N_PDF_Q]
init_pdf_ms_params = outputs[0][N_PDF_Q : N_PDF_Q + N_PDF_MS]
init_r_q_params = outputs[0][N_PDF_Q + N_PDF_MS : N_PDF_Q + N_PDF_MS + N_R_Q]
init_r_ms_params = outputs[0][
    N_PDF_Q + N_PDF_MS + N_R_Q : N_PDF_Q + N_PDF_MS + N_R_Q + N_R_MS
]
output_filename = "/lcrc/project/halotools/alarcon/results/DESC_mocks/vary_DSPS/test_COSMOSloss_230901_test1.npz"
bestfit_params_file = np.load(output_filename)

_npar = 0
init_lgfburst_u_params = bestfit_params_file["best_fit_params"][_npar : _npar + N_BURST_F]
_npar += N_BURST_F
init_burstshape_u_params = bestfit_params_file["best_fit_params"][_npar : _npar + N_BURST_SHAPE]
_npar += N_BURST_SHAPE
init_lgav_dust_u_params = bestfit_params_file["best_fit_params"][_npar : _npar + N_DUST_LGAV]
_npar += N_DUST_LGAV
init_delta_dust_u_params = bestfit_params_file["best_fit_params"][_npar : _npar + N_DUST_DELTA]
_npar += N_DUST_DELTA
init_boris_dust_u_params = bestfit_params_file["best_fit_params"][_npar : _npar + N_DUST_BORIS]

init_params = jnp.concatenate(
    (
        init_pdf_q_params,
        init_pdf_ms_params,
        init_r_q_params,
        init_r_ms_params,
        init_lgfburst_u_params,
        init_burstshape_u_params,
        init_lgav_dust_u_params,
        init_delta_dust_u_params,
        init_boris_dust_u_params,
    )
)
"""
output_filename = "/lcrc/project/halotools/alarcon/results/DESC_mocks/vary_DSPS_Diffstar/test_combloss_230914_test1.npz"
bestfit_params_file = np.load(output_filename)
init_params = bestfit_params_file["best_fit_params"]

ran_key = jran.PRNGKey(np.random.randint(2**32))


print("Predict COSMOS")
pred_cosmos = predict_COSMOS(init_params, loss_data_COSMOS, ran_key)

print("Predict HSC")
pred_hsc = predict_HSC(init_params, loss_data_HSC, ran_key)

print("Predict DEEP2")
pred_deep2 = predict_DEEP2(init_params, loss_data_DEEP2, ran_key)

out_path = "/lcrc/project/halotools/alarcon/results/DESC_mocks/vary_DSPS_Diffstar/"
# fn = "pred_init_params.h5"
fn = "pred_bestfit_params.h5"

with h5py.File(os.path.join(out_path, fn), "w") as f:
    f["pred_cosmos_diff_i_counts"] = pred_cosmos[:7]
    f["pred_cosmos_diff_color_counts"] = pred_cosmos[7:]
    f["pred_hsc_cumcounts_imag_tri"] = pred_hsc
    f["pred_deep2_dNdz_rmag"] = pred_deep2[:4]
    f["pred_deep2_dNdz_imag"] =pred_deep2[4:]
