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
    get_colors_array,
    interpolate_ssp_obs_photflux_table_batch,
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

from diffstarpop.utils import get_t50_p50


from diffstarpop.monte_carlo_diff_halo_population import draw_single_sfh_params
from diffsky.experimental.dspspop.photpop import get_obs_photometry_and_params_singlez

from fit_adam_helpers import jax_adam_wrapper
from jax.example_libraries import optimizers as jax_opt

from diffsky.experimental.photometry_interpolation import interpolate_ssp_photmag_table
from dsps.photometry.photpop import precompute_ssp_obsmags_on_z_table
from dsps.cosmology import DEFAULT_COSMOLOGY
from dsps.data_loaders import load_ssp_templates
from dsps.data_loaders.load_filter_data import load_transmission_curve
from dsps.photometry.utils import interpolate_filter_trans_curves

from diffmah.individual_halo_assembly import _calc_halo_history

_calc_halo_history_vmap = jjit(vmap(_calc_halo_history, in_axes=(None, *[0] * 6)))

from diffstar.constants import LGT0

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

DSPS_data_path = "/lcrc/project/halotools/alarcon/data/DSPS_data/"
DSPS_COSMOS_filter_path = DSPS_data_path + "filters/cosmos/"
DSPS_DEEP2_filter_path = DSPS_data_path + "filters/DEEP2_filters/"
DSPS_SDSS_filter_path = DSPS_data_path + "filters/SDSS_filters/"
target_data_path = "/lcrc/project/halotools/alarcon/data/DESC_mocks_data/"


def get_loss_data(
    t_table,
    gal_z_arr,
    halo_mah_params_arr,
):
    print("Loading HSC filters")
    filter_list_HSC = [
        "COSMOS_HSC_i.h5",
    ]

    def get_filter_data(filter_list, filter_path):
        filter_data = [
            load_transmission_curve(drn=filter_path, bn_pat=filt)
            for filt in filter_list
        ]
        wave_filters = [x[0] for x in filter_data]
        trans_filters = [x[1] for x in filter_data]
        filter_waves, filter_trans = interpolate_filter_trans_curves(
            wave_filters, trans_filters
        )
        return filter_waves, filter_trans

    filter_waves_HSC, filter_trans_HSC = get_filter_data(
        filter_list_HSC, DSPS_COSMOS_filter_path
    )
    print("Calculating SSPs in HSC filters in z_table")
    z_table = np.linspace(gal_z_arr.min() - 1e-3, gal_z_arr.max() + 1e-3, 100)

    ssp_data = load_ssp_templates(drn=DSPS_data_path)
    ssp_lgmet = ssp_data[0]
    ssp_lg_age_gyr = ssp_data[1]
    ssp_waves = ssp_data[2]
    ssp_spectra = ssp_data[3]

    ssp_obs_photflux_table_HSC = get_colors_array(
        z_table,
        ssp_waves,
        ssp_spectra,
        filter_waves_HSC,
        filter_trans_HSC,
        DEFAULT_COSMOLOGY,
    )
    print("Interpolating SSPs to galaxy redshifts")
    gal_ssp_table_HSC = interpolate_ssp_obs_photflux_table_batch(
        gal_z_arr, z_table, ssp_obs_photflux_table_HSC
    )

    print("Calculating p50...")
    log_mah = _calc_halo_history_vmap(jnp.log10(t_table), *halo_mah_params_arr.T)[1]
    log_mah = np.array(log_mah)
    window_length = int(0.1 * len(log_mah))
    window_length = window_length + 1 if window_length % 2 == 0 else window_length
    halo_p50_arr = get_t50_p50(
        t_table, 10**log_mah, 0.5, log_mah[:, -1], window_length=window_length
    )[1]

    halo_mah_params_arr_out = halo_mah_params_arr[:, np.array([1, 2, 4, 5])]

    loss_data = (
        t_table,
        gal_z_arr,
        halo_mah_params_arr_out,
        halo_p50_arr,
        gal_ssp_table_HSC,
        filter_waves_HSC,
        filter_trans_HSC,
        ssp_lgmet,
        ssp_lg_age_gyr,
        DEFAULT_COSMOLOGY,
    )

    return loss_data


@jjit
def get_colors_single_object(
    gal_t_table,
    gal_sfr_table,
    gal_z_obs,
    gal_ssp_obs_photflux_table,
    ran_key,
    filter_waves,
    filter_trans,
    ssp_lgmet,
    ssp_lg_age_gyr,
    cosmo_params,
    lgfburst_u_params,
    burstshape_u_params,
    lgav_dust_u_params,
    delta_dust_u_params,
    boris_dust_u_params,
):
    gal_sfr_table = jnp.atleast_2d(gal_sfr_table)

    res = get_obs_photometry_and_params_singlez(
        ran_key,
        filter_waves,
        filter_trans,
        gal_ssp_obs_photflux_table,
        ssp_lgmet,
        ssp_lg_age_gyr,
        gal_t_table,
        gal_sfr_table,
        lgfburst_u_params,
        burstshape_u_params,
        lgav_dust_u_params,
        delta_dust_u_params,
        boris_dust_u_params,
        cosmo_params,
        gal_z_obs,
    )
    """
    (
        weights,
        lgmet_weights,
        smooth_age_weights,
        bursty_age_weights,
        frac_trans,
        gal_obsflux_nodust,
        gal_obsflux,
    ) = res

    gal_obsflux = gal_obsflux[0]

    gal_obs_mags = -2.5 * jnp.log10(gal_obsflux)
    return gal_obs_mags
    """
    return res


_A = (None, 0, 0, 0, 0, *[None] * 10)
get_colors_pop = jjit(vmap(get_colors_single_object, in_axes=_A))


def get_diffmah_paramsb_by_index(index_arr, subvol_arr):
    mah_cols = ["t0", "logmp_fit", "mah_logtc", "mah_k", "early_index", "late_index"]
    df = np.zeros((len(index_arr), len(mah_cols))) + np.nan
    for subvol in range(0, 576):
        diffmah_params = OrderedDict()
        fn = diffmah_path % subvol
        with h5py.File(fn, "r") as hdf:
            for key in hdf.keys():
                diffmah_params[key] = hdf[key][...]
        subdf = pd.DataFrame(diffmah_params)
        mask = subvol_arr == subvol
        df[mask] = subdf.loc[index_arr[mask], mah_cols].values
    df = pd.DataFrame(df, columns=mah_cols)
    return df


light_id = 0
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

fn = crossmatch_dir + f"survey_z0.02-2.00_x60.00_y60.00_{light_id}.haloid_indices_v1.h5"
with h5py.File(fn, "r") as hdf:
    _halo_id_at_snapshot = hdf["halo_id_at_snapshot"][...]
    _halo_id_at_z0 = hdf["halo_id_at_z0"][...]
    _halo_binary_index = hdf["binary_snapshot_index"][...]
    _halo_binary_subvol = hdf["binary_subvol_index"][...]
mask = _logmp_crossmatch > 11.0
count_all = mask.sum()
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

halo_id_crossmatch = _halo_id_crossmatch[idx_x]
redshift_obs = _redshift_obs[idx_x]
sfr_history_exsitu = _sfr_history_exsitu[idx_x]

halo_id_at_snapshot = _halo_id_at_snapshot[idx_y]
halo_id_at_z0 = _halo_id_at_z0[idx_y]
halo_binary_index = _halo_binary_index[idx_y]
halo_binary_subvol = _halo_binary_subvol[idx_y]
assert np.allclose(halo_id_crossmatch, halo_id_at_snapshot)

count_crossmatch = len(halo_id_at_z0)
crossmatch_downsampling = count_crossmatch / count_all

print("Getting diffmah parameters")
diffmah_df = get_diffmah_paramsb_by_index(halo_binary_index, halo_binary_subvol)
# diffmah_df, idx_reorder = get_diffmah_params(halo_id_at_z0)

# assert False
mah_cols = ["t0", "logmp_fit", "mah_logtc", "mah_k", "early_index", "late_index"]
halo_mah_params_arr = diffmah_df.loc[:, mah_cols].values
halo_mah_params_arr[:, 0] = np.log10(halo_mah_params_arr[:, 0])


SMDPL_a = np.load("/lcrc/project/halotools/alarcon/data/scale_list_SMDPL.npy")
SMDPL_z = 1.0 / SMDPL_a - 1.0
SMDPL_t = Planck18.age(SMDPL_z).value

output_filename = "/lcrc/project/halotools/alarcon/results/DESC_mocks/vary_DSPS_Diffstar/test_combloss_230914_test1.npz"
bestfit_params_file = np.load(output_filename)
params = bestfit_params_file["best_fit_params"]

_npar = 0
pdf_q_u_params = params[_npar : _npar + N_PDF_Q]
_npar += N_PDF_Q
pdf_ms_u_params = params[_npar : _npar + N_PDF_MS]
_npar += N_PDF_MS
r_q_u_params = params[_npar : _npar + N_R_Q]
_npar += N_R_Q
r_ms_u_params = params[_npar : _npar + N_R_MS]
_npar += N_R_MS
lgfburst_u_params = params[_npar : _npar + N_BURST_F]
_npar += N_BURST_F
burstshape_u_params = params[_npar : _npar + N_BURST_SHAPE]
_npar += N_BURST_SHAPE
lgav_dust_u_params = params[_npar : _npar + N_DUST_LGAV]
_npar += N_DUST_LGAV
delta_dust_u_params = params[_npar : _npar + N_DUST_DELTA]
_npar += N_DUST_DELTA
boris_dust_u_params = params[_npar : _npar + N_DUST_BORIS]
_npar += N_DUST_BORIS


ran_key = jran.PRNGKey(np.random.randint(2**32))

indices = np.array_split(
    np.arange(len(redshift_obs)), max(int(len(redshift_obs) / 1e5), 1)
)

for indx in indices:
    sfh_key, phot_key, ran_key = jran.split(ran_key, 3)
    sfh_key_arr = jran.split(sfh_key, len(indx))
    phot_key_arr = jran.split(phot_key, len(indx))

    res = get_loss_data(
        SMDPL_t,
        redshift_obs[indx],
        halo_mah_params_arr[indx],
    )

    (
        gal_t_table,
        gal_z_obs,
        halo_mah_params_arr_out,
        halo_p50_arr,
        gal_ssp_table,
        filter_waves,
        filter_trans,
        ssp_lgmet,
        ssp_lg_age_gyr,
        cosmo_params,
    ) = res

    sfr_params, q_params, gal_sfr_table = draw_single_sfh_params(
        gal_t_table,
        halo_mah_params_arr_out,
        halo_p50_arr,
        sfh_key_arr,
        pdf_q_u_params,
        pdf_ms_u_params,
        r_q_u_params,
        r_ms_u_params,
    )

    res = get_colors_pop(
        gal_t_table,
        gal_sfr_table,
        gal_z_obs,
        gal_ssp_table,
        ran_key,
        filter_waves,
        filter_trans,
        ssp_lgmet,
        ssp_lg_age_gyr,
        cosmo_params,
        lgfburst_u_params,
        burstshape_u_params,
        lgav_dust_u_params,
        delta_dust_u_params,
        boris_dust_u_params,
    )

    (
        gal_fburst,
        gal_lgyr_peak,
        gal_lgyr_max,
        gal_eb,
        gal_dust_delta,
        gal_av,
        frac_unobs,
        gal_obsflux,
    ) = res
    assert False
