import numpy as np
from jax import numpy as jnp
from jax import jit as jjit
from jax import vmap
from jax import grad
from jax import random as jran
import time
from functools import partial


from diffmah.individual_halo_assembly import _calc_halo_history
from diffstar.constants import TODAY
from diffstar.stars import fstar_tools
import astropy.units as u
from astropy.cosmology import Planck13, z_at_value
from astropy.cosmology import Planck18

from dsps.cosmology import DEFAULT_COSMOLOGY
from dsps.data_loaders import load_ssp_templates
from dsps.data_loaders.load_filter_data import load_transmission_curve
from dsps.photometry.utils import interpolate_filter_trans_curves
from dsps.photometry.photpop import precompute_ssp_obsmags_on_z_table

from diffstarpop.fit_lightcone_helpers import loss, loss_deriv
from diffstarpop.fit_lightcone_helpers import (
    N_PDF_Q,
    N_PDF_MS,
    N_R_Q,
    N_R_MS,
    print_loss_deriv,
)

from diffstarpop.json_utils import load_params
from diffstarpop.lightcone_colors import (
    DEFAULT_lgfburst_u_params,
    DEFAULT_burstshape_u_params,
    DEFAULT_lgav_dust_u_params,
    DEFAULT_delta_dust_u_params,
    DEFAULT_boris_dust_u_params,
    sumstats_lightcone_colors_1d,
)

# DSPS_data_path = "/Users/alarcon/Documents/DSPS_data/"
# DSPS_filter_path = "/Users/alarcon/Documents/DSPS_data/filters/"
DSPS_data_path = "/lcrc/project/halotools/alarcon/data/DSPS_data"
DSPS_filter_path = DSPS_data_path + "/filters/cosmos"


# Define our cosmci time array to make some predictions
t_table = np.linspace(0.1, TODAY, 10)
# t_table = np.logspace(-1, np.log10(TODAY), 100)

# t_table = np.linspace(1.0, TODAY, 20)
z_table = np.array([z_at_value(Planck13.age, x * u.Gyr, zmin=-1) for x in t_table])


print(t_table)
print(np.round(z_table, 2))
lgt = np.log10(t_table)
# Define some mass bins for predictions
logm0_bins = np.arange(11.0 - 0.05, 15.05 + 0.05, 0.1)
logm0_binmids = np.arange(11.0, 15.01, 0.3)
logm0_bin_widths = np.ones_like(logm0_binmids) * np.diff(logm0_binmids)[0] / 2.0


# Define some useful quantities and masks for later
fstar_tdelay = 1.0
index_select, index_high = fstar_tools(t_table, fstar_tdelay=fstar_tdelay)

path = "/lcrc/project/halotools/alarcon/data/"
"""
mah_params_arr = np.load(path+"mah_params_arr_576_small.npy")
u_fit_params_arr = np.load(path+"u_fit_params_arr_576_small.npy")
fit_params_arr = np.load(path+"fit_params_arr_576_small.npy")
p50_arr = np.load(path+"p50_arr_576_small.npy")
"""
mah_params_arr = np.load(path + "mah_params_arr_576_all.npy")
u_fit_params_arr = np.load(path + "u_fit_params_arr_576_all.npy")
fit_params_arr = np.load(path + "fit_params_arr_576_all.npy")
p50_arr = np.load(path + "p50_arr_576_all.npy")


logmpeak = mah_params_arr[:, 1]

pm0 = np.histogram(logmpeak, logm0_bins)[0]
pm0 = pm0 / pm0.sum()


Nhalos = 3000
halo_data_MC = []
p50 = []
for i in range(len(logm0_binmids)):
    _sel = (logmpeak > logm0_binmids[i] - logm0_bin_widths[i]) & (
        logmpeak < logm0_binmids[i] + logm0_bin_widths[i]
    )
    print(_sel.sum())
    replace = True if _sel.sum() < Nhalos else False
    sel = np.random.choice(np.arange(len(p50_arr))[_sel], Nhalos, replace=replace)
    halo_data_MC.append(mah_params_arr[sel])
    p50.append(p50_arr[sel])
halo_data_MC = np.concatenate(halo_data_MC, axis=0)[:, np.array([1, 2, 4, 5])]
p50 = np.concatenate(p50, axis=0)


bin_edges_mag = np.linspace(18, 26, 100)
bins_LO_mag = bin_edges_mag[:-1]
bins_HI_mag = bin_edges_mag[1:]
bin_centers_mag = 0.5 * (bins_LO_mag + bins_HI_mag)
delta_bin_mag = np.diff(bin_edges_mag)[0]

bin_edges_color = np.linspace(-1.0, 2.5, 100)
bins_LO_color = bin_edges_color[:-1]
bins_HI_color = bin_edges_color[1:]
bin_centers_color = 0.5 * (bins_LO_color + bins_HI_color)
delta_bin_color = np.diff(bin_edges_color)[0]


def precompute_DSPS_data(z_arr, filter_list):
    cosmo_params = DEFAULT_COSMOLOGY

    # Load ssp data
    ssp_data = load_ssp_templates(drn=DSPS_data_path)
    ssp_lgmet = ssp_data[0]
    ssp_lg_age_gyr = ssp_data[1]
    ssp_waves = ssp_data[2]
    ssp_spectra = ssp_data[3]

    filter_data = [
        load_transmission_curve(drn=DSPS_filter_path, bn_pat=filt)
        for filt in filter_list
    ]
    wave_filters = [x[0] for x in filter_data]
    trans_filters = [x[1] for x in filter_data]
    filter_waves, filter_trans = interpolate_filter_trans_curves(
        wave_filters, trans_filters
    )

    ssp_obs_photflux_table_arr = []
    for z_obs in z_arr:
        # Precompute ssp mags for zobs
        args = [
            ssp_waves,
            ssp_spectra,
            filter_waves,
            filter_trans,
            np.array((z_obs,)),
            *cosmo_params,
        ]
        ssp_obs_photmags = precompute_ssp_obsmags_on_z_table(*args)[0]
        ssp_obs_photflux_table = 10 ** (-0.4 * ssp_obs_photmags)
        ssp_obs_photflux_table_arr.append(ssp_obs_photflux_table)

    ssp_obs_photflux_table_arr = np.array(ssp_obs_photflux_table_arr)

    dsps_data = (
        filter_waves,
        filter_trans,
        ssp_lgmet,
        ssp_lg_age_gyr,
        cosmo_params,
    )

    return ssp_obs_photflux_table_arr, dsps_data


z_arr = np.arange(0.1, 2.01, 0.3)
filter_list = [
    "COSMOS_CFHT_ustar.h5",
    "COSMOS_HSC_g.h5",
    "COSMOS_HSC_r.h5",
    "COSMOS_HSC_i.h5",
    "COSMOS_HSC_z.h5",
    "COSMOS_UVISTA_Y.h5",
    "COSMOS_UVISTA_J.h5",
    "COSMOS_UVISTA_H.h5",
    "COSMOS_UVISTA_Ks.h5",
]

ssp_obs_photflux_table_arr, dsps_data = precompute_DSPS_data(z_arr, filter_list)


target_data = np.load(
    "/lcrc/project/halotools/alarcon/data/counts_target_1d_colors_i_18_23_230720.npy"
)

# path_json = "/Users/alarcon/Documents/source/diffstarpop/diffstarpop/bestfit_diffstarpop_params_UM_hists_v4.json"
path_json = "/home/aalarcongonzalez/source/diffstarpop/diffstarpop/bestfit_diffstarpop_params_UM_hists_v4.json"

outputs = load_params(path_json)
init_pdf_q_params = outputs[0][0:N_PDF_Q]
init_pdf_ms_params = outputs[0][N_PDF_Q : N_PDF_Q + N_PDF_MS]
init_r_q_params = outputs[0][N_PDF_Q + N_PDF_MS : N_PDF_Q + N_PDF_MS + N_R_Q]
init_r_ms_params = outputs[0][
    N_PDF_Q + N_PDF_MS + N_R_Q : N_PDF_Q + N_PDF_MS + N_R_Q + N_R_MS
]

init_params = jnp.concatenate(
    (
        init_pdf_q_params,
        init_pdf_ms_params,
        init_r_q_params,
        init_r_ms_params,
        DEFAULT_lgfburst_u_params,
        DEFAULT_burstshape_u_params,
        DEFAULT_lgav_dust_u_params,
        DEFAULT_delta_dust_u_params,
        DEFAULT_boris_dust_u_params,
    )
)

dVdz = Planck18.differential_comoving_volume(z_arr).value
dVdz = dVdz / dVdz.sum()

ran_key = jran.PRNGKey(np.random.randint(2**32))
n_histories = int(1e2)

ndsig_color = np.ones(2 * n_histories) * delta_bin_color
ndsig_mag = np.ones(2 * n_histories) * delta_bin_mag

loss_data = (
    t_table,
    logm0_binmids,
    halo_data_MC.reshape(len(logm0_binmids), Nhalos, 4),
    p50.reshape(len(logm0_binmids), Nhalos),
    pm0,
    index_select,
    index_high,
    fstar_tdelay,
    ndsig_mag,
    ndsig_color,
    bins_LO_mag,
    bins_HI_mag,
    bins_LO_color,
    bins_HI_color,
    z_arr,
    dVdz,
    dsps_data,
    ssp_obs_photflux_table_arr,
    target_data,
)

t0 = time.time()
loss_value = loss(init_params, loss_data, n_histories, ran_key)
t1 = time.time()
print(t1 - t0)
print("loss:", loss_value)

t0 = time.time()
loss_deriv_values = loss_deriv(init_params, loss_data, n_histories, ran_key)
t1 = time.time()
print(t1 - t0)
print_loss_deriv(loss_deriv_values)
