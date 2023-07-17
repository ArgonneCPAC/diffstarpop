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

# Define our cosmci time array to make some predictions
t_table = np.linspace(0.1, TODAY, 100)
# t_table = np.logspace(-1, np.log10(TODAY), 100)

# t_table = np.linspace(1.0, TODAY, 20)
z_table = np.array([z_at_value(Planck13.age, x * u.Gyr, zmin=-1) for x in t_table])


print(t_table)
print(np.round(z_table,2))
lgt = np.log10(t_table)
# Define some mass bins for predictions
logm0_bins = np.arange(11.0-0.05 , 15.05+0.05, 0.1)
logm0_binmids = np.arange(11.0, 15.01, 0.1)
logm0_bin_widths = np.ones_like(logm0_binmids) * np.diff(logm0_binmids)[0]/2.0


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
mah_params_arr = np.load(path+"mah_params_arr_576_all.npy")
u_fit_params_arr = np.load(path+"u_fit_params_arr_576_all.npy")
fit_params_arr = np.load(path+"fit_params_arr_576_all.npy")
p50_arr = np.load(path+"p50_arr_576_all.npy")


logmpeak = mah_params_arr[:,1]

pm0 = np.histogram(logmpeak, logm0_bins)[0]
pm0 = pm0 / pm0.sum()


Nhalos = 3000
halo_data_MC = []
p50 = []
for i in range(len(logm0_binmids)):
    _sel = (logmpeak > logm0_binmids[i] - logm0_bin_widths[i]) & (logmpeak < logm0_binmids[i] + logm0_bin_widths[i])
    print(_sel.sum())
    replace = True if _sel.sum() < Nhalos else False
    sel = np.random.choice(np.arange(len(p50_arr))[_sel], Nhalos, replace=replace)
    halo_data_MC.append(mah_params_arr[sel])
    p50.append(p50_arr[sel])
halo_data_MC = np.concatenate(halo_data_MC, axis=0)[:, np.array([1,2,4,5])]
p50 = np.concatenate(p50, axis=0)


bins_color = jnp.linspace(-1, 2.5, 25)
bins_magn = jnp.linspace(15, 25, 25)
delta_bins_color = float(jnp.diff(bins_color)[0])
delta_bins_magn = float(jnp.diff(bins_magn)[0])

bins_LO_colcol = []
bins_HI_colcol = []
for i in range(len(bins_color)-1):
    for j in range(len(bins_color)-1):
        bins_LO_colcol.append([bins_color[i], bins_color[j]])
        bins_HI_colcol.append([bins_color[i+1], bins_color[j+1]])
bins_LO_colcol = np.array(bins_LO_colcol)
bins_HI_colcol = np.array(bins_HI_colcol)
print(bins_LO_colcol.shape, bins_HI_colcol.shape)

bins_LO_colmag = []
bins_HI_colmag = []
for i in range(len(bins_magn)-1):
    for j in range(len(bins_color)-1):
        bins_LO_colmag.append([bins_magn[i], bins_color[j]])
        bins_HI_colmag.append([bins_magn[i+1], bins_color[j+1]])
bins_LO_colmag = np.array(bins_LO_colmag)
bins_HI_colmag = np.array(bins_HI_colmag)
print(bins_LO_colmag.shape, bins_HI_colmag.shape)


from diffstarpop.pdf_quenched import DEFAULT_SFH_PDF_QUENCH_PARAMS
from diffstarpop.pdf_mainseq import DEFAULT_SFH_PDF_MAINSEQ_PARAMS
from diffstarpop.pdf_model_assembly_bias_shifts import DEFAULT_R_MAINSEQ_PARAMS, DEFAULT_R_QUENCH_PARAMS

from diffstarpop.lightcone_colors import (
    sumstats_lightcone_colors_single_m0,
    sumstats_lightcone_colors_single_m0_vmap,
    sumstats_lightcone_colors,
)
from diffstarpop.lightcone_colors import (
    DEFAULT_lgfburst_u_params,
    DEFAULT_burstshape_u_params,
    DEFAULT_lgav_dust_u_params,
    DEFAULT_delta_dust_u_params,
    DEFAULT_boris_dust_u_params,
)


from diffstarpop.json_utils import load_params
from astropy.cosmology import Planck18

#path_json = "/Users/alarcon/Documents/source/diffstarpop/diffstarpop/bestfit_diffstarpop_params_UM_hists_v4.json"
path_json = "/home/aalarcongonzalez/source/diffstarpop/diffstarpop/bestfit_diffstarpop_params_UM_hists_v4.json"

outputs = load_params(path_json)

N_PDF_Q = len(DEFAULT_SFH_PDF_QUENCH_PARAMS)
N_PDF_MS = len(DEFAULT_SFH_PDF_MAINSEQ_PARAMS)
N_R_Q = len(DEFAULT_R_QUENCH_PARAMS)
N_R_MS = len(DEFAULT_R_MAINSEQ_PARAMS)



from dsps.cosmology import DEFAULT_COSMOLOGY
from dsps.data_loaders import load_ssp_templates
from dsps.data_loaders.load_filter_data import load_transmission_curve
from dsps.photometry.utils import interpolate_filter_trans_curves
from dsps.photometry.photpop import precompute_ssp_obsmags_on_z_table

#DSPS_data_path = "/Users/alarcon/Documents/DSPS_data/"
#DSPS_filter_path = "/Users/alarcon/Documents/DSPS_data/filters/"
DSPS_data_path = "/lcrc/project/halotools/alarcon/data/DSPS_data"
DSPS_filter_path = DSPS_data_path + "/filters/"



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


z_arr = np.arange(0.1, 2.01, 0.1)
filts = ["u", "g", "r", "i", "z"]
filter_list = [f"lsst_{x}*" for x in filts]
ssp_obs_photflux_table_arr, dsps_data = precompute_DSPS_data(z_arr, filter_list)


dVdz = Planck18.differential_comoving_volume(z_arr).value
dVdz = dVdz / dVdz.sum()

ran_key = jran.PRNGKey(np.random.randint(2**32))
n_histories = int(1e3)

ndsig_colcol = np.ones((2*n_histories, 2))
ndsig_colcol[:,0] *= delta_bins_color
ndsig_colcol[:,1] *= delta_bins_color

ndsig_colmag = np.ones((2*n_histories, 2))
ndsig_colmag[:,0] *= delta_bins_magn
ndsig_colmag[:,1] *= delta_bins_color

print(f"nt = {len(t_table)}")
print(f"n_histories = {n_histories}")
print(f"nm0 = {len(pm0)}")
print(f"nz = {len(z_arr)}")

t0 = time.time()
_res = sumstats_lightcone_colors(
    t_table,
    logm0_binmids,
    halo_data_MC.reshape(len(logm0_binmids),Nhalos,4),
    p50.reshape(len(logm0_binmids),Nhalos),
    pm0,
    n_histories,
    ran_key,
    index_select,
    index_high,
    fstar_tdelay,
    ndsig_colcol,
    ndsig_colmag,
    bins_LO_colcol,
    bins_HI_colcol,
    bins_LO_colmag,
    bins_HI_colmag,
    z_arr,
    dVdz,
    dsps_data,
    ssp_obs_photflux_table_arr,
    outputs[0][0:N_PDF_Q],
    outputs[0][N_PDF_Q:N_PDF_Q+N_PDF_MS],
    outputs[0][N_PDF_Q+N_PDF_MS:N_PDF_Q+N_PDF_MS+N_R_Q],
    outputs[0][N_PDF_Q+N_PDF_MS+N_R_Q:N_PDF_Q+N_PDF_MS+N_R_Q+N_R_MS],
    DEFAULT_lgfburst_u_params,
    DEFAULT_burstshape_u_params,
    DEFAULT_lgav_dust_u_params,
    DEFAULT_delta_dust_u_params,
    DEFAULT_boris_dust_u_params,
)
t1 = time.time()
print(f"Counts calculation: {t1-t0} seconds")

_target = jnp.array(_res)
target = np.zeros_like(_target)
target[:, :-1] = _target[:, 1:]
target[:, -1] = _target[:, 0]
target = jnp.array(target)

@partial(jjit, static_argnames=["n_histories"])
def loss(
    t_table,
    logmh_arr,
    mah_params_arr,
    p50_arr,
    pm0,
    n_histories,
    ran_key,
    index_select,
    index_high,
    fstar_tdelay,
    ndsig_colcol,
    ndsig_colmag,
    bins_LO_colcol,
    bins_HI_colcol,
    bins_LO_colmag,
    bins_HI_colmag,
    z_arr, 
    dVdz,
    dsps_data,
    ssp_obs_photflux_table_arr,
    pdf_parameters_Q,
    pdf_parameters_MS,
    R_model_params_Q,
    R_model_params_MS,
    lgfburst_u_params,
    burstshape_u_params,
    lgav_dust_u_params,
    delta_dust_u_params,
    boris_dust_u_params,
    target,
):
    prediction = sumstats_lightcone_colors(
        t_table,
        logmh_arr,
        mah_params_arr,
        p50_arr,
        pm0,
        n_histories,
        ran_key,
        index_select,
        index_high,
        fstar_tdelay,
        ndsig_colcol,
        ndsig_colmag,
        bins_LO_colcol,
        bins_HI_colcol,
        bins_LO_colmag,
        bins_HI_colmag,
        z_arr, 
        dVdz,
        dsps_data,
        ssp_obs_photflux_table_arr,
        pdf_parameters_Q,
        pdf_parameters_MS,
        R_model_params_Q,
        R_model_params_MS,
        lgfburst_u_params,
        burstshape_u_params,
        lgav_dust_u_params,
        delta_dust_u_params,
        boris_dust_u_params,
    )
    prediction = jnp.array(prediction)
    return jnp.sum((prediction - target)**2)
    

loss_deriv = jjit(grad(loss, argnums=tuple(np.arange(20,29,1))), static_argnames=["n_histories"])


t0 = time.time()
loss_value = loss(
    t_table,
    logm0_binmids,
    halo_data_MC.reshape(len(logm0_binmids),Nhalos,4),
    p50.reshape(len(logm0_binmids),Nhalos),
    pm0,
    n_histories,
    ran_key,
    index_select,
    index_high,
    fstar_tdelay,
    ndsig_colcol,
    ndsig_colmag,
    bins_LO_colcol,
    bins_HI_colcol,
    bins_LO_colmag,
    bins_HI_colmag,
    z_arr,
    dVdz,
    dsps_data,
    ssp_obs_photflux_table_arr,
    outputs[0][0:N_PDF_Q],
    outputs[0][N_PDF_Q:N_PDF_Q+N_PDF_MS],
    outputs[0][N_PDF_Q+N_PDF_MS:N_PDF_Q+N_PDF_MS+N_R_Q],
    outputs[0][N_PDF_Q+N_PDF_MS+N_R_Q:N_PDF_Q+N_PDF_MS+N_R_Q+N_R_MS],
    DEFAULT_lgfburst_u_params,
    DEFAULT_burstshape_u_params,
    DEFAULT_lgav_dust_u_params,
    DEFAULT_delta_dust_u_params,
    DEFAULT_boris_dust_u_params,
    target,
)
t1 = time.time()
print(f"Loss calculation: {t1-t0} seconds")
print(f"Loss value: {loss_value}")

t0 = time.time()
loss_deriv_value = loss_deriv(
    t_table,
    logm0_binmids,
    halo_data_MC.reshape(len(logm0_binmids),Nhalos,4),
    p50.reshape(len(logm0_binmids),Nhalos),
    pm0,
    n_histories,
    ran_key,
    index_select,
    index_high,
    fstar_tdelay,
    ndsig_colcol,
    ndsig_colmag,
    bins_LO_colcol,
    bins_HI_colcol,
    bins_LO_colmag,
    bins_HI_colmag,
    z_arr,
    dVdz,
    dsps_data,
    ssp_obs_photflux_table_arr,
    outputs[0][0:N_PDF_Q],
    outputs[0][N_PDF_Q:N_PDF_Q+N_PDF_MS],
    outputs[0][N_PDF_Q+N_PDF_MS:N_PDF_Q+N_PDF_MS+N_R_Q],
    outputs[0][N_PDF_Q+N_PDF_MS+N_R_Q:N_PDF_Q+N_PDF_MS+N_R_Q+N_R_MS],
    DEFAULT_lgfburst_u_params,
    DEFAULT_burstshape_u_params,
    DEFAULT_lgav_dust_u_params,
    DEFAULT_delta_dust_u_params,
    DEFAULT_boris_dust_u_params,
    target,
)
t1 = time.time()
print(f"Loss grad calculation: {t1-t0} seconds")
print(f"Loss grads: {loss_deriv_value}")
