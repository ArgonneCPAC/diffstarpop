import numpy as np

from jax import numpy as jnp
from jax import jit as jjit
from jax import vmap
from jax import grad
from jax import random as jran
import time

from jax.example_libraries import optimizers as jax_opt

from diffstarpop.monte_carlo_diff_halo_population import (
    sm_sfr_history_diffstar_scan_XsfhXmah_vmap,
    sm_sfr_history_diffstar_scan_MS_XsfhXmah_vmap,
    _jax_get_dt_array,
    calc_hist_mstar_ssfr,
)

from diffstarpop.fit_pop_helpers import (
    loss_hists_vmap,
    loss_hists_vmap_deriv,
)

from diffstar.constants import TODAY
from diffstar.stars import fstar_tools
import astropy.units as u
from astropy.cosmology import Planck13, z_at_value
from diffstarpop.json_utils import load_params

from fit_UM_histograms_helpers import calculate_SMDPL_sumstats, jax_adam_wrapper

t_table = np.linspace(1.0, TODAY, 20)
z_table = np.array([z_at_value(Planck13.age, x * u.Gyr, zmin=-1) for x in t_table])

t_sel_hists = np.argmin(
    np.subtract.outer(z_table, np.arange(0, 2.1, 0.5)) ** 2, axis=0
)[::-1]

print(t_table)
print(z_table)
lgt = np.log10(t_table)

# Define some mass bins for predictions
logm0_binmids = np.linspace(11.0, 14.5, 8)
logm0_bin_widths = np.ones_like(logm0_binmids) * 0.1

# Define some useful quantities and masks for later
fstar_tdelay = 1.0
index_select, index_high = fstar_tools(t_table, fstar_tdelay=fstar_tdelay)

path = "/lcrc/project/halotools/alarcon/data/"

mah_params_arr = np.load(path + "mah_params_arr_SMDPL_576_small.npy")
u_fit_params_arr = np.load(path + "u_fit_params_arr_SMDPL_576_small.npy")
fit_params_arr = np.load(path + "fit_params_arr_SMDPL_576_small.npy")
p50_arr = np.load(path + "p50_arr_SMDPL_576_small.npy")

logmpeak = mah_params_arr[:, 1]


bins_mstar = jnp.linspace(7, 12, 25)
bins_ssfr = jnp.linspace(-13, -8.5, 25)
delta_bins_mstar = float(jnp.diff(bins_mstar)[0])
delta_bins_ssfr = float(jnp.diff(bins_ssfr)[0])

bins_LO = []
bins_HI = []
for i in range(len(bins_mstar) - 1):
    for j in range(len(bins_ssfr) - 1):
        bins_LO.append([bins_mstar[i], bins_ssfr[j]])
        bins_HI.append([bins_mstar[i + 1], bins_ssfr[j + 1]])
bins_LO = np.array(bins_LO)
bins_HI = np.array(bins_HI)


MC_res_target = calculate_SMDPL_sumstats(
    t_table,
    logm0_binmids,
    logm0_bin_widths,
    mah_params_arr,
    u_fit_params_arr,
    p50_arr,
    bins_LO,
    bins_HI,
    delta_bins_mstar,
    delta_bins_ssfr,
    t_sel_hists,
)

Nhalos = 3000
halo_data_MC = []
p50 = []
for i in range(len(logm0_binmids)):
    _sel = logmpeak > logm0_binmids[i] - logm0_bin_widths[i]
    _sel &= logmpeak < logm0_binmids[i] + logm0_bin_widths[i]
    replace = True if _sel.sum() < Nhalos else False
    sel = np.random.choice(np.arange(len(p50_arr))[_sel], Nhalos, replace=replace)
    halo_data_MC.append(mah_params_arr[sel])
    p50.append(p50_arr[sel])
halo_data_MC = np.concatenate(halo_data_MC, axis=0)[:, np.array([1, 2, 4, 5])]
p50 = np.concatenate(p50, axis=0)


path_json = "../diffstarpop/bestfit_diffstarpop_params_UM_hists_v4.json"
outputs = load_params(path_json)


n_histories = int(1e3)
n_step = int(1e1)
step_size = 0.01


ndsig = np.ones((2 * n_histories, 2))
ndsig[:, 0] *= delta_bins_mstar
ndsig[:, 1] *= delta_bins_ssfr

loss_data = (
    t_table,
    logm0_binmids,
    halo_data_MC.reshape(len(logm0_binmids), Nhalos, 4),
    p50.reshape(len(logm0_binmids), Nhalos),
    index_select,
    index_high,
    fstar_tdelay,
    ndsig,
    bins_LO,
    bins_HI,
    t_sel_hists,
    MC_res_target,
)

init_guess = outputs[0].copy()
params_init = init_guess.copy()
ran_key = jran.PRNGKey(np.random.randint(2 ** 32))

opt_init, opt_update, get_params = jax_opt.adam(step_size)
opt_state = opt_init(params_init)
jax_optimizer = (opt_state, opt_update, get_params)

print(f"Running with {n_histories} histories")
_res0 = jax_adam_wrapper(
    params_init,
    loss_data,
    loss_hists_vmap,
    loss_hists_vmap_deriv,
    n_step,
    n_histories,
    ran_key,
    jax_optimizer,
)

best_fit_params = _res0[0]
# jax_optimizer = _res0[4]

#################################

opt_init, opt_update, get_params = jax_opt.adam(step_size)
opt_state = opt_init(best_fit_params)
jax_optimizer = (opt_state, opt_update, get_params)

n_histories = int(1e4)
n_step = int(1e1)
step_size = 0.01

ndsig = np.ones((2 * n_histories, 2))
ndsig[:, 0] *= delta_bins_mstar
ndsig[:, 1] *= delta_bins_ssfr

loss_data = (
    t_table,
    logm0_binmids,
    halo_data_MC.reshape(len(logm0_binmids), Nhalos, 4),
    p50.reshape(len(logm0_binmids), Nhalos),
    index_select,
    index_high,
    fstar_tdelay,
    ndsig,
    bins_LO,
    bins_HI,
    t_sel_hists,
    MC_res_target,
)

print(f"Running with {n_histories} histories")
_res1 = jax_adam_wrapper(
    params_init,
    loss_data,
    loss_hists_vmap,
    loss_hists_vmap_deriv,
    n_step,
    n_histories,
    ran_key,
    jax_optimizer,
)

best_fit_params1 = _res1[0]
# jax_optimizer = _res1[4]

#################################

opt_init, opt_update, get_params = jax_opt.adam(step_size)
opt_state = opt_init(best_fit_params1)
jax_optimizer = (opt_state, opt_update, get_params)

n_histories = int(1e5)
n_step = int(1e1)
step_size = 0.01

ndsig = np.ones((2 * n_histories, 2))
ndsig[:, 0] *= delta_bins_mstar
ndsig[:, 1] *= delta_bins_ssfr

loss_data = (
    t_table,
    logm0_binmids,
    halo_data_MC.reshape(len(logm0_binmids), Nhalos, 4),
    p50.reshape(len(logm0_binmids), Nhalos),
    index_select,
    index_high,
    fstar_tdelay,
    ndsig,
    bins_LO,
    bins_HI,
    t_sel_hists,
    MC_res_target,
)

print(f"Running with {n_histories} histories")
_res2 = jax_adam_wrapper(
    params_init,
    loss_data,
    loss_hists_vmap,
    loss_hists_vmap_deriv,
    n_step,
    n_histories,
    ran_key,
    jax_optimizer,
)

best_fit_params2 = _res2[0]
# jax_optimizer = _res1[4]
