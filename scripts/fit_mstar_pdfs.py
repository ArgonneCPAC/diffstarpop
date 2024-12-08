import h5py
import numpy as np
from jax import (
    numpy as jnp,
    jit as jjit,
    random as jran,
    grad,
    vmap,
)
import argparse
from time import time
from scipy.optimize import minimize
from jax.example_libraries import optimizers as jax_opt

from collections import OrderedDict, namedtuple

from diffstar.defaults import TODAY, LGT0
from diffmah.diffmah_kernels import mah_halopop

from diffstarpop.loss_kernels.mstar_ssfr_loss_tpeak import (
    loss_mstar_ssfr_kern_tobs_grad_wrapper,
    UnboundParams,
)

from diffstarpop.loss_kernels.namedtuple_utils import (
    tuple_to_array,
    register_tuple_new_diffstarpop,
    array_to_tuple_new_diffstarpop,
)
from diffstarpop.kernels.defaults_tpeak_block_cov import (
    DEFAULT_DIFFSTARPOP_U_PARAMS,
    DEFAULT_DIFFSTARPOP_PARAMS,
    get_bounded_diffstarpop_params,
)

BEBOP_SMHM_MEAN_DATA = "/lcrc/project/halotools/alarcon/results/"

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-indir", help="input drn", type=str, default=BEBOP_SMHM_MEAN_DATA
    )
    parser.add_argument(
        "-outdir", help="output drn", type=str, default=BEBOP_SMHM_MEAN_DATA
    )
    parser.add_argument(
        "-make_plot", help="whether to make plot", type=bool, default=False
    )
    parser.add_argument(
        "-nhalos", help="Number of halos for fitting", type=int, default=100
    )

    args = parser.parse_args()
    indir = args.indir
    outdir = args.outdir
    make_plot = args.make_plot
    nhalos = args.nhalos

    # Load SMHM data ---------------------------------------------
    print("Loading SMHM data...")

    with h5py.File(indir + "smdpl_smhm.h5", "r") as hdf:
        redshift_targets = hdf["redshift_targets"][:]
        smhm_diff = hdf["smhm_diff"][:]
        smhm = hdf["smhm"][:]
        logmh_bins = hdf["logmh_bins"][:]
        age_targets = hdf["age_targets"][:]
        """ 
            hdfout["counts_diff"] = wcounts
            hdfout["hist_diff"] = whist
            hdfout["counts"] = counts
            hdfout["hist"] = hist
            hdfout["smhm_diff"] = whist / wcounts
            hdfout["smhm"] = hist / counts
            hdfout["logmh_bins"] = smhm_utils.LOGMH_BINS
            hdfout["subvol_used"] = subvol_used
            
        """

    logmh_binsc = 0.5 * (logmh_bins[1:] + logmh_bins[:-1])

    with h5py.File(indir + "smdpl_smhm_samples_haloes.h5", "r") as hdf:
        logmh_id = hdf["logmh_id"][:]
        logmh_val = hdf["logmh_id"][:]
        mah_params_samp = hdf["mah_params_samp"][:]
        ms_params_samp = hdf["ms_params_samp"][:]
        q_params_samp = hdf["q_params_samp"][:]
        t_peak_samp = hdf["t_peak_samp"][:]
        tobs_id = hdf["tobs_id"][:]
        tobs_val = hdf["tobs_val"][:]
        redshift_val = hdf["redshift_val"][:]

    tpeak_path = "/Users/alarcon/Documents/diffmah_data/tpeak/random_data_241007/"
    with h5py.File(tpeak_path + "smdpl_mstar_ssfr.h5", "r") as hdf:
        mstar_wcounts = hdf["mstar_wcounts"][:]
        mstar_counts = hdf["mstar_counts"][:]
        mstar_ssfr_wcounts_cent = hdf["mstar_ssfr_wcounts_cent"][:]
        mstar_ssfr_wcounts_sat = hdf["mstar_ssfr_wcounts_sat"][:]
        logssfr_bins_pdf = hdf["logssfr_bins_pdf"][:]
        logmstar_bins_pdf = hdf["logmstar_bins_pdf"][:]
        """
        hdfout["mstar_wcounts"] = mstar_wcounts
        hdfout["mstar_counts"] = mstar_counts
        hdfout["mstar_ssfr_wcounts_cent"] = mstar_ssfr_wcounts_cent
        hdfout["mstar_ssfr_wcounts_sat"] = mstar_ssfr_wcounts_sat
        hdfout["logmh_bins"] = smhm_utils.LOGMH_BINS
        hdfout["logmstar_bins_pdf"] = smhm_utils.LOGMSTAR_BINS_PDF
        hdfout["logssfr_bins_pdf"] = smhm_utils.LOGSSFR_BINS_PDF
        hdfout["redshift_targets"] = redshift_targets
        hdfout["age_targets"] = age_targets
        """

    logssfr_binsc_pdf = 0.5 * (logssfr_bins_pdf[1:] + logssfr_bins_pdf[:-1])
    logmstar_binsc_pdf = 0.5 * (logmstar_bins_pdf[1:] + logmstar_bins_pdf[:-1])

    # Create loss_data ---------------------------------------------
    print("Creating loss data...")

    ran_key = jran.PRNGKey(np.random.randint(2**32))

    lgmu_infall = -1.0
    logmhost_infall = 13.0
    gyr_since_infall = 2.0

    mah_params_data = []
    lomg0_data = []
    t_peak_data = []
    lgmu_infall_data = []
    logmhost_infall_data = []
    gyr_since_infall_data = []
    t_obs_targets = []
    mstar_counts_target = []

    tarr_logm0 = np.logspace(-1, LGT0, 50)

    for i in range(len(age_targets)):
        t_target = age_targets[i]

        for j in range(len(logmh_binsc)):
            sel = (tobs_id == i) & (logmh_id == j)

            if sel.sum() < nhalos:
                continue
            arange_sel = np.arange(len(tobs_id))[sel]
            arange_sel = np.random.choice(arange_sel, nhalos, replace=False)
            mah_params_data.append(mah_params_samp[:, arange_sel])
            t_peak_data.append(t_peak_samp[arange_sel])
            lgmu_infall_data.append(np.ones(len(arange_sel)) * lgmu_infall)
            logmhost_infall_data.append(np.ones(len(arange_sel)) * logmhost_infall)
            gyr_since_infall_data.append(np.ones(len(arange_sel)) * gyr_since_infall)
            t_obs_targets.append(t_target)
            mstar_counts_target.append(mstar_wcounts[i, j] / mstar_wcounts[i, j].sum())
            dmhdt_fit, log_mah_fit = mah_halopop(
                mah_params_samp[:, arange_sel].T,
                tarr_logm0,
                t_peak_samp[arange_sel],
                LGT0,
            )
            lomg0_data.append(log_mah_fit[:, -1])
        break

    mah_params_data = np.array(mah_params_data)
    lomg0_data = np.array(lomg0_data)
    t_peak_data = np.array(t_peak_data)
    lgmu_infall_data = np.array(lgmu_infall_data)
    logmhost_infall_data = np.array(logmhost_infall_data)
    gyr_since_infall_data = np.array(gyr_since_infall_data)
    t_obs_targets = np.array(t_obs_targets)
    mstar_counts_target = np.array(mstar_counts_target)

    ran_key_data = jran.split(ran_key, len(mstar_counts_target))
    loss_data = (
        mah_params_data,
        lomg0_data,
        t_peak_data,
        lgmu_infall_data,
        logmhost_infall_data,
        gyr_since_infall_data,
        ran_key_data,
        t_obs_targets,
        logmstar_bins_pdf,
        mstar_counts_target,
    )

    # Register params ---------------------------------------------

    unbound_params_dict = OrderedDict(diffstarpop_u_params=DEFAULT_DIFFSTARPOP_U_PARAMS)
    UnboundParams = namedtuple("UnboundParams", list(unbound_params_dict.keys()))
    register_tuple_new_diffstarpop(UnboundParams)
    all_u_params = UnboundParams(*list(unbound_params_dict.values()))

    # Run fitter ---------------------------------------------
    print("Running fitter...")

    params_init = tuple_to_array(all_u_params)
    loss_mstar_ssfr_kern_tobs_grad_wrapper(params_init, loss_data)

    start = time()

    n_step = int(1e4)
    step_size = 0.01

    loss_arr = np.zeros(n_step).astype("f4") + np.inf

    opt_init, opt_update, get_params = jax_opt.adam(step_size)
    opt_state = opt_init(params_init)

    n_params = len(params_init)
    params_arr = np.zeros((n_step, n_params)).astype("f4")

    n_mah = 100

    no_nan_grads_arr = np.zeros(n_step)
    for istep in range(n_step):
        start = time()
        ran_key, subkey = jran.split(ran_key, 2)

        p = np.array(get_params(opt_state))

        loss, grads = loss_mstar_ssfr_kern_tobs_grad_wrapper(p, loss_data)

        no_nan_params = np.all(np.isfinite(p))
        no_nan_loss = np.isfinite(loss)
        no_nan_grads = np.all(np.isfinite(grads))
        if ~no_nan_params | ~no_nan_loss | ~no_nan_grads:
            # break
            if istep > 0:
                indx_best = np.nanargmin(loss_arr[:istep])
                best_fit_params = params_arr[indx_best]
                best_fit_loss = loss_arr[indx_best]
            else:
                best_fit_params = np.copy(p)
                best_fit_loss = 999.99
        else:
            params_arr[istep, :] = p
            loss_arr[istep] = loss
            opt_state = opt_update(istep, grads, opt_state)

        no_nan_grads_arr[istep] = ~no_nan_grads
        end = time()
        if istep % 100 == 0:
            print(istep, loss, end - start, no_nan_grads)
        if ~no_nan_grads:
            break

    argmin_best = np.argmin(loss_arr)
    best_fit_params = params_arr[argmin_best]

    def return_params_from_result(bestfit):
        bestfit_u_tuple = array_to_tuple_new_diffstarpop(bestfit, UnboundParams)
        diffstarpop_params = get_bounded_diffstarpop_params(
            bestfit_u_tuple.diffstarpop_u_params
        )
        return diffstarpop_params

    best_result = return_params_from_result(best_fit_params)
    best_result = tuple_to_array(best_result)
    np.save(outdir + "bestfit_diffstarpop_params.npy", best_result)
