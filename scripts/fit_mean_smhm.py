
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
from collections import OrderedDict, namedtuple

from diffstarpop.loss_kernels.smhm_loss_tpeak_block_cov import (
    mean_smhm_loss_kern_tobs_wrapper,
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


    with h5py.File(indir+"smdpl_smhm.h5", "r") as hdf:
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

    logmh_binsc = 0.5*(logmh_bins[1:]+logmh_bins[:-1])


    with h5py.File(indir+"smdpl_smhm_samples_haloes.h5", "r") as hdf:
        logmh_id = hdf["logmh_id"][:]
        logmh_val = hdf["logmh_id"][:]
        mah_params_samp = hdf["mah_params_samp"][:]
        ms_params_samp = hdf["ms_params_samp"][:]
        q_params_samp = hdf["q_params_samp"][:]
        t_peak_samp = hdf["t_peak_samp"][:]
        tobs_id = hdf["tobs_id"][:]
        tobs_val = hdf["tobs_val"][:]
        redshift_val = hdf["redshift_val"][:]


    # Create loss_data ---------------------------------------------
    print("Creating loss data...")

    ran_key = jran.PRNGKey(np.random.randint(2**32))

    lgmu_infall = -1.0
    logmhost_infall = 13.0
    gyr_since_infall = 2.0

    mah_params_data = []
    t_peak_data = []
    lgmu_infall_data = []
    logmhost_infall_data = []
    gyr_since_infall_data = []
    t_obs_targets = []
    smhm_targets = []

    for i in range(len(age_targets)):
        t_target = age_targets[i]

        for j in range(len(logmh_binsc)):
            sel = (tobs_id == i)  & (logmh_id == j)

            if sel.sum() < nhalos: continue
            arange_sel = np.arange(len(tobs_id))[sel]
            arange_sel = np.random.choice(arange_sel, nhalos, replace=False)
            mah_params_data.append(mah_params_samp[:, arange_sel])
            t_peak_data.append(t_peak_samp[arange_sel])
            lgmu_infall_data.append(np.ones(len(arange_sel))*lgmu_infall)
            logmhost_infall_data.append(np.ones(len(arange_sel))*logmhost_infall)
            gyr_since_infall_data.append(np.ones(len(arange_sel))*gyr_since_infall)
            t_obs_targets.append(t_target)
            smhm_targets.append(smhm[i,j])

    mah_params_data = np.array(mah_params_data)
    t_peak_data = np.array(t_peak_data)
    lgmu_infall_data = np.array(lgmu_infall_data)
    logmhost_infall_data = np.array(logmhost_infall_data)
    gyr_since_infall_data = np.array(gyr_since_infall_data)
    t_obs_targets = np.array(t_obs_targets)
    smhm_targets = np.array(smhm_targets)

    ran_key_data = jran.split(ran_key, len(smhm_targets))
    loss_data = (
        mah_params_data,
        t_peak_data,
        lgmu_infall_data,
        logmhost_infall_data,
        gyr_since_infall_data,
        ran_key_data,
        t_obs_targets,
        smhm_targets
    )

    # Register params ---------------------------------------------


    unbound_params_dict = OrderedDict(
        diffstarpop_u_params=DEFAULT_DIFFSTARPOP_U_PARAMS
    )
    UnboundParams = namedtuple(
        "UnboundParams", list(unbound_params_dict.keys()))
    register_tuple_new_diffstarpop(UnboundParams)
    all_u_params = UnboundParams(*list(unbound_params_dict.values()))

    # Run fitter ---------------------------------------------
    print("Running fitter...")
    start = time()

    flat_all_u_params = tuple_to_array(all_u_params)
    mean_smhm_loss_kern_tobs_wrapper(flat_all_u_params, loss_data)

    result = minimize(
        mean_smhm_loss_kern_tobs_wrapper, 
        method="L-BFGS-B", 
        x0=flat_all_u_params, 
        jac=True, 
        options=dict(maxiter=200), 
        args=(loss_data,)
    )
    end = time()
    runtime = end - start
    print(f"Total runtime to fit = {runtime:.1f} seconds")


    def return_params_from_result(bestfit):
            bestfit_u_tuple = array_to_tuple_new_diffstarpop(bestfit, UnboundParams)
            diffstarpop_params = get_bounded_diffstarpop_params(bestfit_u_tuple.diffstarpop_u_params)
            return diffstarpop_params
    
    best_result = return_params_from_result(result.x)
    best_result = tuple_to_array(best_result)
    np.save(outdir+"bestfit_diffstarpop_params.npy", best_result)

    # Make plot ---------------------------------------------
    if make_plot:
        print("Making plot...")

        from diffstar.sfh_model_tpeak import _cumulative_mstar_formed_vmap
        from diffstarpop.mc_diffstarpop_tpeak import mc_diffstar_sfh_galpop
        from diffmah import DiffmahParams
        import matplotlib.pyplot as plt
        import matplotlib as mpl

        mstar_plot = np.zeros((len(age_targets), len(logmh_binsc)))
        mstar_plot_grad = np.zeros((len(age_targets), len(logmh_binsc)))

        for i in range(len(age_targets)):
            t_target = age_targets[i]
            print("Age target:", t_target)
            tarr = np.logspace(-1, np.log10(t_target), 50)

            for j in range(len(logmh_binsc)):
                
                sel = (tobs_id == i)  & (logmh_id == j)
                res = mc_diffstar_sfh_galpop(
                    return_params_from_result(result.x),
                    DiffmahParams(*mah_params_samp[:, sel]),
                    t_peak_samp[sel],
                    np.ones(sel.sum())*lgmu_infall,
                    np.ones(sel.sum())*logmhost_infall,
                    np.ones(sel.sum())*gyr_since_infall,
                    ran_key,
                    tarr,
                )
        
                diffstar_params_ms, diffstar_params_q, sfh_ms, sfh_q, frac_q, mc_is_q = res
                mstar_q = _cumulative_mstar_formed_vmap(tarr, sfh_q)
                mstar_ms = _cumulative_mstar_formed_vmap(tarr, sfh_ms)
                mean_mstar_grad_vals = mstar_q[:,-1] * frac_q + mstar_ms[:,-1] * (1 - frac_q)
                mean_mstar_grad = jnp.mean(jnp.log10(mean_mstar_grad_vals))
                
                mean_mstar_plot_vals = mstar_q[:,-1] * mc_is_q.astype(int).astype(float) + mstar_ms[:,-1] * (1.0 - mc_is_q.astype(int))
                mean_mstar_plot_vals = jnp.mean(jnp.log10(mean_mstar_plot_vals))
                mstar_plot[i,j] = mean_mstar_plot_vals
                mstar_plot_grad[i,j] = mean_mstar_grad
                # break

                cmap = plt.get_cmap("plasma")(redshift_targets/redshift_targets.max())

        fig, ax = plt.subplots(1, 1, figsize=(6,4))
        for i in range(len(smhm)):
            ax.plot(10**logmh_binsc, 10**smhm[i], color=cmap[i])
            ax.plot(10**logmh_binsc, 10**mstar_plot[i], color=cmap[i], ls='--')

        norm = mpl.colors.Normalize(vmin=0, vmax=2)

        fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=mpl.cm.plasma),
                    ax = ax, label='Redshift')

        ax.set_yscale('log')
        ax.set_xscale('log')
        ax.set_xlim(9e10, 1e15)

        ax.set_ylabel(r"$\langle M_\star(t_{\rm obs})| M_{\rm halo}(t_{\rm obs}) \rangle$ $[M_\odot]$")
        ax.set_xlabel(r"$M_{\rm halo}(t_{\rm obs})$ $[M_\odot]$")
        plt.savefig(outdir+"smhm_logsm.png", bbox_inches='tight', dpi=300)
        plt.clf()