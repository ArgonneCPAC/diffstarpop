
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

from diffstar.defaults import TODAY, LGT0
from diffmah.diffmah_kernels import DiffmahParams, mah_halopop

from diffstarpop.loss_kernels.smhm_loss_tpeak import (
    mean_smhm_loss_kern_tobs_wrapper,
    get_loss_data,
    UnboundParams,
)

from diffstarpop.loss_kernels.namedtuple_utils_tpeak import (
    tuple_to_array,
    register_tuple_new_diffstarpop_tpeak,
    array_to_tuple_new_diffstarpop_tpeak,
)
from diffstarpop.kernels.defaults_tpeak import (
    DEFAULT_DIFFSTARPOP_U_PARAMS,
    DEFAULT_DIFFSTARPOP_PARAMS,
    get_bounded_diffstarpop_params,
)

BEBOP_SMHM_MEAN_DATA = "/lcrc/project/halotools/alarcon/results/ridge_term_smhm_data/"

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

    loss_data, plot_data = get_loss_data(indir, nhalos)


    # Register params ---------------------------------------------

    unbound_params_dict = OrderedDict(
        diffstarpop_u_params=DEFAULT_DIFFSTARPOP_U_PARAMS
    )
    UnboundParams = namedtuple(
        "UnboundParams", list(unbound_params_dict.keys()))
    register_tuple_new_diffstarpop_tpeak(UnboundParams)
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
            bestfit_u_tuple = array_to_tuple_new_diffstarpop_tpeak(bestfit, UnboundParams)
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
        import matplotlib.pyplot as plt
        import matplotlib as mpl

        (
            age_targets,
            logmh_binsc,
            tobs_id,
            logmh_id,
            tarr_logm0,
            lgmu_infall,
            logmhost_infall,
            gyr_since_infall,
            ran_key,
            redshift_targets,
            smhm,
            mah_params_samp,
        ) = plot_data

        mstar_plot = np.zeros((len(age_targets), len(logmh_binsc)))
        mstar_plot_grad = np.zeros((len(age_targets), len(logmh_binsc)))

        for i in range(len(age_targets)):
            t_target = age_targets[i]
            print("Age target:", t_target)
            tarr = np.logspace(-1, np.log10(t_target), 50)

            for j in range(len(logmh_binsc)):
                
                sel = (tobs_id == i)  & (logmh_id == j)
                mah_pars_ntuple = DiffmahParams(*mah_params_samp[:, sel])
                dmhdt_fit, log_mah_fit = mah_halopop(mah_pars_ntuple, tarr_logm0, LGT0)
                lomg0_vals = log_mah_fit[:, -1]
                res = mc_diffstar_sfh_galpop(
                    return_params_from_result(result.x),
                    mah_pars_ntuple,
                    lomg0_vals,
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