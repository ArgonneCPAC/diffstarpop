import os
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
    loss_mstar_kern_tobs_grad_wrapper,
    get_pred_mstar_data_wrapper,
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

from fit_get_loss_helpers import get_loss_data_pdfs_mstar


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

    parser.add_argument(
        "-nstep", help="Number of steps for fitting", type=int, default=1000
    )
    parser.add_argument(
        "-outname",
        help="output fname for best params",
        type=str,
        default="bestfit_diffstarpop_params",
    )
    parser.add_argument(
        "--params_path",
        type=str,
        default=None,
        help="Path were diffstarpop params are stored",
    )
    parser.add_argument(
        "--print_loss",
        type=int,
        default=100,
        help="How many steps before printing current loss",
    )

    args = parser.parse_args()
    indir = args.indir
    outdir = args.outdir
    make_plot = args.make_plot
    nhalos = args.nhalos
    n_step = args.nstep
    outname = args.outname
    params_path = args.params_path

    # Load MStar pdf data ---------------------------------------------

    loss_data, plot_data = get_loss_data_pdfs_mstar(indir, nhalos)

    # Register params ---------------------------------------------

    unbound_params_dict = OrderedDict(diffstarpop_u_params=DEFAULT_DIFFSTARPOP_U_PARAMS)
    UnboundParams = namedtuple("UnboundParams", list(unbound_params_dict.keys()))
    register_tuple_new_diffstarpop_tpeak(UnboundParams)

    if params_path is None:
        all_u_params = tuple_to_array(DEFAULT_DIFFSTARPOP_U_PARAMS)
    else:
        params = np.load(params_path)
        all_u_params = params["diffstarpop_u_params"]

    # Run fitter ---------------------------------------------
    print("Running fitter...")

    params_init = tuple_to_array(all_u_params)
    loss_mstar_kern_tobs_grad_wrapper(params_init, loss_data)

    start = time()

    step_size = 0.01

    loss_arr = np.zeros(n_step).astype("f4") + np.inf

    opt_init, opt_update, get_params = jax_opt.adam(step_size)
    opt_state = opt_init(params_init)

    n_params = len(params_init)
    params_arr = np.zeros((n_step, n_params)).astype("f4")

    n_mah = 100

    ran_key = jran.PRNGKey(np.random.randint(2**32))

    no_nan_grads_arr = np.zeros(n_step)
    for istep in range(n_step):
        start = time()
        ran_key, subkey = jran.split(ran_key, 2)

        p = np.array(get_params(opt_state))

        loss, grads = loss_mstar_kern_tobs_grad_wrapper(p, loss_data)

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
        if istep % args.print_loss == 0:
            print(istep, loss, end - start, no_nan_grads)
        if ~no_nan_grads:
            break

    argmin_best = np.argmin(loss_arr)
    best_fit_u_params = params_arr[argmin_best]

    def return_params_from_result(best_fit_u_params):
        bestfit_u_tuple = array_to_tuple_new_diffstarpop_tpeak(
            best_fit_u_params, UnboundParams
        )
        diffstarpop_params = get_bounded_diffstarpop_params(
            bestfit_u_tuple.diffstarpop_u_params
        )
        return diffstarpop_params

    best_fit_params = return_params_from_result(best_fit_u_params)
    best_fit_params = tuple_to_array(best_fit_params)

    np.savez(
        os.path.join(outdir, outname) + ".npz",
        diffstarpop_params=best_fit_params,
        diffstarpop_u_params=best_fit_u_params,
    )

    # Make plot ---------------------------------------------
    if make_plot:
        print("Making plot...")

        from matplotlib import pyplot as plt
        from matplotlib.patches import Patch
        from matplotlib.lines import Line2D

        (
            logmstar_bins_pdf,
            mstar_wcounts,
            age_targets,
            redshift_targets,
            tobs_id,
            logmh_id,
            logmh_binsc,
            loss_data_mstar_pred,
        ) = plot_data

        _mstar_counts_pred = get_pred_mstar_data_wrapper(
            best_fit_u_params, loss_data_mstar_pred
        )

        mstar_counts_pred = np.zeros_like(mstar_wcounts) * np.nan
        ij = 0
        for i in range(len(age_targets)):
            t_target = age_targets[i]

            for j in range(len(logmh_binsc)):
                sel = (tobs_id == i) & (logmh_id == j)

                # if sel.sum() == 0: continue
                if sel.sum() < 50:
                    continue
                mstar_counts_pred[i, j] = _mstar_counts_pred[ij]
                ij += 1

        fig, ax = plt.subplots(5, 1, figsize=(12, 16), sharex=False)

        colors_mstar = plt.get_cmap("viridis")(np.linspace(0, 1, 11))

        for i in range(5):

            for j in range(11):
                ax[i].fill_between(
                    logmstar_bins_pdf[1:],
                    0.0,
                    mstar_wcounts[i, j] / mstar_wcounts[i, j].sum(),
                    color=colors_mstar[j],
                    alpha=0.2,
                )
                ax[i].plot(
                    logmstar_bins_pdf[1:],
                    mstar_counts_pred[i, j],
                    color=colors_mstar[j],
                    ls="--",
                )

            ax[i].set_ylim(0, 0.3)
            ax[i].set_xlim(7, 12.0)
            ax[i].set_ylabel(r"$P(M_\star(t_{\rm obs})| M_{\rm halo}(t_{\rm obs}))$")
            ax[i].set_title(
                r"${\rm Redshift}=%.1f$" % redshift_targets[i], y=0.85, x=0.9
            )
            if i < 4:
                ax[i].set_xticklabels([])

        legend_elements = [
            Patch(
                facecolor=colors_mstar[0],
                edgecolor="none",
                label=r"$M_{\rm halo}(t_{\rm obs}))=11$",
                alpha=0.7,
            ),
            Patch(
                facecolor=colors_mstar[-1],
                edgecolor="none",
                label=r"$M_{\rm halo}(t_{\rm obs}))=14.5$",
                alpha=0.7,
            ),
            Line2D([0], [0], color="k", ls="--", label="Diffstarpop"),
        ]
        ax[0].legend(handles=legend_elements, loc=2, fontsize=14)
        ax[4].set_xlabel(r"$\log M_\star(t_{\rm obs})$")

        fig.subplots_adjust(hspace=0.08)

        plt.savefig(
            outdir + "pdf_mstar.png",
            bbox_inches="tight",
            dpi=250,
        )
        plt.clf()
