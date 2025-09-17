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

from diffstarpop.loss_kernels.mstar_ssfr_loss_mgash import (
    loss_mstar_kern_tobs_grad_wrapper,
    loss_mstar_ssfr_kern_tobs_grad_wrapper,
    loss_combined_wrapper,
    loss_combined_3loss_wrapper,
    loss_combined_3loss_kern,
)

from diffstarpop.loss_kernels.namedtuple_utils_mgash import (
    tuple_to_array,
    tuple_to_jax_array,
    register_tuple_new_diffstarpop_tpeak,
    array_to_tuple_new_diffstarpop_tpeak,
)
from diffstarpop.kernels.defaults_mgash import (
    DEFAULT_DIFFSTARPOP_U_PARAMS,
    DEFAULT_DIFFSTARPOP_PARAMS,
    get_bounded_diffstarpop_params,
)

from fit_get_loss_helpers_mgash import (
    get_loss_data_pdfs_mstar,
    get_loss_data_pdfs_ssfr_central,
    get_loss_data_pdfs_ssfr_satellite,
)
from diffstarpop.kernels.params import (
    DiffstarPop_UParams_Diffstarfits_mgash,
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
        "-loss_type",
        help="Which data to target",
        type=str,
        choices=["mstar", "mstar_ssfr_cen", "mstar_ssfr_cen_sat"],
        default="mstar",
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
    nhalos = args.nhalos
    n_step = args.nstep
    outname = args.outname
    params_path = args.params_path
    loss_type = args.loss_type

    # Load MStar pdf data ---------------------------------------------

    if loss_type == "mstar":
        loss_data_mstar, plot_data_mstar = get_loss_data_pdfs_mstar(indir, nhalos)
        loss_data = (loss_data_mstar,)
    elif loss_type == "mstar_ssfr_cen":
        loss_data_mstar, plot_data_mstar = get_loss_data_pdfs_mstar(indir, nhalos)
        loss_data_ssfr_cen, plot_data_ssfr_cen = get_loss_data_pdfs_ssfr_central(
            indir, nhalos
        )
        loss_data = (loss_data_mstar, loss_data_ssfr_cen)
    elif loss_type == "mstar_ssfr_cen_sat":
        loss_data_mstar, plot_data_mstar = get_loss_data_pdfs_mstar(indir, nhalos)
        loss_data_ssfr_cen, plot_data_ssfr_cen = get_loss_data_pdfs_ssfr_central(
            indir, nhalos
        )
        loss_data_ssfr_sat, plot_data_ssfr_sat = get_loss_data_pdfs_ssfr_satellite(
            indir, nhalos
        )
        loss_data = (loss_data_mstar, loss_data_ssfr_cen, loss_data_ssfr_sat)

    # Define loss kernel ---------------------------------------------
    loss_combined_3loss_grad_kern = jjit(grad(loss_combined_3loss_kern, argnums=(0,)))

    def loss_kernel(
        flat_uparams, loss_data_mstar, loss_data_ssfr_cen, loss_data_ssfr_sat
    ):

        namedtuple_uparams = array_to_tuple_new_diffstarpop_tpeak(
            flat_uparams, UnboundParams
        )
        loss = loss_combined_3loss_kern(
            namedtuple_uparams, loss_data_mstar, loss_data_ssfr_cen, loss_data_ssfr_sat
        )
        return loss

    def loss_kernel_jac(
        flat_uparams, loss_data_mstar, loss_data_ssfr_cen, loss_data_ssfr_sat
    ):

        namedtuple_uparams = array_to_tuple_new_diffstarpop_tpeak(
            flat_uparams, UnboundParams
        )
        grads = loss_combined_3loss_grad_kern(
            namedtuple_uparams, loss_data_mstar, loss_data_ssfr_cen, loss_data_ssfr_sat
        )
        grads = tuple_to_jax_array(grads)

        return grads

    # Register params ---------------------------------------------

    unbound_params_dict = OrderedDict(diffstarpop_u_params=DEFAULT_DIFFSTARPOP_U_PARAMS)
    UnboundParams = namedtuple("UnboundParams", list(unbound_params_dict.keys()))
    register_tuple_new_diffstarpop_tpeak(UnboundParams)

    if params_path is None:
        all_u_params = tuple_to_array(DEFAULT_DIFFSTARPOP_U_PARAMS)
    elif params_path.startswith("diffstarfits"):
        sim_name = params_path.split("_")[1:]
        sim_name = ("_").join(sim_name)
        params_tuple = DiffstarPop_UParams_Diffstarfits_mgash[sim_name]
        all_u_params = tuple_to_array(params_tuple)
    else:
        params = np.load(params_path)
        all_u_params = params["diffstarpop_u_params"]

    # Run fitter ---------------------------------------------
    print("Running fitter...")

    params_init = tuple_to_array(all_u_params)
    loss_kernel(params_init, *loss_data)
    loss_kernel_jac(params_init, *loss_data)

    start = time()

    res = minimize(
        fun=loss_kernel,
        x0=params_init,
        jac=loss_kernel_jac,
        args=loss_data,
        method="L-BFGS-B",
    )

    finish = time()

    assert False

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
