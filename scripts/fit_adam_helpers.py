import numpy as np
from jax import random as jran
from jax.example_libraries import optimizers as jax_opt
import time


def jax_adam_wrapper(
    params_init,
    loss_data,
    loss_fun,
    loss_fun_grad,
    n_step,
    n_histories,
    ran_key,
    output_filename=None,
    jax_optimizer=None,
    step_size=0.01,
):
    loss_arr = np.zeros(n_step).astype("f4") + np.inf
    if jax_optimizer is None:
        opt_init, opt_update, get_params = jax_opt.adam(step_size)
        opt_state = opt_init(params_init)
    else:
        opt_state, opt_update, get_params = jax_optimizer

    save_params = False if output_filename is None else True

    n_params = len(params_init)
    params_arr = np.zeros((n_step, n_params)).astype("f4")

    current_indx_best = -1
    # no_nan_grads_arr = np.zeros(n_step)
    for istep in range(n_step):
        t0 = time.time()
        ran_key, subkey = jran.split(ran_key, 2)

        p = np.array(get_params(opt_state))

        loss = loss_fun(p, loss_data, n_histories, ran_key)
        grads = loss_fun_grad(p, loss_data, n_histories, ran_key)

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

            indx_best = np.nanargmin(loss_arr)

            if indx_best != current_indx_best:
                current_indx_best = indx_best
                best_fit_params = params_arr[indx_best]
                best_fit_loss = loss_arr[indx_best]
                np.savez(
                    output_filename,
                    best_fit_loss=best_fit_loss,
                    best_fit_params=best_fit_params,
                )

        # no_nan_grads_arr[istep] = ~no_nan_grads

        t1 = time.time()
        print(istep, loss, t1 - t0, no_nan_grads)
        # if ~no_nan_grads:
        #    break

    indx_best = np.nanargmin(loss_arr)
    best_fit_params = params_arr[indx_best]
    best_fit_loss = loss_arr[indx_best]

    jax_optimizer = (opt_state, opt_update, get_params)
    return best_fit_params, best_fit_loss, loss_arr, params_arr, jax_optimizer
