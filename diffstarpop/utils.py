"""
"""
import numpy as np
from diffstar.utils import jax_np_interp
from halotools.utils import sliding_conditional_percentile
from jax import jit as jjit
from jax import lax
from jax import numpy as jnp
from jax import vmap
from scipy.optimize import minimize


@jjit
def _sigmoid(x, x0, k, ymin, ymax):
    """Sigmoid function implemented w/ `jax.numpy.exp`

    Parameters
    ----------
    x : float or array-like
        Points at which to evaluate the function.
    x0 : float or array-like
        Location of transition.
    k : float or array-like
        Inverse of the width of the transition.
    ymin : float or array-like
        The value as x goes to -infty.
    ymax : float or array-like
        The value as x goes to +infty.

    Returns
    -------
    sigmoid : scalar or array-like, same shape as input

    """
    height_diff = ymax - ymin
    return ymin + height_diff * lax.logistic(k * (x - x0))


@jjit
def _inverse_sigmoid(y, x0=0, k=1, ymin=-1, ymax=1):
    lnarg = (ymax - ymin) / (y - ymin) - 1
    return x0 - jnp.log(lnarg) / k


def minimizer(loss_func, loss_func_deriv, p_init, loss_data):
    """Minimizer mixing scipy's L-BFGS-B minimizer with JAX's Adam as a backup plan.

    Parameters
    ----------
    loss_func : callable
        Differentiable function to minimize.
        Must accept inputs (params, data) and return a scalar,
        and be differentiable using jax.grad.
    loss_func_deriv : callable
        Returns the gradient wrt the parameters of loss_func.
        Must accept inputs (params, data) and return a scalar.
    p_init : ndarray of shape (n_params, )
        Initial guess at the parameters. The fitter uses this guess to draw
        random initial guesses with small perturbations around these values.
    loss_data : sequence
        Sequence of floats and arrays storing whatever data is needed
        to compute loss_func(p_init, loss_data)

    Returns
    -------
    p_best : ndarray of shape (n_params, )
        Stores the best-fit value of the parameters after n_step steps
    loss_best : float
        Final value of the loss
    success : int
        -1 if NaN or inf is encountered by the fitter, causing termination before n_step
        0 for a fit that fails with L-BFGS-B but terminates without problems using Adam
        1 for a fit that terminates with no such problems using L-BFGS-B

    """

    res = minimize(
        loss_func, x0=p_init, method="L-BFGS-B", jac=loss_func_deriv, args=(loss_data,)
    )
    p_best = res.x
    loss_best = float(res.fun)
    success = 1

    return p_best, loss_best, success


@jjit
def _tw_cuml_lax_kern(x, m, h):
    """Triweight kernel version of an err function.
    This kernel accepts and returns scalars for all arguments
    """
    z = (x - m) / h
    val = (
        -5 * z**7 / 69984
        + 7 * z**5 / 2592
        - 35 * z**3 / 864
        + 35 * z / 96
        + 1 / 2
    )
    val = lax.cond(z < -3, lambda s: 0.0, lambda s: val, z)
    val = lax.cond(z > 3, lambda s: 1.0, lambda s: val, z)
    return val


@jjit
def _tw_bin_weight_lax_kern(x, sig, lo, hi):
    """Triweight kernel integrated across the boundaries of a single bin.
    This kernel accepts and returns scalars for all arguments
    """
    a = _tw_cuml_lax_kern(x, lo, sig)
    b = _tw_cuml_lax_kern(x, hi, sig)
    return a - b


def return_searchsorted_like_results(mstar, mstar_frac):
    _tmp = mstar - mstar[:, [-1]] * mstar_frac
    _tmp = np.array(_tmp)
    _tmp[_tmp < 0] = np.inf
    _res = np.argmin(_tmp, axis=1)
    _res = np.clip(_res, 1, None).astype(int)
    return _res


jax_np_interp_vmap = jjit(vmap(jax_np_interp, in_axes=(0, 0, None, 0)))


def get_t50_p50(t_table, histories, threshold, logmpeak, window_length=101):
    hist_val_at_threshold = histories[:, -1] * threshold
    indices = return_searchsorted_like_results(histories, threshold)
    t50 = jax_np_interp_vmap(hist_val_at_threshold, histories, t_table, indices)
    p50 = sliding_conditional_percentile(logmpeak, t50, window_length=window_length)
    return t50, p50
