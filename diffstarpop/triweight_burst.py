"""
"""
from jax import numpy as jnp
from jax import jit as jjit
from jax import grad
from jax import vmap


BURST_KERN_LGTLO, BURST_KERN_DLGT = 4, 1
C0 = 1 / 2
C1 = 35 / 96
C3 = -35 / 864
C5 = 7 / 2592
C7 = -5 / 69984
BURST_TW_X0 = BURST_KERN_LGTLO + BURST_KERN_DLGT / 2
BURST_TW_H = BURST_KERN_DLGT / 6


@jjit
def _tw_cuml_kern(x, m, h):
    """Triweight kernel version of an err function."""
    z = (x - m) / h
    zz = z * z
    zzz = zz * z
    val = C0 + C1 * z + C3 * zzz + C5 * zzz * zz + C7 * zzz * zzz * z
    val = jnp.where(z < -3, 0, val)
    val = jnp.where(z > 3, 1, val)
    return val


@jjit
def _tw_sigmoid(x, x0, tw_h, ymin, ymax):
    height_diff = ymax - ymin
    body = _tw_cuml_kern(x, x0, tw_h)
    return ymin + height_diff * body


@jjit
def _cuml_prob_lgage(lgyr_since_burst):
    return _tw_sigmoid(lgyr_since_burst, BURST_TW_X0, BURST_TW_H, 0, 1)


@jjit
def _cuml_prob_age(yr_since_burst):
    return _cuml_prob_lgage(jnp.log10(yr_since_burst))


_tw_burst_kern = jjit(vmap(grad(_cuml_prob_age), in_axes=0))


@jjit
def _sfh_post_burst(yr_since_burst, total_mstar_formed):
    return _tw_burst_kern(yr_since_burst) * total_mstar_formed


@jjit
def _mstar_post_burst(yr_since_burst, total_mstar_formed):
    lgt = jnp.log10(yr_since_burst)
    norm = total_mstar_formed
    res = _cuml_prob_lgage(lgt) * norm
    return res
