"""This module implements kernels for Monte Carlo generating Diffstar SFHs
"""

from diffstar import (
    DiffstarParams,
    calc_sfh_galpop,
    calc_sfh_singlegal,
    get_bounded_diffstar_params,
)
from diffstar.defaults import FB, LGT0
from jax import jit as jjit
from jax import random as jran
from jax import vmap

from .kernel_wrapper_cens import mc_diffstar_u_params_singlecen_kernel


@jjit
def mc_diffstar_sfh_singlecen(
    diffstarpop_params, mah_params, p50, ran_key, tarr, lgt0=LGT0, fb=FB
):
    diffstar_params_q, diffstar_params_ms, frac_q = mc_diffstar_params_singlecen(
        diffstarpop_params,
        mah_params,
        p50,
        ran_key,
    )
    sfh_q = calc_sfh_singlegal(diffstar_params_q, mah_params, tarr, lgt0=lgt0, fb=fb)
    sfh_ms = calc_sfh_singlegal(diffstar_params_ms, mah_params, tarr, lgt0=lgt0, fb=fb)
    return diffstar_params_q, diffstar_params_ms, sfh_q, sfh_ms, frac_q


@jjit
def mc_diffstar_params_singlecen(diffstarpop_params, mah_params, p50, ran_key):
    _res = mc_diffstar_u_params_singlecen_kernel(
        diffstarpop_params.sfh_pdf_mainseq_params,
        diffstarpop_params.sfh_pdf_quench_params,
        diffstarpop_params.assembias_mainseq_params,
        diffstarpop_params.assembias_quench_params,
        mah_params,
        p50,
        ran_key,
    )
    diffstar_u_params_q, diffstar_u_params_ms, frac_q = _res
    diffstar_params_q = get_bounded_diffstar_params(diffstar_u_params_q)
    diffstar_params_ms = get_bounded_diffstar_params(diffstar_u_params_ms)
    diffstar_params_q = DiffstarParams(*diffstar_params_q)
    diffstar_params_ms = DiffstarParams(*diffstar_params_ms)
    return diffstar_params_q, diffstar_params_ms, frac_q


@jjit
def mc_diffstar_u_params_singlecen(diffstarpop_params, mah_params, p50, ran_key):
    """"""
    args = (
        diffstarpop_params.sfh_pdf_mainseq_params,
        diffstarpop_params.sfh_pdf_quench_params,
        diffstarpop_params.assembias_mainseq_params,
        diffstarpop_params.assembias_quench_params,
        mah_params,
        p50,
        ran_key,
    )
    _res = mc_diffstar_u_params_singlecen_kernel(*args)
    diffstar_u_params_q, diffstar_u_params_ms, frac_q = _res
    return diffstar_u_params_q, diffstar_u_params_ms, frac_q


_POP = (None, 0, 0, 0)
mc_diffstar_u_params_cenpop_kernel = jjit(
    vmap(mc_diffstar_u_params_singlecen, in_axes=_POP)
)


@jjit
def mc_diffstar_u_params_cenpop(diffstarpop_params, mah_params, p50, ran_key):
    """"""
    ngals = p50.size
    ran_keys = jran.split(ran_key, ngals)
    _res = mc_diffstar_u_params_cenpop_kernel(
        diffstarpop_params,
        mah_params,
        p50,
        ran_keys,
    )
    diffstar_u_params_q, diffstar_u_params_ms, frac_q = _res
    return diffstar_u_params_q, diffstar_u_params_ms, frac_q


get_bounded_diffstar_params_galpop = jjit(vmap(get_bounded_diffstar_params, in_axes=0))


@jjit
def mc_diffstar_params_cenpop(diffstarpop_params, mah_params, p50, ran_key):
    """"""
    _res = mc_diffstar_u_params_cenpop(diffstarpop_params, mah_params, p50, ran_key)
    diffstar_u_params_q, diffstar_u_params_ms, frac_q = _res
    diffstar_params_q = get_bounded_diffstar_params_galpop(diffstar_u_params_q)
    diffstar_params_ms = get_bounded_diffstar_params_galpop(diffstar_u_params_ms)
    return diffstar_params_q, diffstar_params_ms, frac_q


@jjit
def mc_diffstar_sfh_cenpop(
    diffstarpop_params, mah_params, p50, ran_key, tarr, lgt0=LGT0, fb=FB
):
    diffstar_params_q, diffstar_params_ms, frac_q = mc_diffstar_params_cenpop(
        diffstarpop_params, mah_params, p50, ran_key
    )
    sfh_q = calc_sfh_galpop(diffstar_params_q, mah_params, tarr, lgt0=lgt0, fb=fb)
    sfh_ms = calc_sfh_galpop(diffstar_params_ms, mah_params, tarr, lgt0=lgt0, fb=fb)
    return diffstar_params_q, diffstar_params_ms, sfh_q, sfh_ms, frac_q
