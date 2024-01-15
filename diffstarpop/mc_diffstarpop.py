"""
"""
from collections import OrderedDict

from diffstar import DiffstarParams, DiffstarUParams, get_bounded_diffstar_params
from jax import jit as jjit
from jax import random as jran
from jax import vmap

from .kernels import mc_diffstar_u_params_singlegal_kernel


@jjit
def mc_diffstar_params_singlegal(diffstarpop_params, mah_params, p50, ran_key):
    diffstar_u_params = mc_diffstar_u_params_singlegal(
        diffstarpop_params, mah_params, p50, ran_key
    )
    diffstar_params = get_bounded_diffstar_params(diffstar_u_params)
    return DiffstarParams(*diffstar_params)


@jjit
def mc_diffstar_u_params_singlegal(diffstarpop_params, mah_params, p50, ran_key):
    """"""
    pdf_pdict_MS = _get_pdict_from_namedtuple(diffstarpop_params.sfh_pdf_mainseq_params)
    pdf_pdict_Q = _get_pdict_from_namedtuple(diffstarpop_params.sfh_pdf_quench_params)
    R_model_pdict_MS = _get_pdict_from_namedtuple(
        diffstarpop_params.assembias_mainseq_params
    )
    R_model_pdict_Q = _get_pdict_from_namedtuple(
        diffstarpop_params.assembias_quench_params
    )

    diffstar_u_params = mc_diffstar_u_params_singlegal_kernel(
        mah_params,
        p50,
        ran_key,
        pdf_pdict_Q=pdf_pdict_Q,
        pdf_pdict_MS=pdf_pdict_MS,
        R_model_pdict_Q=R_model_pdict_Q,
        R_model_pdict_MS=R_model_pdict_MS,
    )
    return DiffstarUParams(diffstar_u_params[:4], diffstar_u_params[4:])


_POP = (0, 0, 0, None, None, None, None)
mc_diffstar_u_params_galpop_kernel = jjit(
    vmap(mc_diffstar_u_params_singlegal_kernel, in_axes=_POP)
)


@jjit
def mc_diffstar_u_params_galpop(diffstarpop_params, mah_params, p50, ran_key):
    """"""
    pdf_pdict_MS = _get_pdict_from_namedtuple(diffstarpop_params.sfh_pdf_mainseq_params)
    pdf_pdict_Q = _get_pdict_from_namedtuple(diffstarpop_params.sfh_pdf_quench_params)
    R_model_pdict_MS = _get_pdict_from_namedtuple(
        diffstarpop_params.assembias_mainseq_params
    )
    R_model_pdict_Q = _get_pdict_from_namedtuple(
        diffstarpop_params.assembias_quench_params
    )

    ngals = p50.size
    ran_keys = jran.split(ran_key, ngals)
    diffstar_u_params = mc_diffstar_u_params_galpop_kernel(
        mah_params,
        p50,
        ran_keys,
        pdf_pdict_Q,
        pdf_pdict_MS,
        R_model_pdict_Q,
        R_model_pdict_MS,
    )
    return DiffstarUParams(diffstar_u_params[:, :4], diffstar_u_params[:, 4:])


@jjit
def _get_pdict_from_namedtuple(params):
    return OrderedDict([(key, val) for key, val in zip(params._fields, params)])
