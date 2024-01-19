"""Model of a main sequence galaxy population calibrated to SMDPL halos."""
from collections import OrderedDict, namedtuple

from diffstar import DEFAULT_DIFFSTAR_U_PARAMS
from jax import jit as jjit
from jax import numpy as jnp

from ..utils import _sigmoid

TODAY = 13.8
LGT0 = jnp.log10(TODAY)

FIXED_UH = DEFAULT_DIFFSTAR_U_PARAMS.u_ms_params.u_indx_hi

LGM_X0, LGM_K = 13.0, 0.5

DEFAULT_SFH_PDF_MAINSEQ_PDICT = OrderedDict(
    mean_ulgm_mainseq_ylo=9.978,
    mean_ulgm_mainseq_yhi=14.630,
    mean_ulgy_mainseq_ylo=-2.075,
    mean_ulgy_mainseq_yhi=1.942,
    mean_ul_mainseq_ylo=-6.616,
    mean_ul_mainseq_yhi=9.419,
    mean_utau_mainseq_ylo=47.540,
    mean_utau_mainseq_yhi=-57.150,
    chol_ulgm_ulgm_mainseq_ylo=-0.705,
    chol_ulgm_ulgm_mainseq_yhi=0.061,
    chol_ulgy_ulgy_mainseq_ylo=-0.673,
    chol_ulgy_ulgy_mainseq_yhi=-0.495,
    chol_ul_ul_mainseq_ylo=1.185,
    chol_ul_ul_mainseq_yhi=-2.361,
    chol_utau_utau_mainseq_ylo=-0.840,
    chol_utau_utau_mainseq_yhi=1.493,
    chol_ulgy_ulgm_mainseq_ylo=-1.402,
    chol_ulgy_ulgm_mainseq_yhi=-1.760,
    chol_ul_ulgm_mainseq_ylo=0.030,
    chol_ul_ulgm_mainseq_yhi=-0.053,
    chol_ul_ulgy_mainseq_ylo=-0.326,
    chol_ul_ulgy_mainseq_yhi=-0.389,
    chol_utau_ulgm_mainseq_ylo=0.742,
    chol_utau_ulgm_mainseq_yhi=1.163,
    chol_utau_ulgy_mainseq_ylo=-1.461,
    chol_utau_ulgy_mainseq_yhi=-1.529,
    chol_utau_ul_mainseq_ylo=0.114,
    chol_utau_ul_mainseq_yhi=0.440,
)
MainseqMassOnlyParams = namedtuple("Params", list(DEFAULT_SFH_PDF_MAINSEQ_PDICT.keys()))
DEFAULT_SFH_PDF_MAINSEQ_PARAMS = MainseqMassOnlyParams(**DEFAULT_SFH_PDF_MAINSEQ_PDICT)


@jjit
def _fun(x, ymin, ymax):
    return _sigmoid(x, LGM_X0, LGM_K, ymin, ymax)


@jjit
def _fun_chol_diag(x, ymin, ymax):
    _res = 10 ** _fun(x, ymin, ymax)
    return _res


@jjit
def _get_cov_scalar(
    ulgm_ulgm,
    ulgy_ulgy,
    ul_ul,
    utau_utau,
    ulgy_ulgm,
    ul_ulgm,
    ul_ulgy,
    utau_ulgm,
    utau_ulgy,
    utau_ul,
):
    chol = jnp.zeros((4, 4)).astype("f4")
    chol = chol.at[(0, 0)].set(ulgm_ulgm)
    chol = chol.at[(1, 1)].set(ulgy_ulgy)
    chol = chol.at[(2, 2)].set(ul_ul)
    chol = chol.at[(3, 3)].set(utau_utau)

    chol = chol.at[(1, 0)].set(ulgy_ulgm * ulgy_ulgy * ulgm_ulgm)
    chol = chol.at[(2, 0)].set(ul_ulgm * ul_ul * ulgm_ulgm)
    chol = chol.at[(2, 1)].set(ul_ulgy * ul_ul * ulgy_ulgy)
    chol = chol.at[(3, 0)].set(utau_ulgm * utau_utau * ulgm_ulgm)
    chol = chol.at[(3, 1)].set(utau_ulgy * utau_utau * ulgy_ulgy)
    chol = chol.at[(3, 2)].set(utau_ul * utau_utau * ul_ul)

    cov = jnp.dot(chol, chol.T)
    return cov


@jjit
def _get_mean_u_params_mainseq(lgm, params):
    params = MainseqMassOnlyParams(*params)
    ulgm = _fun(lgm, params.mean_ulgm_mainseq_ylo, params.mean_ulgm_mainseq_yhi)
    ulgy = _fun(lgm, params.mean_ulgy_mainseq_ylo, params.mean_ulgy_mainseq_yhi)
    ul = _fun(lgm, params.mean_ul_mainseq_ylo, params.mean_ul_mainseq_yhi)
    utau = _fun(lgm, params.mean_utau_mainseq_ylo, params.mean_utau_mainseq_yhi)
    return ulgm, ulgy, ul, utau


@jjit
def _get_chol_params_mainseq(lgm, params):
    ulgm_ulgm = _fun_chol_diag(
        lgm, params.chol_ulgm_ulgm_mainseq_ylo, params.chol_ulgm_ulgm_mainseq_yhi
    )
    ulgy_ulgy = _fun_chol_diag(
        lgm, params.chol_ulgy_ulgy_mainseq_ylo, params.chol_ulgy_ulgy_mainseq_yhi
    )
    ul_ul = _fun_chol_diag(
        lgm, params.chol_ul_ul_mainseq_ylo, params.chol_ul_ul_mainseq_yhi
    )
    utau_utau = _fun_chol_diag(
        lgm, params.chol_utau_utau_mainseq_ylo, params.chol_utau_utau_mainseq_yhi
    )
    ulgy_ulgm = _fun(
        lgm, params.chol_ulgy_ulgm_mainseq_ylo, params.chol_ulgy_ulgm_mainseq_yhi
    )
    ul_ulgm = _fun(
        lgm, params.chol_ul_ulgm_mainseq_ylo, params.chol_ul_ulgm_mainseq_yhi
    )
    ul_ulgy = _fun(
        lgm, params.chol_ul_ulgy_mainseq_ylo, params.chol_ul_ulgy_mainseq_yhi
    )
    utau_ulgm = _fun(
        lgm, params.chol_utau_ulgm_mainseq_ylo, params.chol_utau_ulgm_mainseq_yhi
    )
    utau_ulgy = _fun(
        lgm, params.chol_utau_ulgy_mainseq_ylo, params.chol_utau_ulgy_mainseq_yhi
    )
    utau_ul = _fun(
        lgm, params.chol_utau_ul_mainseq_ylo, params.chol_utau_ul_mainseq_yhi
    )

    chol_params = (
        ulgm_ulgm,
        ulgy_ulgy,
        ul_ul,
        utau_utau,
        ulgy_ulgm,
        ul_ulgm,
        ul_ulgy,
        utau_ulgm,
        utau_ulgy,
        utau_ul,
    )

    return chol_params


@jjit
def _get_cov_mainseq(lgm, params):
    params = MainseqMassOnlyParams(*params)
    chol_params = _get_chol_params_mainseq(lgm, params)
    cov_ms = _get_cov_scalar(*chol_params)
    return cov_ms
