"""Model of a main sequence galaxy population calibrated to SMDPL halos."""
from collections import OrderedDict

from jax import jit as jjit
from jax import numpy as jnp
from jax import vmap

from ..utils import _sigmoid

TODAY = 13.8
LGT0 = jnp.log10(TODAY)

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


_get_cov_vmap = jjit(vmap(_get_cov_scalar, in_axes=(*[0] * 10,)))


@jjit
def mean_ulgm_mainseq_vs_lgm0(
    lgm0,
    mean_ulgm_mainseq_ylo=DEFAULT_SFH_PDF_MAINSEQ_PDICT["mean_ulgm_mainseq_ylo"],
    mean_ulgm_mainseq_yhi=DEFAULT_SFH_PDF_MAINSEQ_PDICT["mean_ulgm_mainseq_yhi"],
):
    return _fun(lgm0, mean_ulgm_mainseq_ylo, mean_ulgm_mainseq_yhi)


@jjit
def mean_ulgy_mainseq_vs_lgm0(
    lgm0,
    mean_ulgy_mainseq_ylo=DEFAULT_SFH_PDF_MAINSEQ_PDICT["mean_ulgy_mainseq_ylo"],
    mean_ulgy_mainseq_yhi=DEFAULT_SFH_PDF_MAINSEQ_PDICT["mean_ulgy_mainseq_yhi"],
):
    return _fun(lgm0, mean_ulgy_mainseq_ylo, mean_ulgy_mainseq_yhi)


@jjit
def mean_ul_mainseq_vs_lgm0(
    lgm0,
    mean_ul_mainseq_ylo=DEFAULT_SFH_PDF_MAINSEQ_PDICT["mean_ul_mainseq_ylo"],
    mean_ul_mainseq_yhi=DEFAULT_SFH_PDF_MAINSEQ_PDICT["mean_ul_mainseq_yhi"],
):
    return _fun(lgm0, mean_ul_mainseq_ylo, mean_ul_mainseq_yhi)


@jjit
def mean_utau_mainseq_vs_lgm0(
    lgm0,
    mean_utau_mainseq_ylo=DEFAULT_SFH_PDF_MAINSEQ_PDICT["mean_utau_mainseq_ylo"],
    mean_utau_mainseq_yhi=DEFAULT_SFH_PDF_MAINSEQ_PDICT["mean_utau_mainseq_yhi"],
):
    return _fun(lgm0, mean_utau_mainseq_ylo, mean_utau_mainseq_yhi)


@jjit
def chol_ulgm_ulgm_mainseq_vs_lgm0(
    lgm0,
    chol_ulgm_ulgm_mainseq_ylo=DEFAULT_SFH_PDF_MAINSEQ_PDICT[
        "chol_ulgm_ulgm_mainseq_ylo"
    ],
    chol_ulgm_ulgm_mainseq_yhi=DEFAULT_SFH_PDF_MAINSEQ_PDICT[
        "chol_ulgm_ulgm_mainseq_yhi"
    ],
):
    _res = _fun_chol_diag(lgm0, chol_ulgm_ulgm_mainseq_ylo, chol_ulgm_ulgm_mainseq_yhi)
    return _res


@jjit
def chol_ulgy_ulgy_mainseq_vs_lgm0(
    lgm0,
    chol_ulgy_ulgy_mainseq_ylo=DEFAULT_SFH_PDF_MAINSEQ_PDICT[
        "chol_ulgy_ulgy_mainseq_ylo"
    ],
    chol_ulgy_ulgy_mainseq_yhi=DEFAULT_SFH_PDF_MAINSEQ_PDICT[
        "chol_ulgy_ulgy_mainseq_yhi"
    ],
):
    _res = _fun_chol_diag(lgm0, chol_ulgy_ulgy_mainseq_ylo, chol_ulgy_ulgy_mainseq_yhi)
    return _res


@jjit
def chol_ul_ul_mainseq_vs_lgm0(
    lgm0,
    chol_ul_ul_mainseq_ylo=DEFAULT_SFH_PDF_MAINSEQ_PDICT["chol_ul_ul_mainseq_ylo"],
    chol_ul_ul_mainseq_yhi=DEFAULT_SFH_PDF_MAINSEQ_PDICT["chol_ul_ul_mainseq_yhi"],
):
    _res = _fun_chol_diag(lgm0, chol_ul_ul_mainseq_ylo, chol_ul_ul_mainseq_yhi)
    return _res


@jjit
def chol_utau_utau_mainseq_vs_lgm0(
    lgm0,
    chol_utau_utau_mainseq_ylo=DEFAULT_SFH_PDF_MAINSEQ_PDICT[
        "chol_utau_utau_mainseq_ylo"
    ],
    chol_utau_utau_mainseq_yhi=DEFAULT_SFH_PDF_MAINSEQ_PDICT[
        "chol_utau_utau_mainseq_yhi"
    ],
):
    _res = _fun_chol_diag(lgm0, chol_utau_utau_mainseq_ylo, chol_utau_utau_mainseq_yhi)
    return _res


@jjit
def chol_ulgy_ulgm_mainseq_vs_lgm0(
    lgm0,
    chol_ulgy_ulgm_mainseq_ylo=DEFAULT_SFH_PDF_MAINSEQ_PDICT[
        "chol_ulgy_ulgm_mainseq_ylo"
    ],
    chol_ulgy_ulgm_mainseq_yhi=DEFAULT_SFH_PDF_MAINSEQ_PDICT[
        "chol_ulgy_ulgm_mainseq_yhi"
    ],
):
    _res = _fun(lgm0, chol_ulgy_ulgm_mainseq_ylo, chol_ulgy_ulgm_mainseq_yhi)
    return _res


@jjit
def chol_ul_ulgm_mainseq_vs_lgm0(
    lgm0,
    chol_ul_ulgm_mainseq_ylo=DEFAULT_SFH_PDF_MAINSEQ_PDICT["chol_ul_ulgm_mainseq_ylo"],
    chol_ul_ulgm_mainseq_yhi=DEFAULT_SFH_PDF_MAINSEQ_PDICT["chol_ul_ulgm_mainseq_yhi"],
):
    _res = _fun(lgm0, chol_ul_ulgm_mainseq_ylo, chol_ul_ulgm_mainseq_yhi)
    return _res


@jjit
def chol_ul_ulgy_mainseq_vs_lgm0(
    lgm0,
    chol_ul_ulgy_mainseq_ylo=DEFAULT_SFH_PDF_MAINSEQ_PDICT["chol_ul_ulgy_mainseq_ylo"],
    chol_ul_ulgy_mainseq_yhi=DEFAULT_SFH_PDF_MAINSEQ_PDICT["chol_ul_ulgy_mainseq_yhi"],
):
    _res = _fun(lgm0, chol_ul_ulgy_mainseq_ylo, chol_ul_ulgy_mainseq_yhi)
    return _res


@jjit
def chol_utau_ulgm_mainseq_vs_lgm0(
    lgm0,
    chol_utau_ulgm_mainseq_ylo=DEFAULT_SFH_PDF_MAINSEQ_PDICT[
        "chol_utau_ulgm_mainseq_ylo"
    ],
    chol_utau_ulgm_mainseq_yhi=DEFAULT_SFH_PDF_MAINSEQ_PDICT[
        "chol_utau_ulgm_mainseq_yhi"
    ],
):
    _res = _fun(lgm0, chol_utau_ulgm_mainseq_ylo, chol_utau_ulgm_mainseq_yhi)
    return _res


@jjit
def chol_utau_ulgy_mainseq_vs_lgm0(
    lgm0,
    chol_utau_ulgy_mainseq_ylo=DEFAULT_SFH_PDF_MAINSEQ_PDICT[
        "chol_utau_ulgy_mainseq_ylo"
    ],
    chol_utau_ulgy_mainseq_yhi=DEFAULT_SFH_PDF_MAINSEQ_PDICT[
        "chol_utau_ulgy_mainseq_yhi"
    ],
):
    _res = _fun(lgm0, chol_utau_ulgy_mainseq_ylo, chol_utau_ulgy_mainseq_yhi)
    return _res


@jjit
def chol_utau_ul_mainseq_vs_lgm0(
    lgm0,
    chol_utau_ul_mainseq_ylo=DEFAULT_SFH_PDF_MAINSEQ_PDICT["chol_utau_ul_mainseq_ylo"],
    chol_utau_ul_mainseq_yhi=DEFAULT_SFH_PDF_MAINSEQ_PDICT["chol_utau_ul_mainseq_yhi"],
):
    _res = _fun(lgm0, chol_utau_ul_mainseq_ylo, chol_utau_ul_mainseq_yhi)
    return _res


def get_default_params(lgm):
    ulgm_MS = mean_ulgm_mainseq_vs_lgm0(lgm)
    ulgy_MS = mean_ulgy_mainseq_vs_lgm0(lgm)
    ul_MS = mean_ul_mainseq_vs_lgm0(lgm)
    utau_MS = mean_utau_mainseq_vs_lgm0(lgm)
    ulgm_ulgm_MS = chol_ulgm_ulgm_mainseq_vs_lgm0(lgm)
    ulgy_ulgy_MS = chol_ulgy_ulgy_mainseq_vs_lgm0(lgm)
    ul_ul_MS = chol_ul_ul_mainseq_vs_lgm0(lgm)
    utau_utau_MS = chol_utau_utau_mainseq_vs_lgm0(lgm)
    ulgy_ulgm_MS = chol_ulgy_ulgm_mainseq_vs_lgm0(lgm)
    ul_ulgm_MS = chol_ul_ulgm_mainseq_vs_lgm0(lgm)
    ul_ulgy_MS = chol_ul_ulgy_mainseq_vs_lgm0(lgm)
    utau_ulgm_MS = chol_utau_ulgm_mainseq_vs_lgm0(lgm)
    utau_ulgy_MS = chol_utau_ulgy_mainseq_vs_lgm0(lgm)
    utau_ul_MS = chol_utau_ul_mainseq_vs_lgm0(lgm)

    all_params = (
        ulgm_MS,
        ulgy_MS,
        ul_MS,
        utau_MS,
        ulgm_ulgm_MS,
        ulgy_ulgy_MS,
        ul_ul_MS,
        utau_utau_MS,
        ulgy_ulgm_MS,
        ul_ulgm_MS,
        ul_ulgy_MS,
        utau_ulgm_MS,
        utau_ulgy_MS,
        utau_ul_MS,
    )
    return all_params


@jjit
def get_smah_means_and_covs_mainseq(
    logmp_arr,
    mean_ulgm_mainseq_ylo=DEFAULT_SFH_PDF_MAINSEQ_PDICT["mean_ulgm_mainseq_ylo"],
    mean_ulgm_mainseq_yhi=DEFAULT_SFH_PDF_MAINSEQ_PDICT["mean_ulgm_mainseq_yhi"],
    mean_ulgy_mainseq_ylo=DEFAULT_SFH_PDF_MAINSEQ_PDICT["mean_ulgy_mainseq_ylo"],
    mean_ulgy_mainseq_yhi=DEFAULT_SFH_PDF_MAINSEQ_PDICT["mean_ulgy_mainseq_yhi"],
    mean_ul_mainseq_ylo=DEFAULT_SFH_PDF_MAINSEQ_PDICT["mean_ul_mainseq_ylo"],
    mean_ul_mainseq_yhi=DEFAULT_SFH_PDF_MAINSEQ_PDICT["mean_ul_mainseq_yhi"],
    mean_utau_mainseq_ylo=DEFAULT_SFH_PDF_MAINSEQ_PDICT["mean_utau_mainseq_ylo"],
    mean_utau_mainseq_yhi=DEFAULT_SFH_PDF_MAINSEQ_PDICT["mean_utau_mainseq_yhi"],
    chol_ulgm_ulgm_mainseq_ylo=DEFAULT_SFH_PDF_MAINSEQ_PDICT[
        "chol_ulgm_ulgm_mainseq_ylo"
    ],
    chol_ulgm_ulgm_mainseq_yhi=DEFAULT_SFH_PDF_MAINSEQ_PDICT[
        "chol_ulgm_ulgm_mainseq_yhi"
    ],
    chol_ulgy_ulgy_mainseq_ylo=DEFAULT_SFH_PDF_MAINSEQ_PDICT[
        "chol_ulgy_ulgy_mainseq_ylo"
    ],
    chol_ulgy_ulgy_mainseq_yhi=DEFAULT_SFH_PDF_MAINSEQ_PDICT[
        "chol_ulgy_ulgy_mainseq_yhi"
    ],
    chol_ul_ul_mainseq_ylo=DEFAULT_SFH_PDF_MAINSEQ_PDICT["chol_ul_ul_mainseq_ylo"],
    chol_ul_ul_mainseq_yhi=DEFAULT_SFH_PDF_MAINSEQ_PDICT["chol_ul_ul_mainseq_yhi"],
    chol_utau_utau_mainseq_ylo=DEFAULT_SFH_PDF_MAINSEQ_PDICT[
        "chol_utau_utau_mainseq_ylo"
    ],
    chol_utau_utau_mainseq_yhi=DEFAULT_SFH_PDF_MAINSEQ_PDICT[
        "chol_utau_utau_mainseq_yhi"
    ],
    chol_ulgy_ulgm_mainseq_ylo=DEFAULT_SFH_PDF_MAINSEQ_PDICT[
        "chol_ulgy_ulgm_mainseq_ylo"
    ],
    chol_ulgy_ulgm_mainseq_yhi=DEFAULT_SFH_PDF_MAINSEQ_PDICT[
        "chol_ulgy_ulgm_mainseq_yhi"
    ],
    chol_ul_ulgm_mainseq_ylo=DEFAULT_SFH_PDF_MAINSEQ_PDICT["chol_ul_ulgm_mainseq_ylo"],
    chol_ul_ulgm_mainseq_yhi=DEFAULT_SFH_PDF_MAINSEQ_PDICT["chol_ul_ulgm_mainseq_yhi"],
    chol_ul_ulgy_mainseq_ylo=DEFAULT_SFH_PDF_MAINSEQ_PDICT["chol_ul_ulgy_mainseq_ylo"],
    chol_ul_ulgy_mainseq_yhi=DEFAULT_SFH_PDF_MAINSEQ_PDICT["chol_ul_ulgy_mainseq_yhi"],
    chol_utau_ulgm_mainseq_ylo=DEFAULT_SFH_PDF_MAINSEQ_PDICT[
        "chol_utau_ulgm_mainseq_ylo"
    ],
    chol_utau_ulgm_mainseq_yhi=DEFAULT_SFH_PDF_MAINSEQ_PDICT[
        "chol_utau_ulgm_mainseq_yhi"
    ],
    chol_utau_ulgy_mainseq_ylo=DEFAULT_SFH_PDF_MAINSEQ_PDICT[
        "chol_utau_ulgy_mainseq_ylo"
    ],
    chol_utau_ulgy_mainseq_yhi=DEFAULT_SFH_PDF_MAINSEQ_PDICT[
        "chol_utau_ulgy_mainseq_yhi"
    ],
    chol_utau_ul_mainseq_ylo=DEFAULT_SFH_PDF_MAINSEQ_PDICT["chol_utau_ul_mainseq_ylo"],
    chol_utau_ul_mainseq_yhi=DEFAULT_SFH_PDF_MAINSEQ_PDICT["chol_utau_ul_mainseq_yhi"],
):
    _res = _get_mean_smah_params_mainseq(
        logmp_arr,
        mean_ulgm_mainseq_ylo,
        mean_ulgm_mainseq_yhi,
        mean_ulgy_mainseq_ylo,
        mean_ulgy_mainseq_yhi,
        mean_ul_mainseq_ylo,
        mean_ul_mainseq_yhi,
        mean_utau_mainseq_ylo,
        mean_utau_mainseq_yhi,
    )

    means_mainseq = jnp.array(_res).T

    covs_mainseq = _get_covs_mainseq(
        logmp_arr,
        chol_ulgm_ulgm_mainseq_ylo,
        chol_ulgm_ulgm_mainseq_yhi,
        chol_ulgy_ulgy_mainseq_ylo,
        chol_ulgy_ulgy_mainseq_yhi,
        chol_ul_ul_mainseq_ylo,
        chol_ul_ul_mainseq_yhi,
        chol_utau_utau_mainseq_ylo,
        chol_utau_utau_mainseq_yhi,
        chol_ulgy_ulgm_mainseq_ylo,
        chol_ulgy_ulgm_mainseq_yhi,
        chol_ul_ulgm_mainseq_ylo,
        chol_ul_ulgm_mainseq_yhi,
        chol_ul_ulgy_mainseq_ylo,
        chol_ul_ulgy_mainseq_yhi,
        chol_utau_ulgm_mainseq_ylo,
        chol_utau_ulgm_mainseq_yhi,
        chol_utau_ulgy_mainseq_ylo,
        chol_utau_ulgy_mainseq_yhi,
        chol_utau_ul_mainseq_ylo,
        chol_utau_ul_mainseq_yhi,
    )
    return means_mainseq, covs_mainseq


@jjit
def _get_mean_smah_params_mainseq(
    lgm,
    mean_ulgm_mainseq_ylo=DEFAULT_SFH_PDF_MAINSEQ_PDICT["mean_ulgm_mainseq_ylo"],
    mean_ulgm_mainseq_yhi=DEFAULT_SFH_PDF_MAINSEQ_PDICT["mean_ulgm_mainseq_yhi"],
    mean_ulgy_mainseq_ylo=DEFAULT_SFH_PDF_MAINSEQ_PDICT["mean_ulgy_mainseq_ylo"],
    mean_ulgy_mainseq_yhi=DEFAULT_SFH_PDF_MAINSEQ_PDICT["mean_ulgy_mainseq_yhi"],
    mean_ul_mainseq_ylo=DEFAULT_SFH_PDF_MAINSEQ_PDICT["mean_ul_mainseq_ylo"],
    mean_ul_mainseq_yhi=DEFAULT_SFH_PDF_MAINSEQ_PDICT["mean_ul_mainseq_yhi"],
    mean_utau_mainseq_ylo=DEFAULT_SFH_PDF_MAINSEQ_PDICT["mean_utau_mainseq_ylo"],
    mean_utau_mainseq_yhi=DEFAULT_SFH_PDF_MAINSEQ_PDICT["mean_utau_mainseq_yhi"],
):
    ulgm = mean_ulgm_mainseq_vs_lgm0(lgm, mean_ulgm_mainseq_ylo, mean_ulgm_mainseq_yhi)
    ulgy = mean_ulgy_mainseq_vs_lgm0(lgm, mean_ulgy_mainseq_ylo, mean_ulgy_mainseq_yhi)
    ul = mean_ul_mainseq_vs_lgm0(lgm, mean_ul_mainseq_ylo, mean_ul_mainseq_yhi)
    utau = mean_utau_mainseq_vs_lgm0(lgm, mean_utau_mainseq_ylo, mean_utau_mainseq_yhi)
    return ulgm, ulgy, ul, utau


@jjit
def _get_covs_mainseq(
    lgmp_arr,
    chol_ulgm_ulgm_mainseq_ylo=DEFAULT_SFH_PDF_MAINSEQ_PDICT[
        "chol_ulgm_ulgm_mainseq_ylo"
    ],
    chol_ulgm_ulgm_mainseq_yhi=DEFAULT_SFH_PDF_MAINSEQ_PDICT[
        "chol_ulgm_ulgm_mainseq_yhi"
    ],
    chol_ulgy_ulgy_mainseq_ylo=DEFAULT_SFH_PDF_MAINSEQ_PDICT[
        "chol_ulgy_ulgy_mainseq_ylo"
    ],
    chol_ulgy_ulgy_mainseq_yhi=DEFAULT_SFH_PDF_MAINSEQ_PDICT[
        "chol_ulgy_ulgy_mainseq_yhi"
    ],
    chol_ul_ul_mainseq_ylo=DEFAULT_SFH_PDF_MAINSEQ_PDICT["chol_ul_ul_mainseq_ylo"],
    chol_ul_ul_mainseq_yhi=DEFAULT_SFH_PDF_MAINSEQ_PDICT["chol_ul_ul_mainseq_yhi"],
    chol_utau_utau_mainseq_ylo=DEFAULT_SFH_PDF_MAINSEQ_PDICT[
        "chol_utau_utau_mainseq_ylo"
    ],
    chol_utau_utau_mainseq_yhi=DEFAULT_SFH_PDF_MAINSEQ_PDICT[
        "chol_utau_utau_mainseq_yhi"
    ],
    chol_ulgy_ulgm_mainseq_ylo=DEFAULT_SFH_PDF_MAINSEQ_PDICT[
        "chol_ulgy_ulgm_mainseq_ylo"
    ],
    chol_ulgy_ulgm_mainseq_yhi=DEFAULT_SFH_PDF_MAINSEQ_PDICT[
        "chol_ulgy_ulgm_mainseq_yhi"
    ],
    chol_ul_ulgm_mainseq_ylo=DEFAULT_SFH_PDF_MAINSEQ_PDICT["chol_ul_ulgm_mainseq_ylo"],
    chol_ul_ulgm_mainseq_yhi=DEFAULT_SFH_PDF_MAINSEQ_PDICT["chol_ul_ulgm_mainseq_yhi"],
    chol_ul_ulgy_mainseq_ylo=DEFAULT_SFH_PDF_MAINSEQ_PDICT["chol_ul_ulgy_mainseq_ylo"],
    chol_ul_ulgy_mainseq_yhi=DEFAULT_SFH_PDF_MAINSEQ_PDICT["chol_ul_ulgy_mainseq_yhi"],
    chol_utau_ulgm_mainseq_ylo=DEFAULT_SFH_PDF_MAINSEQ_PDICT[
        "chol_utau_ulgm_mainseq_ylo"
    ],
    chol_utau_ulgm_mainseq_yhi=DEFAULT_SFH_PDF_MAINSEQ_PDICT[
        "chol_utau_ulgm_mainseq_yhi"
    ],
    chol_utau_ulgy_mainseq_ylo=DEFAULT_SFH_PDF_MAINSEQ_PDICT[
        "chol_utau_ulgy_mainseq_ylo"
    ],
    chol_utau_ulgy_mainseq_yhi=DEFAULT_SFH_PDF_MAINSEQ_PDICT[
        "chol_utau_ulgy_mainseq_yhi"
    ],
    chol_utau_ul_mainseq_ylo=DEFAULT_SFH_PDF_MAINSEQ_PDICT["chol_utau_ul_mainseq_ylo"],
    chol_utau_ul_mainseq_yhi=DEFAULT_SFH_PDF_MAINSEQ_PDICT["chol_utau_ul_mainseq_yhi"],
):
    _res = _get_chol_params_mainseq(
        lgmp_arr,
        chol_ulgm_ulgm_mainseq_ylo,
        chol_ulgm_ulgm_mainseq_yhi,
        chol_ulgy_ulgy_mainseq_ylo,
        chol_ulgy_ulgy_mainseq_yhi,
        chol_ul_ul_mainseq_ylo,
        chol_ul_ul_mainseq_yhi,
        chol_utau_utau_mainseq_ylo,
        chol_utau_utau_mainseq_yhi,
        chol_ulgy_ulgm_mainseq_ylo,
        chol_ulgy_ulgm_mainseq_yhi,
        chol_ul_ulgm_mainseq_ylo,
        chol_ul_ulgm_mainseq_yhi,
        chol_ul_ulgy_mainseq_ylo,
        chol_ul_ulgy_mainseq_yhi,
        chol_utau_ulgm_mainseq_ylo,
        chol_utau_ulgm_mainseq_yhi,
        chol_utau_ulgy_mainseq_ylo,
        chol_utau_ulgy_mainseq_yhi,
        chol_utau_ul_mainseq_ylo,
        chol_utau_ul_mainseq_yhi,
    )
    return _get_cov_vmap(*_res)


@jjit
def _get_chol_params_mainseq(
    lgm,
    chol_ulgm_ulgm_mainseq_ylo=DEFAULT_SFH_PDF_MAINSEQ_PDICT[
        "chol_ulgm_ulgm_mainseq_ylo"
    ],
    chol_ulgm_ulgm_mainseq_yhi=DEFAULT_SFH_PDF_MAINSEQ_PDICT[
        "chol_ulgm_ulgm_mainseq_yhi"
    ],
    chol_ulgy_ulgy_mainseq_ylo=DEFAULT_SFH_PDF_MAINSEQ_PDICT[
        "chol_ulgy_ulgy_mainseq_ylo"
    ],
    chol_ulgy_ulgy_mainseq_yhi=DEFAULT_SFH_PDF_MAINSEQ_PDICT[
        "chol_ulgy_ulgy_mainseq_yhi"
    ],
    chol_ul_ul_mainseq_ylo=DEFAULT_SFH_PDF_MAINSEQ_PDICT["chol_ul_ul_mainseq_ylo"],
    chol_ul_ul_mainseq_yhi=DEFAULT_SFH_PDF_MAINSEQ_PDICT["chol_ul_ul_mainseq_yhi"],
    chol_utau_utau_mainseq_ylo=DEFAULT_SFH_PDF_MAINSEQ_PDICT[
        "chol_utau_utau_mainseq_ylo"
    ],
    chol_utau_utau_mainseq_yhi=DEFAULT_SFH_PDF_MAINSEQ_PDICT[
        "chol_utau_utau_mainseq_yhi"
    ],
    chol_ulgy_ulgm_mainseq_ylo=DEFAULT_SFH_PDF_MAINSEQ_PDICT[
        "chol_ulgy_ulgm_mainseq_ylo"
    ],
    chol_ulgy_ulgm_mainseq_yhi=DEFAULT_SFH_PDF_MAINSEQ_PDICT[
        "chol_ulgy_ulgm_mainseq_yhi"
    ],
    chol_ul_ulgm_mainseq_ylo=DEFAULT_SFH_PDF_MAINSEQ_PDICT["chol_ul_ulgm_mainseq_ylo"],
    chol_ul_ulgm_mainseq_yhi=DEFAULT_SFH_PDF_MAINSEQ_PDICT["chol_ul_ulgm_mainseq_yhi"],
    chol_ul_ulgy_mainseq_ylo=DEFAULT_SFH_PDF_MAINSEQ_PDICT["chol_ul_ulgy_mainseq_ylo"],
    chol_ul_ulgy_mainseq_yhi=DEFAULT_SFH_PDF_MAINSEQ_PDICT["chol_ul_ulgy_mainseq_yhi"],
    chol_utau_ulgm_mainseq_ylo=DEFAULT_SFH_PDF_MAINSEQ_PDICT[
        "chol_utau_ulgm_mainseq_ylo"
    ],
    chol_utau_ulgm_mainseq_yhi=DEFAULT_SFH_PDF_MAINSEQ_PDICT[
        "chol_utau_ulgm_mainseq_yhi"
    ],
    chol_utau_ulgy_mainseq_ylo=DEFAULT_SFH_PDF_MAINSEQ_PDICT[
        "chol_utau_ulgy_mainseq_ylo"
    ],
    chol_utau_ulgy_mainseq_yhi=DEFAULT_SFH_PDF_MAINSEQ_PDICT[
        "chol_utau_ulgy_mainseq_yhi"
    ],
    chol_utau_ul_mainseq_ylo=DEFAULT_SFH_PDF_MAINSEQ_PDICT["chol_utau_ul_mainseq_ylo"],
    chol_utau_ul_mainseq_yhi=DEFAULT_SFH_PDF_MAINSEQ_PDICT["chol_utau_ul_mainseq_yhi"],
):
    ulgm_ulgm = chol_ulgm_ulgm_mainseq_vs_lgm0(
        lgm, chol_ulgm_ulgm_mainseq_ylo, chol_ulgm_ulgm_mainseq_yhi
    )
    ulgy_ulgy = chol_ulgy_ulgy_mainseq_vs_lgm0(
        lgm, chol_ulgy_ulgy_mainseq_ylo, chol_ulgy_ulgy_mainseq_yhi
    )
    ul_ul = chol_ul_ul_mainseq_vs_lgm0(
        lgm, chol_ul_ul_mainseq_ylo, chol_ul_ul_mainseq_yhi
    )
    utau_utau = chol_utau_utau_mainseq_vs_lgm0(
        lgm, chol_utau_utau_mainseq_ylo, chol_utau_utau_mainseq_yhi
    )
    ulgy_ulgm = chol_ulgy_ulgm_mainseq_vs_lgm0(
        lgm, chol_ulgy_ulgm_mainseq_ylo, chol_ulgy_ulgm_mainseq_yhi
    )
    ul_ulgm = chol_ul_ulgm_mainseq_vs_lgm0(
        lgm, chol_ul_ulgm_mainseq_ylo, chol_ul_ulgm_mainseq_yhi
    )
    ul_ulgy = chol_ul_ulgy_mainseq_vs_lgm0(
        lgm, chol_ul_ulgy_mainseq_ylo, chol_ul_ulgy_mainseq_yhi
    )
    utau_ulgm = chol_utau_ulgm_mainseq_vs_lgm0(
        lgm, chol_utau_ulgm_mainseq_ylo, chol_utau_ulgm_mainseq_yhi
    )
    utau_ulgy = chol_utau_ulgy_mainseq_vs_lgm0(
        lgm, chol_utau_ulgy_mainseq_ylo, chol_utau_ulgy_mainseq_yhi
    )
    utau_ul = chol_utau_ul_mainseq_vs_lgm0(
        lgm, chol_utau_ul_mainseq_ylo, chol_utau_ul_mainseq_yhi
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
