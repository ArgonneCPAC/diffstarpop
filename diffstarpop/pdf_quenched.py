"""Model of a quenched galaxy population calibrated to SMDPL halos."""
from collections import OrderedDict
from jax import jit as jjit
from jax import numpy as jnp
from jax import vmap

TODAY = 13.8
LGT0 = jnp.log10(TODAY)

LGM_X0, LGM_K = 13.0, 0.5

DEFAULT_SFH_PDF_QUENCH_PARAMS = OrderedDict(
    frac_quench_x0=11.860,
    frac_quench_k=1.611,
    frac_quench_ylo=-0.872,
    frac_quench_yhi=2.139,
    mean_ulgm_quench_ylo=11.540,
    mean_ulgm_quench_yhi=12.080,
    mean_ulgy_quench_ylo=0.481,
    mean_ulgy_quench_yhi=-0.223,
    mean_ul_quench_ylo=-1.274,
    mean_ul_quench_yhi=1.766,
    mean_utau_quench_ylo=55.480,
    mean_utau_quench_yhi=-66.540,
    mean_uqt_quench_ylo=1.744,
    mean_uqt_quench_yhi=0.042,
    mean_uqs_quench_ylo=-2.979,
    mean_uqs_quench_yhi=3.520,
    mean_udrop_quench_ylo=-0.508,
    mean_udrop_quench_yhi=-3.785,
    mean_urej_quench_ylo=2.139,
    mean_urej_quench_yhi=-3.043,
    chol_ulgm_ulgm_quench_ylo=-1.645,
    chol_ulgm_ulgm_quench_yhi=0.010,
    chol_ulgy_ulgy_quench_ylo=-1.125,
    chol_ulgy_ulgy_quench_yhi=-0.530,
    chol_ul_ul_quench_ylo=-0.701,
    chol_ul_ul_quench_yhi=0.544,
    chol_utau_utau_quench_ylo=0.833,
    chol_utau_utau_quench_yhi=1.100,
    chol_uqt_uqt_quench_ylo=-1.001,
    chol_uqt_uqt_quench_yhi=-1.228,
    chol_uqs_uqs_quench_ylo=-0.814,
    chol_uqs_uqs_quench_yhi=-0.560,
    chol_udrop_udrop_quench_ylo=-0.612,
    chol_udrop_udrop_quench_yhi=-0.824,
    chol_urej_urej_quench_ylo=0.560,
    chol_urej_urej_quench_yhi=-1.103,
    chol_ulgy_ulgm_quench_ylo=-0.809,
    chol_ulgy_ulgm_quench_yhi=-1.790,
    chol_ul_ulgm_quench_ylo=0.277,
    chol_ul_ulgm_quench_yhi=0.357,
    chol_ul_ulgy_quench_ylo=0.152,
    chol_ul_ulgy_quench_yhi=0.068,
    chol_utau_ulgm_quench_ylo=-1.214,
    chol_utau_ulgm_quench_yhi=-0.822,
    chol_utau_ulgy_quench_ylo=0.115,
    chol_utau_ulgy_quench_yhi=0.204,
    chol_utau_ul_quench_ylo=-0.566,
    chol_utau_ul_quench_yhi=-0.848,
    chol_uqt_ulgm_quench_ylo=0.632,
    chol_uqt_ulgm_quench_yhi=0.486,
    chol_uqt_ulgy_quench_ylo=-0.003,
    chol_uqt_ulgy_quench_yhi=-0.210,
    chol_uqt_ul_quench_ylo=0.109,
    chol_uqt_ul_quench_yhi=0.092,
    chol_uqt_utau_quench_ylo=0.542,
    chol_uqt_utau_quench_yhi=-0.029,
    chol_uqs_ulgm_quench_ylo=0.541,
    chol_uqs_ulgm_quench_yhi=0.864,
    chol_uqs_ulgy_quench_ylo=0.479,
    chol_uqs_ulgy_quench_yhi=0.624,
    chol_uqs_ul_quench_ylo=0.582,
    chol_uqs_ul_quench_yhi=0.717,
    chol_uqs_utau_quench_ylo=0.050,
    chol_uqs_utau_quench_yhi=-0.037,
    chol_uqs_uqt_quench_ylo=-0.395,
    chol_uqs_uqt_quench_yhi=-0.508,
    chol_udrop_ulgm_quench_ylo=-0.811,
    chol_udrop_ulgm_quench_yhi=-1.007,
    chol_udrop_ulgy_quench_ylo=-0.213,
    chol_udrop_ulgy_quench_yhi=-0.564,
    chol_udrop_ul_quench_ylo=0.131,
    chol_udrop_ul_quench_yhi=-0.305,
    chol_udrop_utau_quench_ylo=0.446,
    chol_udrop_utau_quench_yhi=0.297,
    chol_udrop_uqt_quench_ylo=2.323,
    chol_udrop_uqt_quench_yhi=3.009,
    chol_udrop_uqs_quench_ylo=1.021,
    chol_udrop_uqs_quench_yhi=-0.074,
    chol_urej_ulgm_quench_ylo=-0.099,
    chol_urej_ulgm_quench_yhi=-0.695,
    chol_urej_ulgy_quench_ylo=0.069,
    chol_urej_ulgy_quench_yhi=-1.062,
    chol_urej_ul_quench_ylo=0.531,
    chol_urej_ul_quench_yhi=1.126,
    chol_urej_utau_quench_ylo=0.351,
    chol_urej_utau_quench_yhi=-0.137,
    chol_urej_uqt_quench_ylo=-0.508,
    chol_urej_uqt_quench_yhi=0.758,
    chol_urej_uqs_quench_ylo=1.561,
    chol_urej_uqs_quench_yhi=2.030,
    chol_urej_udrop_quench_ylo=-1.445,
    chol_urej_udrop_quench_yhi=-2.245,
)


@jjit
def _sigmoid(x, logtc, k, ymin, ymax):
    height_diff = ymax - ymin
    return ymin + height_diff / (1.0 + jnp.exp(-k * (x - logtc)))


@jjit
def _inverse_sigmoid(y, x0=0, k=1, ymin=-1, ymax=1):
    lnarg = (ymax - ymin) / (y - ymin) - 1
    return x0 - jnp.log(lnarg) / k


@jjit
def _fun(x, ymin, ymax):
    return _sigmoid(x, LGM_X0, LGM_K, ymin, ymax)


@jjit
def _fun_Mcrit(x, ymin, ymax):
    return _sigmoid(x, 12.0, 4.0, ymin, ymax)


@jjit
def _fun_chol_diag(x, ymin, ymax):
    _res = 10 ** _fun(x, ymin, ymax)
    return _res


@jjit
def _bound_fquench(x):
    return _sigmoid(x, 0.5, 4.0, 0.0, 1.0)


@jjit
def _fun_fquench(x, x0, k, ymin, ymax):
    _res = _sigmoid(x, x0, k, ymin, ymax)
    return _bound_fquench(_res)


@jjit
def _get_cov_scalar(
    ulgm_ulgm,
    ulgy_ulgy,
    ul_ul,
    utau_utau,
    uqt_uqt,
    uqs_uqs,
    udrop_udrop,
    urej_urej,
    ulgy_ulgm,
    ul_ulgm,
    ul_ulgy,
    utau_ulgm,
    utau_ulgy,
    utau_ul,
    uqt_ulgm,
    uqt_ulgy,
    uqt_ul,
    uqt_utau,
    uqs_ulgm,
    uqs_ulgy,
    uqs_ul,
    uqs_utau,
    uqs_uqt,
    udrop_ulgm,
    udrop_ulgy,
    udrop_ul,
    udrop_utau,
    udrop_uqt,
    udrop_uqs,
    urej_ulgm,
    urej_ulgy,
    urej_ul,
    urej_utau,
    urej_uqt,
    urej_uqs,
    urej_udrop,
):
    chol = jnp.zeros((8, 8)).astype("f4")
    chol = chol.at[(0, 0)].set(ulgm_ulgm)
    chol = chol.at[(1, 1)].set(ulgy_ulgy)
    chol = chol.at[(2, 2)].set(ul_ul)
    chol = chol.at[(3, 3)].set(utau_utau)
    chol = chol.at[(4, 4)].set(uqt_uqt)
    chol = chol.at[(5, 5)].set(uqs_uqs)
    chol = chol.at[(6, 6)].set(udrop_udrop)
    chol = chol.at[(7, 7)].set(urej_urej)

    chol = chol.at[(1, 0)].set(ulgy_ulgm * ulgy_ulgy * ulgm_ulgm)
    chol = chol.at[(2, 0)].set(ul_ulgm * ul_ul * ulgm_ulgm)
    chol = chol.at[(2, 1)].set(ul_ulgy * ul_ul * ulgy_ulgy)
    chol = chol.at[(3, 0)].set(utau_ulgm * utau_utau * ulgm_ulgm)
    chol = chol.at[(3, 1)].set(utau_ulgy * utau_utau * ulgy_ulgy)
    chol = chol.at[(3, 2)].set(utau_ul * utau_utau * ul_ul)
    chol = chol.at[(4, 0)].set(uqt_ulgm * uqt_uqt * ulgm_ulgm)
    chol = chol.at[(4, 1)].set(uqt_ulgy * uqt_uqt * ulgy_ulgy)
    chol = chol.at[(4, 2)].set(uqt_ul * uqt_uqt * ul_ul)
    chol = chol.at[(4, 3)].set(uqt_utau * uqt_uqt * utau_utau)
    chol = chol.at[(5, 0)].set(uqs_ulgm * uqs_uqs * ulgm_ulgm)
    chol = chol.at[(5, 1)].set(uqs_ulgy * uqs_uqs * ulgy_ulgy)
    chol = chol.at[(5, 2)].set(uqs_ul * uqs_uqs * ul_ul)
    chol = chol.at[(5, 3)].set(uqs_utau * uqs_uqs * utau_utau)
    chol = chol.at[(5, 4)].set(uqs_uqt * uqs_uqs * uqt_uqt)
    chol = chol.at[(6, 0)].set(udrop_ulgm * udrop_udrop * ulgm_ulgm)
    chol = chol.at[(6, 1)].set(udrop_ulgy * udrop_udrop * ulgy_ulgy)
    chol = chol.at[(6, 2)].set(udrop_ul * udrop_udrop * ul_ul)
    chol = chol.at[(6, 3)].set(udrop_utau * udrop_udrop * utau_utau)
    chol = chol.at[(6, 4)].set(udrop_uqt * udrop_udrop * uqt_uqt)
    chol = chol.at[(6, 5)].set(udrop_uqs * udrop_udrop * uqs_uqs)
    chol = chol.at[(7, 0)].set(urej_ulgm * urej_urej * ulgm_ulgm)
    chol = chol.at[(7, 1)].set(urej_ulgy * urej_urej * ulgy_ulgy)
    chol = chol.at[(7, 2)].set(urej_ul * urej_urej * ul_ul)
    chol = chol.at[(7, 3)].set(urej_utau * urej_urej * utau_utau)
    chol = chol.at[(7, 4)].set(urej_uqt * urej_urej * uqt_uqt)
    chol = chol.at[(7, 5)].set(urej_uqs * urej_urej * uqs_uqs)
    chol = chol.at[(7, 6)].set(urej_udrop * urej_urej * udrop_udrop)

    cov = jnp.dot(chol, chol.T)
    return cov


_get_cov_vmap = jjit(vmap(_get_cov_scalar, in_axes=(*[0] * 36,)))


@jjit
def frac_quench_vs_lgm0(
    lgm0,
    frac_quench_x0=DEFAULT_SFH_PDF_QUENCH_PARAMS["frac_quench_x0"],
    frac_quench_k=DEFAULT_SFH_PDF_QUENCH_PARAMS["frac_quench_k"],
    frac_quench_ylo=DEFAULT_SFH_PDF_QUENCH_PARAMS["frac_quench_ylo"],
    frac_quench_yhi=DEFAULT_SFH_PDF_QUENCH_PARAMS["frac_quench_yhi"],
):
    return _fun_fquench(
        lgm0, frac_quench_x0, frac_quench_k, frac_quench_ylo, frac_quench_yhi
    )


@jjit
def mean_ulgm_quench_vs_lgm0(
    lgm0,
    mean_ulgm_quench_ylo=DEFAULT_SFH_PDF_QUENCH_PARAMS["mean_ulgm_quench_ylo"],
    mean_ulgm_quench_yhi=DEFAULT_SFH_PDF_QUENCH_PARAMS["mean_ulgm_quench_yhi"],
):
    return _fun_Mcrit(lgm0, mean_ulgm_quench_ylo, mean_ulgm_quench_yhi)


@jjit
def mean_ulgy_quench_vs_lgm0(
    lgm0,
    mean_ulgy_quench_ylo=DEFAULT_SFH_PDF_QUENCH_PARAMS["mean_ulgy_quench_ylo"],
    mean_ulgy_quench_yhi=DEFAULT_SFH_PDF_QUENCH_PARAMS["mean_ulgy_quench_yhi"],
):
    return _fun(lgm0, mean_ulgy_quench_ylo, mean_ulgy_quench_yhi)


@jjit
def mean_ul_quench_vs_lgm0(
    lgm0,
    mean_ul_quench_ylo=DEFAULT_SFH_PDF_QUENCH_PARAMS["mean_ul_quench_ylo"],
    mean_ul_quench_yhi=DEFAULT_SFH_PDF_QUENCH_PARAMS["mean_ul_quench_yhi"],
):
    return _fun(lgm0, mean_ul_quench_ylo, mean_ul_quench_yhi)


@jjit
def mean_utau_quench_vs_lgm0(
    lgm0,
    mean_utau_quench_ylo=DEFAULT_SFH_PDF_QUENCH_PARAMS["mean_utau_quench_ylo"],
    mean_utau_quench_yhi=DEFAULT_SFH_PDF_QUENCH_PARAMS["mean_utau_quench_yhi"],
):
    return _fun(lgm0, mean_utau_quench_ylo, mean_utau_quench_yhi)


@jjit
def mean_uqt_quench_vs_lgm0(
    lgm0,
    mean_uqt_quench_ylo=DEFAULT_SFH_PDF_QUENCH_PARAMS["mean_uqt_quench_ylo"],
    mean_uqt_quench_yhi=DEFAULT_SFH_PDF_QUENCH_PARAMS["mean_uqt_quench_yhi"],
):
    return _fun(lgm0, mean_uqt_quench_ylo, mean_uqt_quench_yhi)


@jjit
def mean_uqs_quench_vs_lgm0(
    lgm0,
    mean_uqs_quench_ylo=DEFAULT_SFH_PDF_QUENCH_PARAMS["mean_uqs_quench_ylo"],
    mean_uqs_quench_yhi=DEFAULT_SFH_PDF_QUENCH_PARAMS["mean_uqs_quench_yhi"],
):
    return _fun(lgm0, mean_uqs_quench_ylo, mean_uqs_quench_yhi)


@jjit
def mean_udrop_quench_vs_lgm0(
    lgm0,
    mean_udrop_quench_ylo=DEFAULT_SFH_PDF_QUENCH_PARAMS["mean_udrop_quench_ylo"],
    mean_udrop_quench_yhi=DEFAULT_SFH_PDF_QUENCH_PARAMS["mean_udrop_quench_yhi"],
):
    return _fun(lgm0, mean_udrop_quench_ylo, mean_udrop_quench_yhi)


@jjit
def mean_urej_quench_vs_lgm0(
    lgm0,
    mean_urej_quench_ylo=DEFAULT_SFH_PDF_QUENCH_PARAMS["mean_urej_quench_ylo"],
    mean_urej_quench_yhi=DEFAULT_SFH_PDF_QUENCH_PARAMS["mean_urej_quench_yhi"],
):
    return _fun(lgm0, mean_urej_quench_ylo, mean_urej_quench_yhi)


@jjit
def chol_ulgm_ulgm_quench_vs_lgm0(
    lgm0,
    chol_ulgm_ulgm_quench_ylo=DEFAULT_SFH_PDF_QUENCH_PARAMS[
        "chol_ulgm_ulgm_quench_ylo"
    ],
    chol_ulgm_ulgm_quench_yhi=DEFAULT_SFH_PDF_QUENCH_PARAMS[
        "chol_ulgm_ulgm_quench_yhi"
    ],
):
    _res = _fun_chol_diag(lgm0, chol_ulgm_ulgm_quench_ylo, chol_ulgm_ulgm_quench_yhi)
    return _res


@jjit
def chol_ulgy_ulgy_quench_vs_lgm0(
    lgm0,
    chol_ulgy_ulgy_quench_ylo=DEFAULT_SFH_PDF_QUENCH_PARAMS[
        "chol_ulgy_ulgy_quench_ylo"
    ],
    chol_ulgy_ulgy_quench_yhi=DEFAULT_SFH_PDF_QUENCH_PARAMS[
        "chol_ulgy_ulgy_quench_yhi"
    ],
):
    _res = _fun_chol_diag(lgm0, chol_ulgy_ulgy_quench_ylo, chol_ulgy_ulgy_quench_yhi)
    return _res


@jjit
def chol_ul_ul_quench_vs_lgm0(
    lgm0,
    chol_ul_ul_quench_ylo=DEFAULT_SFH_PDF_QUENCH_PARAMS["chol_ul_ul_quench_ylo"],
    chol_ul_ul_quench_yhi=DEFAULT_SFH_PDF_QUENCH_PARAMS["chol_ul_ul_quench_yhi"],
):
    _res = _fun_chol_diag(lgm0, chol_ul_ul_quench_ylo, chol_ul_ul_quench_yhi)
    return _res


@jjit
def chol_utau_utau_quench_vs_lgm0(
    lgm0,
    chol_utau_utau_quench_ylo=DEFAULT_SFH_PDF_QUENCH_PARAMS[
        "chol_utau_utau_quench_ylo"
    ],
    chol_utau_utau_quench_yhi=DEFAULT_SFH_PDF_QUENCH_PARAMS[
        "chol_utau_utau_quench_yhi"
    ],
):
    _res = _fun_chol_diag(lgm0, chol_utau_utau_quench_ylo, chol_utau_utau_quench_yhi)
    return _res


@jjit
def chol_uqt_uqt_quench_vs_lgm0(
    lgm0,
    chol_uqt_uqt_quench_ylo=DEFAULT_SFH_PDF_QUENCH_PARAMS["chol_uqt_uqt_quench_ylo"],
    chol_uqt_uqt_quench_yhi=DEFAULT_SFH_PDF_QUENCH_PARAMS["chol_uqt_uqt_quench_yhi"],
):
    _res = _fun_chol_diag(lgm0, chol_uqt_uqt_quench_ylo, chol_uqt_uqt_quench_yhi)
    return _res


@jjit
def chol_uqs_uqs_quench_vs_lgm0(
    lgm0,
    chol_uqs_uqs_quench_ylo=DEFAULT_SFH_PDF_QUENCH_PARAMS["chol_uqs_uqs_quench_ylo"],
    chol_uqs_uqs_quench_yhi=DEFAULT_SFH_PDF_QUENCH_PARAMS["chol_uqs_uqs_quench_yhi"],
):
    _res = _fun_chol_diag(lgm0, chol_uqs_uqs_quench_ylo, chol_uqs_uqs_quench_yhi)
    return _res


@jjit
def chol_udrop_udrop_quench_vs_lgm0(
    lgm0,
    chol_udrop_udrop_quench_ylo=DEFAULT_SFH_PDF_QUENCH_PARAMS[
        "chol_udrop_udrop_quench_ylo"
    ],
    chol_udrop_udrop_quench_yhi=DEFAULT_SFH_PDF_QUENCH_PARAMS[
        "chol_udrop_udrop_quench_yhi"
    ],
):
    _res = _fun_chol_diag(
        lgm0, chol_udrop_udrop_quench_ylo, chol_udrop_udrop_quench_yhi
    )
    return _res


@jjit
def chol_urej_urej_quench_vs_lgm0(
    lgm0,
    chol_urej_urej_quench_ylo=DEFAULT_SFH_PDF_QUENCH_PARAMS[
        "chol_urej_urej_quench_ylo"
    ],
    chol_urej_urej_quench_yhi=DEFAULT_SFH_PDF_QUENCH_PARAMS[
        "chol_urej_urej_quench_yhi"
    ],
):
    _res = _fun_chol_diag(lgm0, chol_urej_urej_quench_ylo, chol_urej_urej_quench_yhi)
    return _res


@jjit
def chol_ulgy_ulgm_quench_vs_lgm0(
    lgm0,
    chol_ulgy_ulgm_quench_ylo=DEFAULT_SFH_PDF_QUENCH_PARAMS[
        "chol_ulgy_ulgm_quench_ylo"
    ],
    chol_ulgy_ulgm_quench_yhi=DEFAULT_SFH_PDF_QUENCH_PARAMS[
        "chol_ulgy_ulgm_quench_yhi"
    ],
):
    _res = _fun(lgm0, chol_ulgy_ulgm_quench_ylo, chol_ulgy_ulgm_quench_yhi)
    return _res


@jjit
def chol_ul_ulgm_quench_vs_lgm0(
    lgm0,
    chol_ul_ulgm_quench_ylo=DEFAULT_SFH_PDF_QUENCH_PARAMS["chol_ul_ulgm_quench_ylo"],
    chol_ul_ulgm_quench_yhi=DEFAULT_SFH_PDF_QUENCH_PARAMS["chol_ul_ulgm_quench_yhi"],
):
    _res = _fun(lgm0, chol_ul_ulgm_quench_ylo, chol_ul_ulgm_quench_yhi)
    return _res


@jjit
def chol_ul_ulgy_quench_vs_lgm0(
    lgm0,
    chol_ul_ulgy_quench_ylo=DEFAULT_SFH_PDF_QUENCH_PARAMS["chol_ul_ulgy_quench_ylo"],
    chol_ul_ulgy_quench_yhi=DEFAULT_SFH_PDF_QUENCH_PARAMS["chol_ul_ulgy_quench_yhi"],
):
    _res = _fun(lgm0, chol_ul_ulgy_quench_ylo, chol_ul_ulgy_quench_yhi)
    return _res


@jjit
def chol_utau_ulgm_quench_vs_lgm0(
    lgm0,
    chol_utau_ulgm_quench_ylo=DEFAULT_SFH_PDF_QUENCH_PARAMS[
        "chol_utau_ulgm_quench_ylo"
    ],
    chol_utau_ulgm_quench_yhi=DEFAULT_SFH_PDF_QUENCH_PARAMS[
        "chol_utau_ulgm_quench_yhi"
    ],
):
    _res = _fun(lgm0, chol_utau_ulgm_quench_ylo, chol_utau_ulgm_quench_yhi)
    return _res


@jjit
def chol_utau_ulgy_quench_vs_lgm0(
    lgm0,
    chol_utau_ulgy_quench_ylo=DEFAULT_SFH_PDF_QUENCH_PARAMS[
        "chol_utau_ulgy_quench_ylo"
    ],
    chol_utau_ulgy_quench_yhi=DEFAULT_SFH_PDF_QUENCH_PARAMS[
        "chol_utau_ulgy_quench_yhi"
    ],
):
    _res = _fun(lgm0, chol_utau_ulgy_quench_ylo, chol_utau_ulgy_quench_yhi)
    return _res


@jjit
def chol_utau_ul_quench_vs_lgm0(
    lgm0,
    chol_utau_ul_quench_ylo=DEFAULT_SFH_PDF_QUENCH_PARAMS["chol_utau_ul_quench_ylo"],
    chol_utau_ul_quench_yhi=DEFAULT_SFH_PDF_QUENCH_PARAMS["chol_utau_ul_quench_yhi"],
):
    _res = _fun(lgm0, chol_utau_ul_quench_ylo, chol_utau_ul_quench_yhi)
    return _res


@jjit
def chol_uqt_ulgm_quench_vs_lgm0(
    lgm0,
    chol_uqt_ulgm_quench_ylo=DEFAULT_SFH_PDF_QUENCH_PARAMS["chol_uqt_ulgm_quench_ylo"],
    chol_uqt_ulgm_quench_yhi=DEFAULT_SFH_PDF_QUENCH_PARAMS["chol_uqt_ulgm_quench_yhi"],
):
    _res = _fun(lgm0, chol_uqt_ulgm_quench_ylo, chol_uqt_ulgm_quench_yhi)
    return _res


@jjit
def chol_uqt_ulgy_quench_vs_lgm0(
    lgm0,
    chol_uqt_ulgy_quench_ylo=DEFAULT_SFH_PDF_QUENCH_PARAMS["chol_uqt_ulgy_quench_ylo"],
    chol_uqt_ulgy_quench_yhi=DEFAULT_SFH_PDF_QUENCH_PARAMS["chol_uqt_ulgy_quench_yhi"],
):
    _res = _fun(lgm0, chol_uqt_ulgy_quench_ylo, chol_uqt_ulgy_quench_yhi)
    return _res


@jjit
def chol_uqt_ul_quench_vs_lgm0(
    lgm0,
    chol_uqt_ul_quench_ylo=DEFAULT_SFH_PDF_QUENCH_PARAMS["chol_uqt_ul_quench_ylo"],
    chol_uqt_ul_quench_yhi=DEFAULT_SFH_PDF_QUENCH_PARAMS["chol_uqt_ul_quench_yhi"],
):
    _res = _fun(lgm0, chol_uqt_ul_quench_ylo, chol_uqt_ul_quench_yhi)
    return _res


@jjit
def chol_uqt_utau_quench_vs_lgm0(
    lgm0,
    chol_uqt_utau_quench_ylo=DEFAULT_SFH_PDF_QUENCH_PARAMS["chol_uqt_utau_quench_ylo"],
    chol_uqt_utau_quench_yhi=DEFAULT_SFH_PDF_QUENCH_PARAMS["chol_uqt_utau_quench_yhi"],
):
    _res = _fun(lgm0, chol_uqt_utau_quench_ylo, chol_uqt_utau_quench_yhi)
    return _res


@jjit
def chol_uqs_ulgm_quench_vs_lgm0(
    lgm0,
    chol_uqs_ulgm_quench_ylo=DEFAULT_SFH_PDF_QUENCH_PARAMS["chol_uqs_ulgm_quench_ylo"],
    chol_uqs_ulgm_quench_yhi=DEFAULT_SFH_PDF_QUENCH_PARAMS["chol_uqs_ulgm_quench_yhi"],
):
    _res = _fun(lgm0, chol_uqs_ulgm_quench_ylo, chol_uqs_ulgm_quench_yhi)
    return _res


@jjit
def chol_uqs_ulgy_quench_vs_lgm0(
    lgm0,
    chol_uqs_ulgy_quench_ylo=DEFAULT_SFH_PDF_QUENCH_PARAMS["chol_uqs_ulgy_quench_ylo"],
    chol_uqs_ulgy_quench_yhi=DEFAULT_SFH_PDF_QUENCH_PARAMS["chol_uqs_ulgy_quench_yhi"],
):
    _res = _fun(lgm0, chol_uqs_ulgy_quench_ylo, chol_uqs_ulgy_quench_yhi)
    return _res


@jjit
def chol_uqs_ul_quench_vs_lgm0(
    lgm0,
    chol_uqs_ul_quench_ylo=DEFAULT_SFH_PDF_QUENCH_PARAMS["chol_uqs_ul_quench_ylo"],
    chol_uqs_ul_quench_yhi=DEFAULT_SFH_PDF_QUENCH_PARAMS["chol_uqs_ul_quench_yhi"],
):
    _res = _fun(lgm0, chol_uqs_ul_quench_ylo, chol_uqs_ul_quench_yhi)
    return _res


@jjit
def chol_uqs_utau_quench_vs_lgm0(
    lgm0,
    chol_uqs_utau_quench_ylo=DEFAULT_SFH_PDF_QUENCH_PARAMS["chol_uqs_utau_quench_ylo"],
    chol_uqs_utau_quench_yhi=DEFAULT_SFH_PDF_QUENCH_PARAMS["chol_uqs_utau_quench_yhi"],
):
    _res = _fun(lgm0, chol_uqs_utau_quench_ylo, chol_uqs_utau_quench_yhi)
    return _res


@jjit
def chol_uqs_uqt_quench_vs_lgm0(
    lgm0,
    chol_uqs_uqt_quench_ylo=DEFAULT_SFH_PDF_QUENCH_PARAMS["chol_uqs_uqt_quench_ylo"],
    chol_uqs_uqt_quench_yhi=DEFAULT_SFH_PDF_QUENCH_PARAMS["chol_uqs_uqt_quench_yhi"],
):
    _res = _fun(lgm0, chol_uqs_uqt_quench_ylo, chol_uqs_uqt_quench_yhi)
    return _res


@jjit
def chol_udrop_ulgm_quench_vs_lgm0(
    lgm0,
    chol_udrop_ulgm_quench_ylo=DEFAULT_SFH_PDF_QUENCH_PARAMS[
        "chol_udrop_ulgm_quench_ylo"
    ],
    chol_udrop_ulgm_quench_yhi=DEFAULT_SFH_PDF_QUENCH_PARAMS[
        "chol_udrop_ulgm_quench_yhi"
    ],
):
    _res = _fun(lgm0, chol_udrop_ulgm_quench_ylo, chol_udrop_ulgm_quench_yhi)
    return _res


@jjit
def chol_udrop_ulgy_quench_vs_lgm0(
    lgm0,
    chol_udrop_ulgy_quench_ylo=DEFAULT_SFH_PDF_QUENCH_PARAMS[
        "chol_udrop_ulgy_quench_ylo"
    ],
    chol_udrop_ulgy_quench_yhi=DEFAULT_SFH_PDF_QUENCH_PARAMS[
        "chol_udrop_ulgy_quench_yhi"
    ],
):
    _res = _fun(lgm0, chol_udrop_ulgy_quench_ylo, chol_udrop_ulgy_quench_yhi)
    return _res


@jjit
def chol_udrop_ul_quench_vs_lgm0(
    lgm0,
    chol_udrop_ul_quench_ylo=DEFAULT_SFH_PDF_QUENCH_PARAMS["chol_udrop_ul_quench_ylo"],
    chol_udrop_ul_quench_yhi=DEFAULT_SFH_PDF_QUENCH_PARAMS["chol_udrop_ul_quench_yhi"],
):
    _res = _fun(lgm0, chol_udrop_ul_quench_ylo, chol_udrop_ul_quench_yhi)
    return _res


@jjit
def chol_udrop_utau_quench_vs_lgm0(
    lgm0,
    chol_udrop_utau_quench_ylo=DEFAULT_SFH_PDF_QUENCH_PARAMS[
        "chol_udrop_utau_quench_ylo"
    ],
    chol_udrop_utau_quench_yhi=DEFAULT_SFH_PDF_QUENCH_PARAMS[
        "chol_udrop_utau_quench_yhi"
    ],
):
    _res = _fun(lgm0, chol_udrop_utau_quench_ylo, chol_udrop_utau_quench_yhi)
    return _res


@jjit
def chol_udrop_uqt_quench_vs_lgm0(
    lgm0,
    chol_udrop_uqt_quench_ylo=DEFAULT_SFH_PDF_QUENCH_PARAMS[
        "chol_udrop_uqt_quench_ylo"
    ],
    chol_udrop_uqt_quench_yhi=DEFAULT_SFH_PDF_QUENCH_PARAMS[
        "chol_udrop_uqt_quench_yhi"
    ],
):
    _res = _fun(lgm0, chol_udrop_uqt_quench_ylo, chol_udrop_uqt_quench_yhi)
    return _res


@jjit
def chol_udrop_uqs_quench_vs_lgm0(
    lgm0,
    chol_udrop_uqs_quench_ylo=DEFAULT_SFH_PDF_QUENCH_PARAMS[
        "chol_udrop_uqs_quench_ylo"
    ],
    chol_udrop_uqs_quench_yhi=DEFAULT_SFH_PDF_QUENCH_PARAMS[
        "chol_udrop_uqs_quench_yhi"
    ],
):
    _res = _fun(lgm0, chol_udrop_uqs_quench_ylo, chol_udrop_uqs_quench_yhi)
    return _res


@jjit
def chol_urej_ulgm_quench_vs_lgm0(
    lgm0,
    chol_urej_ulgm_quench_ylo=DEFAULT_SFH_PDF_QUENCH_PARAMS[
        "chol_urej_ulgm_quench_ylo"
    ],
    chol_urej_ulgm_quench_yhi=DEFAULT_SFH_PDF_QUENCH_PARAMS[
        "chol_urej_ulgm_quench_yhi"
    ],
):
    _res = _fun(lgm0, chol_urej_ulgm_quench_ylo, chol_urej_ulgm_quench_yhi)
    return _res


@jjit
def chol_urej_ulgy_quench_vs_lgm0(
    lgm0,
    chol_urej_ulgy_quench_ylo=DEFAULT_SFH_PDF_QUENCH_PARAMS[
        "chol_urej_ulgy_quench_ylo"
    ],
    chol_urej_ulgy_quench_yhi=DEFAULT_SFH_PDF_QUENCH_PARAMS[
        "chol_urej_ulgy_quench_yhi"
    ],
):
    _res = _fun(lgm0, chol_urej_ulgy_quench_ylo, chol_urej_ulgy_quench_yhi)
    return _res


@jjit
def chol_urej_ul_quench_vs_lgm0(
    lgm0,
    chol_urej_ul_quench_ylo=DEFAULT_SFH_PDF_QUENCH_PARAMS["chol_urej_ul_quench_ylo"],
    chol_urej_ul_quench_yhi=DEFAULT_SFH_PDF_QUENCH_PARAMS["chol_urej_ul_quench_yhi"],
):
    _res = _fun(lgm0, chol_urej_ul_quench_ylo, chol_urej_ul_quench_yhi)
    return _res


@jjit
def chol_urej_utau_quench_vs_lgm0(
    lgm0,
    chol_urej_utau_quench_ylo=DEFAULT_SFH_PDF_QUENCH_PARAMS[
        "chol_urej_utau_quench_ylo"
    ],
    chol_urej_utau_quench_yhi=DEFAULT_SFH_PDF_QUENCH_PARAMS[
        "chol_urej_utau_quench_yhi"
    ],
):
    _res = _fun(lgm0, chol_urej_utau_quench_ylo, chol_urej_utau_quench_yhi)
    return _res


@jjit
def chol_urej_uqt_quench_vs_lgm0(
    lgm0,
    chol_urej_uqt_quench_ylo=DEFAULT_SFH_PDF_QUENCH_PARAMS["chol_urej_uqt_quench_ylo"],
    chol_urej_uqt_quench_yhi=DEFAULT_SFH_PDF_QUENCH_PARAMS["chol_urej_uqt_quench_yhi"],
):
    _res = _fun(lgm0, chol_urej_uqt_quench_ylo, chol_urej_uqt_quench_yhi)
    return _res


@jjit
def chol_urej_uqs_quench_vs_lgm0(
    lgm0,
    chol_urej_uqs_quench_ylo=DEFAULT_SFH_PDF_QUENCH_PARAMS["chol_urej_uqs_quench_ylo"],
    chol_urej_uqs_quench_yhi=DEFAULT_SFH_PDF_QUENCH_PARAMS["chol_urej_uqs_quench_yhi"],
):
    _res = _fun(lgm0, chol_urej_uqs_quench_ylo, chol_urej_uqs_quench_yhi)
    return _res


@jjit
def chol_urej_udrop_quench_vs_lgm0(
    lgm0,
    chol_urej_udrop_quench_ylo=DEFAULT_SFH_PDF_QUENCH_PARAMS[
        "chol_urej_udrop_quench_ylo"
    ],
    chol_urej_udrop_quench_yhi=DEFAULT_SFH_PDF_QUENCH_PARAMS[
        "chol_urej_udrop_quench_yhi"
    ],
):
    _res = _fun(lgm0, chol_urej_udrop_quench_ylo, chol_urej_udrop_quench_yhi)
    return _res


def get_default_params(lgm):
    frac_quench = frac_quench_vs_lgm0(lgm)
    ulgm_q = mean_ulgm_quench_vs_lgm0(lgm)
    ulgy_q = mean_ulgy_quench_vs_lgm0(lgm)
    ul_q = mean_ul_quench_vs_lgm0(lgm)
    utau_q = mean_utau_quench_vs_lgm0(lgm)
    uqt_q = mean_uqt_quench_vs_lgm0(lgm)
    uqs_q = mean_uqs_quench_vs_lgm0(lgm)
    udrop_q = mean_udrop_quench_vs_lgm0(lgm)
    urej_q = mean_urej_quench_vs_lgm0(lgm)
    ulgm_ulgm_q = chol_ulgm_ulgm_quench_vs_lgm0(lgm)
    ulgy_ulgy_q = chol_ulgy_ulgy_quench_vs_lgm0(lgm)
    ul_ul_q = chol_ul_ul_quench_vs_lgm0(lgm)
    utau_utau_q = chol_utau_utau_quench_vs_lgm0(lgm)
    uqt_uqt_q = chol_uqt_uqt_quench_vs_lgm0(lgm)
    uqs_uqs_q = chol_uqs_uqs_quench_vs_lgm0(lgm)
    udrop_udrop_q = chol_udrop_udrop_quench_vs_lgm0(lgm)
    urej_urej_q = chol_urej_urej_quench_vs_lgm0(lgm)
    ulgy_ulgm_q = chol_ulgy_ulgm_quench_vs_lgm0(lgm)
    ul_ulgm_q = chol_ul_ulgm_quench_vs_lgm0(lgm)
    ul_ulgy_q = chol_ul_ulgy_quench_vs_lgm0(lgm)
    utau_ulgm_q = chol_utau_ulgm_quench_vs_lgm0(lgm)
    utau_ulgy_q = chol_utau_ulgy_quench_vs_lgm0(lgm)
    utau_ul_q = chol_utau_ul_quench_vs_lgm0(lgm)
    uqt_ulgm_q = chol_uqt_ulgm_quench_vs_lgm0(lgm)
    uqt_ulgy_q = chol_uqt_ulgy_quench_vs_lgm0(lgm)
    uqt_ul_q = chol_uqt_ul_quench_vs_lgm0(lgm)
    uqt_utau_q = chol_uqt_utau_quench_vs_lgm0(lgm)
    uqs_ulgm_q = chol_uqs_ulgm_quench_vs_lgm0(lgm)
    uqs_ulgy_q = chol_uqs_ulgy_quench_vs_lgm0(lgm)
    uqs_ul_q = chol_uqs_ul_quench_vs_lgm0(lgm)
    uqs_utau_q = chol_uqs_utau_quench_vs_lgm0(lgm)
    uqs_uqt_q = chol_uqs_uqt_quench_vs_lgm0(lgm)
    udrop_ulgm_q = chol_udrop_ulgm_quench_vs_lgm0(lgm)
    udrop_ulgy_q = chol_udrop_ulgy_quench_vs_lgm0(lgm)
    udrop_ul_q = chol_udrop_ul_quench_vs_lgm0(lgm)
    udrop_utau_q = chol_udrop_utau_quench_vs_lgm0(lgm)
    udrop_uqt_q = chol_udrop_uqt_quench_vs_lgm0(lgm)
    udrop_uqs_q = chol_udrop_uqs_quench_vs_lgm0(lgm)
    urej_ulgm_q = chol_urej_ulgm_quench_vs_lgm0(lgm)
    urej_ulgy_q = chol_urej_ulgy_quench_vs_lgm0(lgm)
    urej_ul_q = chol_urej_ul_quench_vs_lgm0(lgm)
    urej_utau_q = chol_urej_utau_quench_vs_lgm0(lgm)
    urej_uqt_q = chol_urej_uqt_quench_vs_lgm0(lgm)
    urej_uqs_q = chol_urej_uqs_quench_vs_lgm0(lgm)
    urej_udrop_q = chol_urej_udrop_quench_vs_lgm0(lgm)

    all_params = (
        frac_quench,
        ulgm_q,
        ulgy_q,
        ul_q,
        utau_q,
        uqt_q,
        uqs_q,
        udrop_q,
        urej_q,
        ulgm_ulgm_q,
        ulgy_ulgy_q,
        ul_ul_q,
        utau_utau_q,
        uqt_uqt_q,
        uqs_uqs_q,
        udrop_udrop_q,
        urej_urej_q,
        ulgy_ulgm_q,
        ul_ulgm_q,
        ul_ulgy_q,
        utau_ulgm_q,
        utau_ulgy_q,
        utau_ul_q,
        uqt_ulgm_q,
        uqt_ulgy_q,
        uqt_ul_q,
        uqt_utau_q,
        uqs_ulgm_q,
        uqs_ulgy_q,
        uqs_ul_q,
        uqs_utau_q,
        uqs_uqt_q,
        udrop_ulgm_q,
        udrop_ulgy_q,
        udrop_ul_q,
        udrop_utau_q,
        udrop_uqt_q,
        udrop_uqs_q,
        urej_ulgm_q,
        urej_ulgy_q,
        urej_ul_q,
        urej_utau_q,
        urej_uqt_q,
        urej_uqs_q,
        urej_udrop_q,
    )
    return all_params


@jjit
def get_smah_means_and_covs_quench(
    logmp_arr,
    frac_quench_x0=DEFAULT_SFH_PDF_QUENCH_PARAMS["frac_quench_x0"],
    frac_quench_k=DEFAULT_SFH_PDF_QUENCH_PARAMS["frac_quench_k"],
    frac_quench_ylo=DEFAULT_SFH_PDF_QUENCH_PARAMS["frac_quench_ylo"],
    frac_quench_yhi=DEFAULT_SFH_PDF_QUENCH_PARAMS["frac_quench_yhi"],
    mean_ulgm_quench_ylo=DEFAULT_SFH_PDF_QUENCH_PARAMS["mean_ulgm_quench_ylo"],
    mean_ulgm_quench_yhi=DEFAULT_SFH_PDF_QUENCH_PARAMS["mean_ulgm_quench_yhi"],
    mean_ulgy_quench_ylo=DEFAULT_SFH_PDF_QUENCH_PARAMS["mean_ulgy_quench_ylo"],
    mean_ulgy_quench_yhi=DEFAULT_SFH_PDF_QUENCH_PARAMS["mean_ulgy_quench_yhi"],
    mean_ul_quench_ylo=DEFAULT_SFH_PDF_QUENCH_PARAMS["mean_ul_quench_ylo"],
    mean_ul_quench_yhi=DEFAULT_SFH_PDF_QUENCH_PARAMS["mean_ul_quench_yhi"],
    mean_utau_quench_ylo=DEFAULT_SFH_PDF_QUENCH_PARAMS["mean_utau_quench_ylo"],
    mean_utau_quench_yhi=DEFAULT_SFH_PDF_QUENCH_PARAMS["mean_utau_quench_yhi"],
    mean_uqt_quench_ylo=DEFAULT_SFH_PDF_QUENCH_PARAMS["mean_uqt_quench_ylo"],
    mean_uqt_quench_yhi=DEFAULT_SFH_PDF_QUENCH_PARAMS["mean_uqt_quench_yhi"],
    mean_uqs_quench_ylo=DEFAULT_SFH_PDF_QUENCH_PARAMS["mean_uqs_quench_ylo"],
    mean_uqs_quench_yhi=DEFAULT_SFH_PDF_QUENCH_PARAMS["mean_uqs_quench_yhi"],
    mean_udrop_quench_ylo=DEFAULT_SFH_PDF_QUENCH_PARAMS["mean_udrop_quench_ylo"],
    mean_udrop_quench_yhi=DEFAULT_SFH_PDF_QUENCH_PARAMS["mean_udrop_quench_yhi"],
    mean_urej_quench_ylo=DEFAULT_SFH_PDF_QUENCH_PARAMS["mean_urej_quench_ylo"],
    mean_urej_quench_yhi=DEFAULT_SFH_PDF_QUENCH_PARAMS["mean_urej_quench_yhi"],
    chol_ulgm_ulgm_quench_ylo=DEFAULT_SFH_PDF_QUENCH_PARAMS[
        "chol_ulgm_ulgm_quench_ylo"
    ],
    chol_ulgm_ulgm_quench_yhi=DEFAULT_SFH_PDF_QUENCH_PARAMS[
        "chol_ulgm_ulgm_quench_yhi"
    ],
    chol_ulgy_ulgy_quench_ylo=DEFAULT_SFH_PDF_QUENCH_PARAMS[
        "chol_ulgy_ulgy_quench_ylo"
    ],
    chol_ulgy_ulgy_quench_yhi=DEFAULT_SFH_PDF_QUENCH_PARAMS[
        "chol_ulgy_ulgy_quench_yhi"
    ],
    chol_ul_ul_quench_ylo=DEFAULT_SFH_PDF_QUENCH_PARAMS["chol_ul_ul_quench_ylo"],
    chol_ul_ul_quench_yhi=DEFAULT_SFH_PDF_QUENCH_PARAMS["chol_ul_ul_quench_yhi"],
    chol_utau_utau_quench_ylo=DEFAULT_SFH_PDF_QUENCH_PARAMS[
        "chol_utau_utau_quench_ylo"
    ],
    chol_utau_utau_quench_yhi=DEFAULT_SFH_PDF_QUENCH_PARAMS[
        "chol_utau_utau_quench_yhi"
    ],
    chol_uqt_uqt_quench_ylo=DEFAULT_SFH_PDF_QUENCH_PARAMS["chol_uqt_uqt_quench_ylo"],
    chol_uqt_uqt_quench_yhi=DEFAULT_SFH_PDF_QUENCH_PARAMS["chol_uqt_uqt_quench_yhi"],
    chol_uqs_uqs_quench_ylo=DEFAULT_SFH_PDF_QUENCH_PARAMS["chol_uqs_uqs_quench_ylo"],
    chol_uqs_uqs_quench_yhi=DEFAULT_SFH_PDF_QUENCH_PARAMS["chol_uqs_uqs_quench_yhi"],
    chol_udrop_udrop_quench_ylo=DEFAULT_SFH_PDF_QUENCH_PARAMS[
        "chol_udrop_udrop_quench_ylo"
    ],
    chol_udrop_udrop_quench_yhi=DEFAULT_SFH_PDF_QUENCH_PARAMS[
        "chol_udrop_udrop_quench_yhi"
    ],
    chol_urej_urej_quench_ylo=DEFAULT_SFH_PDF_QUENCH_PARAMS[
        "chol_urej_urej_quench_ylo"
    ],
    chol_urej_urej_quench_yhi=DEFAULT_SFH_PDF_QUENCH_PARAMS[
        "chol_urej_urej_quench_yhi"
    ],
    chol_ulgy_ulgm_quench_ylo=DEFAULT_SFH_PDF_QUENCH_PARAMS[
        "chol_ulgy_ulgm_quench_ylo"
    ],
    chol_ulgy_ulgm_quench_yhi=DEFAULT_SFH_PDF_QUENCH_PARAMS[
        "chol_ulgy_ulgm_quench_yhi"
    ],
    chol_ul_ulgm_quench_ylo=DEFAULT_SFH_PDF_QUENCH_PARAMS["chol_ul_ulgm_quench_ylo"],
    chol_ul_ulgm_quench_yhi=DEFAULT_SFH_PDF_QUENCH_PARAMS["chol_ul_ulgm_quench_yhi"],
    chol_ul_ulgy_quench_ylo=DEFAULT_SFH_PDF_QUENCH_PARAMS["chol_ul_ulgy_quench_ylo"],
    chol_ul_ulgy_quench_yhi=DEFAULT_SFH_PDF_QUENCH_PARAMS["chol_ul_ulgy_quench_yhi"],
    chol_utau_ulgm_quench_ylo=DEFAULT_SFH_PDF_QUENCH_PARAMS[
        "chol_utau_ulgm_quench_ylo"
    ],
    chol_utau_ulgm_quench_yhi=DEFAULT_SFH_PDF_QUENCH_PARAMS[
        "chol_utau_ulgm_quench_yhi"
    ],
    chol_utau_ulgy_quench_ylo=DEFAULT_SFH_PDF_QUENCH_PARAMS[
        "chol_utau_ulgy_quench_ylo"
    ],
    chol_utau_ulgy_quench_yhi=DEFAULT_SFH_PDF_QUENCH_PARAMS[
        "chol_utau_ulgy_quench_yhi"
    ],
    chol_utau_ul_quench_ylo=DEFAULT_SFH_PDF_QUENCH_PARAMS["chol_utau_ul_quench_ylo"],
    chol_utau_ul_quench_yhi=DEFAULT_SFH_PDF_QUENCH_PARAMS["chol_utau_ul_quench_yhi"],
    chol_uqt_ulgm_quench_ylo=DEFAULT_SFH_PDF_QUENCH_PARAMS["chol_uqt_ulgm_quench_ylo"],
    chol_uqt_ulgm_quench_yhi=DEFAULT_SFH_PDF_QUENCH_PARAMS["chol_uqt_ulgm_quench_yhi"],
    chol_uqt_ulgy_quench_ylo=DEFAULT_SFH_PDF_QUENCH_PARAMS["chol_uqt_ulgy_quench_ylo"],
    chol_uqt_ulgy_quench_yhi=DEFAULT_SFH_PDF_QUENCH_PARAMS["chol_uqt_ulgy_quench_yhi"],
    chol_uqt_ul_quench_ylo=DEFAULT_SFH_PDF_QUENCH_PARAMS["chol_uqt_ul_quench_ylo"],
    chol_uqt_ul_quench_yhi=DEFAULT_SFH_PDF_QUENCH_PARAMS["chol_uqt_ul_quench_yhi"],
    chol_uqt_utau_quench_ylo=DEFAULT_SFH_PDF_QUENCH_PARAMS["chol_uqt_utau_quench_ylo"],
    chol_uqt_utau_quench_yhi=DEFAULT_SFH_PDF_QUENCH_PARAMS["chol_uqt_utau_quench_yhi"],
    chol_uqs_ulgm_quench_ylo=DEFAULT_SFH_PDF_QUENCH_PARAMS["chol_uqs_ulgm_quench_ylo"],
    chol_uqs_ulgm_quench_yhi=DEFAULT_SFH_PDF_QUENCH_PARAMS["chol_uqs_ulgm_quench_yhi"],
    chol_uqs_ulgy_quench_ylo=DEFAULT_SFH_PDF_QUENCH_PARAMS["chol_uqs_ulgy_quench_ylo"],
    chol_uqs_ulgy_quench_yhi=DEFAULT_SFH_PDF_QUENCH_PARAMS["chol_uqs_ulgy_quench_yhi"],
    chol_uqs_ul_quench_ylo=DEFAULT_SFH_PDF_QUENCH_PARAMS["chol_uqs_ul_quench_ylo"],
    chol_uqs_ul_quench_yhi=DEFAULT_SFH_PDF_QUENCH_PARAMS["chol_uqs_ul_quench_yhi"],
    chol_uqs_utau_quench_ylo=DEFAULT_SFH_PDF_QUENCH_PARAMS["chol_uqs_utau_quench_ylo"],
    chol_uqs_utau_quench_yhi=DEFAULT_SFH_PDF_QUENCH_PARAMS["chol_uqs_utau_quench_yhi"],
    chol_uqs_uqt_quench_ylo=DEFAULT_SFH_PDF_QUENCH_PARAMS["chol_uqs_uqt_quench_ylo"],
    chol_uqs_uqt_quench_yhi=DEFAULT_SFH_PDF_QUENCH_PARAMS["chol_uqs_uqt_quench_yhi"],
    chol_udrop_ulgm_quench_ylo=DEFAULT_SFH_PDF_QUENCH_PARAMS[
        "chol_udrop_ulgm_quench_ylo"
    ],
    chol_udrop_ulgm_quench_yhi=DEFAULT_SFH_PDF_QUENCH_PARAMS[
        "chol_udrop_ulgm_quench_yhi"
    ],
    chol_udrop_ulgy_quench_ylo=DEFAULT_SFH_PDF_QUENCH_PARAMS[
        "chol_udrop_ulgy_quench_ylo"
    ],
    chol_udrop_ulgy_quench_yhi=DEFAULT_SFH_PDF_QUENCH_PARAMS[
        "chol_udrop_ulgy_quench_yhi"
    ],
    chol_udrop_ul_quench_ylo=DEFAULT_SFH_PDF_QUENCH_PARAMS["chol_udrop_ul_quench_ylo"],
    chol_udrop_ul_quench_yhi=DEFAULT_SFH_PDF_QUENCH_PARAMS["chol_udrop_ul_quench_yhi"],
    chol_udrop_utau_quench_ylo=DEFAULT_SFH_PDF_QUENCH_PARAMS[
        "chol_udrop_utau_quench_ylo"
    ],
    chol_udrop_utau_quench_yhi=DEFAULT_SFH_PDF_QUENCH_PARAMS[
        "chol_udrop_utau_quench_yhi"
    ],
    chol_udrop_uqt_quench_ylo=DEFAULT_SFH_PDF_QUENCH_PARAMS[
        "chol_udrop_uqt_quench_ylo"
    ],
    chol_udrop_uqt_quench_yhi=DEFAULT_SFH_PDF_QUENCH_PARAMS[
        "chol_udrop_uqt_quench_yhi"
    ],
    chol_udrop_uqs_quench_ylo=DEFAULT_SFH_PDF_QUENCH_PARAMS[
        "chol_udrop_uqs_quench_ylo"
    ],
    chol_udrop_uqs_quench_yhi=DEFAULT_SFH_PDF_QUENCH_PARAMS[
        "chol_udrop_uqs_quench_yhi"
    ],
    chol_urej_ulgm_quench_ylo=DEFAULT_SFH_PDF_QUENCH_PARAMS[
        "chol_urej_ulgm_quench_ylo"
    ],
    chol_urej_ulgm_quench_yhi=DEFAULT_SFH_PDF_QUENCH_PARAMS[
        "chol_urej_ulgm_quench_yhi"
    ],
    chol_urej_ulgy_quench_ylo=DEFAULT_SFH_PDF_QUENCH_PARAMS[
        "chol_urej_ulgy_quench_ylo"
    ],
    chol_urej_ulgy_quench_yhi=DEFAULT_SFH_PDF_QUENCH_PARAMS[
        "chol_urej_ulgy_quench_yhi"
    ],
    chol_urej_ul_quench_ylo=DEFAULT_SFH_PDF_QUENCH_PARAMS["chol_urej_ul_quench_ylo"],
    chol_urej_ul_quench_yhi=DEFAULT_SFH_PDF_QUENCH_PARAMS["chol_urej_ul_quench_yhi"],
    chol_urej_utau_quench_ylo=DEFAULT_SFH_PDF_QUENCH_PARAMS[
        "chol_urej_utau_quench_ylo"
    ],
    chol_urej_utau_quench_yhi=DEFAULT_SFH_PDF_QUENCH_PARAMS[
        "chol_urej_utau_quench_yhi"
    ],
    chol_urej_uqt_quench_ylo=DEFAULT_SFH_PDF_QUENCH_PARAMS["chol_urej_uqt_quench_ylo"],
    chol_urej_uqt_quench_yhi=DEFAULT_SFH_PDF_QUENCH_PARAMS["chol_urej_uqt_quench_yhi"],
    chol_urej_uqs_quench_ylo=DEFAULT_SFH_PDF_QUENCH_PARAMS["chol_urej_uqs_quench_ylo"],
    chol_urej_uqs_quench_yhi=DEFAULT_SFH_PDF_QUENCH_PARAMS["chol_urej_uqs_quench_yhi"],
    chol_urej_udrop_quench_ylo=DEFAULT_SFH_PDF_QUENCH_PARAMS[
        "chol_urej_udrop_quench_ylo"
    ],
    chol_urej_udrop_quench_yhi=DEFAULT_SFH_PDF_QUENCH_PARAMS[
        "chol_urej_udrop_quench_yhi"
    ],
):

    frac_quench = frac_quench_vs_lgm0(
        logmp_arr, frac_quench_x0, frac_quench_k, frac_quench_ylo, frac_quench_yhi
    )
    _res = _get_mean_smah_params_quench(
        logmp_arr,
        mean_ulgm_quench_ylo,
        mean_ulgm_quench_yhi,
        mean_ulgy_quench_ylo,
        mean_ulgy_quench_yhi,
        mean_ul_quench_ylo,
        mean_ul_quench_yhi,
        mean_utau_quench_ylo,
        mean_utau_quench_yhi,
        mean_uqt_quench_ylo,
        mean_uqt_quench_yhi,
        mean_uqs_quench_ylo,
        mean_uqs_quench_yhi,
        mean_udrop_quench_ylo,
        mean_udrop_quench_yhi,
        mean_urej_quench_ylo,
        mean_urej_quench_yhi,
    )

    means_quench = jnp.array(_res).T

    covs_quench = _get_covs_quench(
        logmp_arr,
        chol_ulgm_ulgm_quench_ylo,
        chol_ulgm_ulgm_quench_yhi,
        chol_ulgy_ulgy_quench_ylo,
        chol_ulgy_ulgy_quench_yhi,
        chol_ul_ul_quench_ylo,
        chol_ul_ul_quench_yhi,
        chol_utau_utau_quench_ylo,
        chol_utau_utau_quench_yhi,
        chol_uqt_uqt_quench_ylo,
        chol_uqt_uqt_quench_yhi,
        chol_uqs_uqs_quench_ylo,
        chol_uqs_uqs_quench_yhi,
        chol_udrop_udrop_quench_ylo,
        chol_udrop_udrop_quench_yhi,
        chol_urej_urej_quench_ylo,
        chol_urej_urej_quench_yhi,
        chol_ulgy_ulgm_quench_ylo,
        chol_ulgy_ulgm_quench_yhi,
        chol_ul_ulgm_quench_ylo,
        chol_ul_ulgm_quench_yhi,
        chol_ul_ulgy_quench_ylo,
        chol_ul_ulgy_quench_yhi,
        chol_utau_ulgm_quench_ylo,
        chol_utau_ulgm_quench_yhi,
        chol_utau_ulgy_quench_ylo,
        chol_utau_ulgy_quench_yhi,
        chol_utau_ul_quench_ylo,
        chol_utau_ul_quench_yhi,
        chol_uqt_ulgm_quench_ylo,
        chol_uqt_ulgm_quench_yhi,
        chol_uqt_ulgy_quench_ylo,
        chol_uqt_ulgy_quench_yhi,
        chol_uqt_ul_quench_ylo,
        chol_uqt_ul_quench_yhi,
        chol_uqt_utau_quench_ylo,
        chol_uqt_utau_quench_yhi,
        chol_uqs_ulgm_quench_ylo,
        chol_uqs_ulgm_quench_yhi,
        chol_uqs_ulgy_quench_ylo,
        chol_uqs_ulgy_quench_yhi,
        chol_uqs_ul_quench_ylo,
        chol_uqs_ul_quench_yhi,
        chol_uqs_utau_quench_ylo,
        chol_uqs_utau_quench_yhi,
        chol_uqs_uqt_quench_ylo,
        chol_uqs_uqt_quench_yhi,
        chol_udrop_ulgm_quench_ylo,
        chol_udrop_ulgm_quench_yhi,
        chol_udrop_ulgy_quench_ylo,
        chol_udrop_ulgy_quench_yhi,
        chol_udrop_ul_quench_ylo,
        chol_udrop_ul_quench_yhi,
        chol_udrop_utau_quench_ylo,
        chol_udrop_utau_quench_yhi,
        chol_udrop_uqt_quench_ylo,
        chol_udrop_uqt_quench_yhi,
        chol_udrop_uqs_quench_ylo,
        chol_udrop_uqs_quench_yhi,
        chol_urej_ulgm_quench_ylo,
        chol_urej_ulgm_quench_yhi,
        chol_urej_ulgy_quench_ylo,
        chol_urej_ulgy_quench_yhi,
        chol_urej_ul_quench_ylo,
        chol_urej_ul_quench_yhi,
        chol_urej_utau_quench_ylo,
        chol_urej_utau_quench_yhi,
        chol_urej_uqt_quench_ylo,
        chol_urej_uqt_quench_yhi,
        chol_urej_uqs_quench_ylo,
        chol_urej_uqs_quench_yhi,
        chol_urej_udrop_quench_ylo,
        chol_urej_udrop_quench_yhi,
    )
    return frac_quench, means_quench, covs_quench


@jjit
def _get_mean_smah_params_quench(
    lgm,
    mean_ulgm_quench_ylo=DEFAULT_SFH_PDF_QUENCH_PARAMS["mean_ulgm_quench_ylo"],
    mean_ulgm_quench_yhi=DEFAULT_SFH_PDF_QUENCH_PARAMS["mean_ulgm_quench_yhi"],
    mean_ulgy_quench_ylo=DEFAULT_SFH_PDF_QUENCH_PARAMS["mean_ulgy_quench_ylo"],
    mean_ulgy_quench_yhi=DEFAULT_SFH_PDF_QUENCH_PARAMS["mean_ulgy_quench_yhi"],
    mean_ul_quench_ylo=DEFAULT_SFH_PDF_QUENCH_PARAMS["mean_ul_quench_ylo"],
    mean_ul_quench_yhi=DEFAULT_SFH_PDF_QUENCH_PARAMS["mean_ul_quench_yhi"],
    mean_utau_quench_ylo=DEFAULT_SFH_PDF_QUENCH_PARAMS["mean_utau_quench_ylo"],
    mean_utau_quench_yhi=DEFAULT_SFH_PDF_QUENCH_PARAMS["mean_utau_quench_yhi"],
    mean_uqt_quench_ylo=DEFAULT_SFH_PDF_QUENCH_PARAMS["mean_uqt_quench_ylo"],
    mean_uqt_quench_yhi=DEFAULT_SFH_PDF_QUENCH_PARAMS["mean_uqt_quench_yhi"],
    mean_uqs_quench_ylo=DEFAULT_SFH_PDF_QUENCH_PARAMS["mean_uqs_quench_ylo"],
    mean_uqs_quench_yhi=DEFAULT_SFH_PDF_QUENCH_PARAMS["mean_uqs_quench_yhi"],
    mean_udrop_quench_ylo=DEFAULT_SFH_PDF_QUENCH_PARAMS["mean_udrop_quench_ylo"],
    mean_udrop_quench_yhi=DEFAULT_SFH_PDF_QUENCH_PARAMS["mean_udrop_quench_yhi"],
    mean_urej_quench_ylo=DEFAULT_SFH_PDF_QUENCH_PARAMS["mean_urej_quench_ylo"],
    mean_urej_quench_yhi=DEFAULT_SFH_PDF_QUENCH_PARAMS["mean_urej_quench_yhi"],
):
    ulgm = mean_ulgm_quench_vs_lgm0(lgm, mean_ulgm_quench_ylo, mean_ulgm_quench_yhi)
    ulgy = mean_ulgy_quench_vs_lgm0(lgm, mean_ulgy_quench_ylo, mean_ulgy_quench_yhi)
    ul = mean_ul_quench_vs_lgm0(lgm, mean_ul_quench_ylo, mean_ul_quench_yhi)
    utau = mean_utau_quench_vs_lgm0(lgm, mean_utau_quench_ylo, mean_utau_quench_yhi)
    uqt = mean_uqt_quench_vs_lgm0(lgm, mean_uqt_quench_ylo, mean_uqt_quench_yhi)
    uqs = mean_uqs_quench_vs_lgm0(lgm, mean_uqs_quench_ylo, mean_uqs_quench_yhi)
    udrop = mean_udrop_quench_vs_lgm0(lgm, mean_udrop_quench_ylo, mean_udrop_quench_yhi)
    urej = mean_urej_quench_vs_lgm0(lgm, mean_urej_quench_ylo, mean_urej_quench_yhi)
    return ulgm, ulgy, ul, utau, uqt, uqs, udrop, urej


@jjit
def _get_covs_quench(
    lgmp_arr,
    chol_ulgm_ulgm_quench_ylo=DEFAULT_SFH_PDF_QUENCH_PARAMS[
        "chol_ulgm_ulgm_quench_ylo"
    ],
    chol_ulgm_ulgm_quench_yhi=DEFAULT_SFH_PDF_QUENCH_PARAMS[
        "chol_ulgm_ulgm_quench_yhi"
    ],
    chol_ulgy_ulgy_quench_ylo=DEFAULT_SFH_PDF_QUENCH_PARAMS[
        "chol_ulgy_ulgy_quench_ylo"
    ],
    chol_ulgy_ulgy_quench_yhi=DEFAULT_SFH_PDF_QUENCH_PARAMS[
        "chol_ulgy_ulgy_quench_yhi"
    ],
    chol_ul_ul_quench_ylo=DEFAULT_SFH_PDF_QUENCH_PARAMS["chol_ul_ul_quench_ylo"],
    chol_ul_ul_quench_yhi=DEFAULT_SFH_PDF_QUENCH_PARAMS["chol_ul_ul_quench_yhi"],
    chol_utau_utau_quench_ylo=DEFAULT_SFH_PDF_QUENCH_PARAMS[
        "chol_utau_utau_quench_ylo"
    ],
    chol_utau_utau_quench_yhi=DEFAULT_SFH_PDF_QUENCH_PARAMS[
        "chol_utau_utau_quench_yhi"
    ],
    chol_uqt_uqt_quench_ylo=DEFAULT_SFH_PDF_QUENCH_PARAMS["chol_uqt_uqt_quench_ylo"],
    chol_uqt_uqt_quench_yhi=DEFAULT_SFH_PDF_QUENCH_PARAMS["chol_uqt_uqt_quench_yhi"],
    chol_uqs_uqs_quench_ylo=DEFAULT_SFH_PDF_QUENCH_PARAMS["chol_uqs_uqs_quench_ylo"],
    chol_uqs_uqs_quench_yhi=DEFAULT_SFH_PDF_QUENCH_PARAMS["chol_uqs_uqs_quench_yhi"],
    chol_udrop_udrop_quench_ylo=DEFAULT_SFH_PDF_QUENCH_PARAMS[
        "chol_udrop_udrop_quench_ylo"
    ],
    chol_udrop_udrop_quench_yhi=DEFAULT_SFH_PDF_QUENCH_PARAMS[
        "chol_udrop_udrop_quench_yhi"
    ],
    chol_urej_urej_quench_ylo=DEFAULT_SFH_PDF_QUENCH_PARAMS[
        "chol_urej_urej_quench_ylo"
    ],
    chol_urej_urej_quench_yhi=DEFAULT_SFH_PDF_QUENCH_PARAMS[
        "chol_urej_urej_quench_yhi"
    ],
    chol_ulgy_ulgm_quench_ylo=DEFAULT_SFH_PDF_QUENCH_PARAMS[
        "chol_ulgy_ulgm_quench_ylo"
    ],
    chol_ulgy_ulgm_quench_yhi=DEFAULT_SFH_PDF_QUENCH_PARAMS[
        "chol_ulgy_ulgm_quench_yhi"
    ],
    chol_ul_ulgm_quench_ylo=DEFAULT_SFH_PDF_QUENCH_PARAMS["chol_ul_ulgm_quench_ylo"],
    chol_ul_ulgm_quench_yhi=DEFAULT_SFH_PDF_QUENCH_PARAMS["chol_ul_ulgm_quench_yhi"],
    chol_ul_ulgy_quench_ylo=DEFAULT_SFH_PDF_QUENCH_PARAMS["chol_ul_ulgy_quench_ylo"],
    chol_ul_ulgy_quench_yhi=DEFAULT_SFH_PDF_QUENCH_PARAMS["chol_ul_ulgy_quench_yhi"],
    chol_utau_ulgm_quench_ylo=DEFAULT_SFH_PDF_QUENCH_PARAMS[
        "chol_utau_ulgm_quench_ylo"
    ],
    chol_utau_ulgm_quench_yhi=DEFAULT_SFH_PDF_QUENCH_PARAMS[
        "chol_utau_ulgm_quench_yhi"
    ],
    chol_utau_ulgy_quench_ylo=DEFAULT_SFH_PDF_QUENCH_PARAMS[
        "chol_utau_ulgy_quench_ylo"
    ],
    chol_utau_ulgy_quench_yhi=DEFAULT_SFH_PDF_QUENCH_PARAMS[
        "chol_utau_ulgy_quench_yhi"
    ],
    chol_utau_ul_quench_ylo=DEFAULT_SFH_PDF_QUENCH_PARAMS["chol_utau_ul_quench_ylo"],
    chol_utau_ul_quench_yhi=DEFAULT_SFH_PDF_QUENCH_PARAMS["chol_utau_ul_quench_yhi"],
    chol_uqt_ulgm_quench_ylo=DEFAULT_SFH_PDF_QUENCH_PARAMS["chol_uqt_ulgm_quench_ylo"],
    chol_uqt_ulgm_quench_yhi=DEFAULT_SFH_PDF_QUENCH_PARAMS["chol_uqt_ulgm_quench_yhi"],
    chol_uqt_ulgy_quench_ylo=DEFAULT_SFH_PDF_QUENCH_PARAMS["chol_uqt_ulgy_quench_ylo"],
    chol_uqt_ulgy_quench_yhi=DEFAULT_SFH_PDF_QUENCH_PARAMS["chol_uqt_ulgy_quench_yhi"],
    chol_uqt_ul_quench_ylo=DEFAULT_SFH_PDF_QUENCH_PARAMS["chol_uqt_ul_quench_ylo"],
    chol_uqt_ul_quench_yhi=DEFAULT_SFH_PDF_QUENCH_PARAMS["chol_uqt_ul_quench_yhi"],
    chol_uqt_utau_quench_ylo=DEFAULT_SFH_PDF_QUENCH_PARAMS["chol_uqt_utau_quench_ylo"],
    chol_uqt_utau_quench_yhi=DEFAULT_SFH_PDF_QUENCH_PARAMS["chol_uqt_utau_quench_yhi"],
    chol_uqs_ulgm_quench_ylo=DEFAULT_SFH_PDF_QUENCH_PARAMS["chol_uqs_ulgm_quench_ylo"],
    chol_uqs_ulgm_quench_yhi=DEFAULT_SFH_PDF_QUENCH_PARAMS["chol_uqs_ulgm_quench_yhi"],
    chol_uqs_ulgy_quench_ylo=DEFAULT_SFH_PDF_QUENCH_PARAMS["chol_uqs_ulgy_quench_ylo"],
    chol_uqs_ulgy_quench_yhi=DEFAULT_SFH_PDF_QUENCH_PARAMS["chol_uqs_ulgy_quench_yhi"],
    chol_uqs_ul_quench_ylo=DEFAULT_SFH_PDF_QUENCH_PARAMS["chol_uqs_ul_quench_ylo"],
    chol_uqs_ul_quench_yhi=DEFAULT_SFH_PDF_QUENCH_PARAMS["chol_uqs_ul_quench_yhi"],
    chol_uqs_utau_quench_ylo=DEFAULT_SFH_PDF_QUENCH_PARAMS["chol_uqs_utau_quench_ylo"],
    chol_uqs_utau_quench_yhi=DEFAULT_SFH_PDF_QUENCH_PARAMS["chol_uqs_utau_quench_yhi"],
    chol_uqs_uqt_quench_ylo=DEFAULT_SFH_PDF_QUENCH_PARAMS["chol_uqs_uqt_quench_ylo"],
    chol_uqs_uqt_quench_yhi=DEFAULT_SFH_PDF_QUENCH_PARAMS["chol_uqs_uqt_quench_yhi"],
    chol_udrop_ulgm_quench_ylo=DEFAULT_SFH_PDF_QUENCH_PARAMS[
        "chol_udrop_ulgm_quench_ylo"
    ],
    chol_udrop_ulgm_quench_yhi=DEFAULT_SFH_PDF_QUENCH_PARAMS[
        "chol_udrop_ulgm_quench_yhi"
    ],
    chol_udrop_ulgy_quench_ylo=DEFAULT_SFH_PDF_QUENCH_PARAMS[
        "chol_udrop_ulgy_quench_ylo"
    ],
    chol_udrop_ulgy_quench_yhi=DEFAULT_SFH_PDF_QUENCH_PARAMS[
        "chol_udrop_ulgy_quench_yhi"
    ],
    chol_udrop_ul_quench_ylo=DEFAULT_SFH_PDF_QUENCH_PARAMS["chol_udrop_ul_quench_ylo"],
    chol_udrop_ul_quench_yhi=DEFAULT_SFH_PDF_QUENCH_PARAMS["chol_udrop_ul_quench_yhi"],
    chol_udrop_utau_quench_ylo=DEFAULT_SFH_PDF_QUENCH_PARAMS[
        "chol_udrop_utau_quench_ylo"
    ],
    chol_udrop_utau_quench_yhi=DEFAULT_SFH_PDF_QUENCH_PARAMS[
        "chol_udrop_utau_quench_yhi"
    ],
    chol_udrop_uqt_quench_ylo=DEFAULT_SFH_PDF_QUENCH_PARAMS[
        "chol_udrop_uqt_quench_ylo"
    ],
    chol_udrop_uqt_quench_yhi=DEFAULT_SFH_PDF_QUENCH_PARAMS[
        "chol_udrop_uqt_quench_yhi"
    ],
    chol_udrop_uqs_quench_ylo=DEFAULT_SFH_PDF_QUENCH_PARAMS[
        "chol_udrop_uqs_quench_ylo"
    ],
    chol_udrop_uqs_quench_yhi=DEFAULT_SFH_PDF_QUENCH_PARAMS[
        "chol_udrop_uqs_quench_yhi"
    ],
    chol_urej_ulgm_quench_ylo=DEFAULT_SFH_PDF_QUENCH_PARAMS[
        "chol_urej_ulgm_quench_ylo"
    ],
    chol_urej_ulgm_quench_yhi=DEFAULT_SFH_PDF_QUENCH_PARAMS[
        "chol_urej_ulgm_quench_yhi"
    ],
    chol_urej_ulgy_quench_ylo=DEFAULT_SFH_PDF_QUENCH_PARAMS[
        "chol_urej_ulgy_quench_ylo"
    ],
    chol_urej_ulgy_quench_yhi=DEFAULT_SFH_PDF_QUENCH_PARAMS[
        "chol_urej_ulgy_quench_yhi"
    ],
    chol_urej_ul_quench_ylo=DEFAULT_SFH_PDF_QUENCH_PARAMS["chol_urej_ul_quench_ylo"],
    chol_urej_ul_quench_yhi=DEFAULT_SFH_PDF_QUENCH_PARAMS["chol_urej_ul_quench_yhi"],
    chol_urej_utau_quench_ylo=DEFAULT_SFH_PDF_QUENCH_PARAMS[
        "chol_urej_utau_quench_ylo"
    ],
    chol_urej_utau_quench_yhi=DEFAULT_SFH_PDF_QUENCH_PARAMS[
        "chol_urej_utau_quench_yhi"
    ],
    chol_urej_uqt_quench_ylo=DEFAULT_SFH_PDF_QUENCH_PARAMS["chol_urej_uqt_quench_ylo"],
    chol_urej_uqt_quench_yhi=DEFAULT_SFH_PDF_QUENCH_PARAMS["chol_urej_uqt_quench_yhi"],
    chol_urej_uqs_quench_ylo=DEFAULT_SFH_PDF_QUENCH_PARAMS["chol_urej_uqs_quench_ylo"],
    chol_urej_uqs_quench_yhi=DEFAULT_SFH_PDF_QUENCH_PARAMS["chol_urej_uqs_quench_yhi"],
    chol_urej_udrop_quench_ylo=DEFAULT_SFH_PDF_QUENCH_PARAMS[
        "chol_urej_udrop_quench_ylo"
    ],
    chol_urej_udrop_quench_yhi=DEFAULT_SFH_PDF_QUENCH_PARAMS[
        "chol_urej_udrop_quench_yhi"
    ],
):

    _res = _get_chol_params_quench(
        lgmp_arr,
        chol_ulgm_ulgm_quench_ylo,
        chol_ulgm_ulgm_quench_yhi,
        chol_ulgy_ulgy_quench_ylo,
        chol_ulgy_ulgy_quench_yhi,
        chol_ul_ul_quench_ylo,
        chol_ul_ul_quench_yhi,
        chol_utau_utau_quench_ylo,
        chol_utau_utau_quench_yhi,
        chol_uqt_uqt_quench_ylo,
        chol_uqt_uqt_quench_yhi,
        chol_uqs_uqs_quench_ylo,
        chol_uqs_uqs_quench_yhi,
        chol_udrop_udrop_quench_ylo,
        chol_udrop_udrop_quench_yhi,
        chol_urej_urej_quench_ylo,
        chol_urej_urej_quench_yhi,
        chol_ulgy_ulgm_quench_ylo,
        chol_ulgy_ulgm_quench_yhi,
        chol_ul_ulgm_quench_ylo,
        chol_ul_ulgm_quench_yhi,
        chol_ul_ulgy_quench_ylo,
        chol_ul_ulgy_quench_yhi,
        chol_utau_ulgm_quench_ylo,
        chol_utau_ulgm_quench_yhi,
        chol_utau_ulgy_quench_ylo,
        chol_utau_ulgy_quench_yhi,
        chol_utau_ul_quench_ylo,
        chol_utau_ul_quench_yhi,
        chol_uqt_ulgm_quench_ylo,
        chol_uqt_ulgm_quench_yhi,
        chol_uqt_ulgy_quench_ylo,
        chol_uqt_ulgy_quench_yhi,
        chol_uqt_ul_quench_ylo,
        chol_uqt_ul_quench_yhi,
        chol_uqt_utau_quench_ylo,
        chol_uqt_utau_quench_yhi,
        chol_uqs_ulgm_quench_ylo,
        chol_uqs_ulgm_quench_yhi,
        chol_uqs_ulgy_quench_ylo,
        chol_uqs_ulgy_quench_yhi,
        chol_uqs_ul_quench_ylo,
        chol_uqs_ul_quench_yhi,
        chol_uqs_utau_quench_ylo,
        chol_uqs_utau_quench_yhi,
        chol_uqs_uqt_quench_ylo,
        chol_uqs_uqt_quench_yhi,
        chol_udrop_ulgm_quench_ylo,
        chol_udrop_ulgm_quench_yhi,
        chol_udrop_ulgy_quench_ylo,
        chol_udrop_ulgy_quench_yhi,
        chol_udrop_ul_quench_ylo,
        chol_udrop_ul_quench_yhi,
        chol_udrop_utau_quench_ylo,
        chol_udrop_utau_quench_yhi,
        chol_udrop_uqt_quench_ylo,
        chol_udrop_uqt_quench_yhi,
        chol_udrop_uqs_quench_ylo,
        chol_udrop_uqs_quench_yhi,
        chol_urej_ulgm_quench_ylo,
        chol_urej_ulgm_quench_yhi,
        chol_urej_ulgy_quench_ylo,
        chol_urej_ulgy_quench_yhi,
        chol_urej_ul_quench_ylo,
        chol_urej_ul_quench_yhi,
        chol_urej_utau_quench_ylo,
        chol_urej_utau_quench_yhi,
        chol_urej_uqt_quench_ylo,
        chol_urej_uqt_quench_yhi,
        chol_urej_uqs_quench_ylo,
        chol_urej_uqs_quench_yhi,
        chol_urej_udrop_quench_ylo,
        chol_urej_udrop_quench_yhi,
    )
    return _get_cov_vmap(*_res)


@jjit
def _get_chol_params_quench(
    lgm,
    chol_ulgm_ulgm_quench_ylo=DEFAULT_SFH_PDF_QUENCH_PARAMS[
        "chol_ulgm_ulgm_quench_ylo"
    ],
    chol_ulgm_ulgm_quench_yhi=DEFAULT_SFH_PDF_QUENCH_PARAMS[
        "chol_ulgm_ulgm_quench_yhi"
    ],
    chol_ulgy_ulgy_quench_ylo=DEFAULT_SFH_PDF_QUENCH_PARAMS[
        "chol_ulgy_ulgy_quench_ylo"
    ],
    chol_ulgy_ulgy_quench_yhi=DEFAULT_SFH_PDF_QUENCH_PARAMS[
        "chol_ulgy_ulgy_quench_yhi"
    ],
    chol_ul_ul_quench_ylo=DEFAULT_SFH_PDF_QUENCH_PARAMS["chol_ul_ul_quench_ylo"],
    chol_ul_ul_quench_yhi=DEFAULT_SFH_PDF_QUENCH_PARAMS["chol_ul_ul_quench_yhi"],
    chol_utau_utau_quench_ylo=DEFAULT_SFH_PDF_QUENCH_PARAMS[
        "chol_utau_utau_quench_ylo"
    ],
    chol_utau_utau_quench_yhi=DEFAULT_SFH_PDF_QUENCH_PARAMS[
        "chol_utau_utau_quench_yhi"
    ],
    chol_uqt_uqt_quench_ylo=DEFAULT_SFH_PDF_QUENCH_PARAMS["chol_uqt_uqt_quench_ylo"],
    chol_uqt_uqt_quench_yhi=DEFAULT_SFH_PDF_QUENCH_PARAMS["chol_uqt_uqt_quench_yhi"],
    chol_uqs_uqs_quench_ylo=DEFAULT_SFH_PDF_QUENCH_PARAMS["chol_uqs_uqs_quench_ylo"],
    chol_uqs_uqs_quench_yhi=DEFAULT_SFH_PDF_QUENCH_PARAMS["chol_uqs_uqs_quench_yhi"],
    chol_udrop_udrop_quench_ylo=DEFAULT_SFH_PDF_QUENCH_PARAMS[
        "chol_udrop_udrop_quench_ylo"
    ],
    chol_udrop_udrop_quench_yhi=DEFAULT_SFH_PDF_QUENCH_PARAMS[
        "chol_udrop_udrop_quench_yhi"
    ],
    chol_urej_urej_quench_ylo=DEFAULT_SFH_PDF_QUENCH_PARAMS[
        "chol_urej_urej_quench_ylo"
    ],
    chol_urej_urej_quench_yhi=DEFAULT_SFH_PDF_QUENCH_PARAMS[
        "chol_urej_urej_quench_yhi"
    ],
    chol_ulgy_ulgm_quench_ylo=DEFAULT_SFH_PDF_QUENCH_PARAMS[
        "chol_ulgy_ulgm_quench_ylo"
    ],
    chol_ulgy_ulgm_quench_yhi=DEFAULT_SFH_PDF_QUENCH_PARAMS[
        "chol_ulgy_ulgm_quench_yhi"
    ],
    chol_ul_ulgm_quench_ylo=DEFAULT_SFH_PDF_QUENCH_PARAMS["chol_ul_ulgm_quench_ylo"],
    chol_ul_ulgm_quench_yhi=DEFAULT_SFH_PDF_QUENCH_PARAMS["chol_ul_ulgm_quench_yhi"],
    chol_ul_ulgy_quench_ylo=DEFAULT_SFH_PDF_QUENCH_PARAMS["chol_ul_ulgy_quench_ylo"],
    chol_ul_ulgy_quench_yhi=DEFAULT_SFH_PDF_QUENCH_PARAMS["chol_ul_ulgy_quench_yhi"],
    chol_utau_ulgm_quench_ylo=DEFAULT_SFH_PDF_QUENCH_PARAMS[
        "chol_utau_ulgm_quench_ylo"
    ],
    chol_utau_ulgm_quench_yhi=DEFAULT_SFH_PDF_QUENCH_PARAMS[
        "chol_utau_ulgm_quench_yhi"
    ],
    chol_utau_ulgy_quench_ylo=DEFAULT_SFH_PDF_QUENCH_PARAMS[
        "chol_utau_ulgy_quench_ylo"
    ],
    chol_utau_ulgy_quench_yhi=DEFAULT_SFH_PDF_QUENCH_PARAMS[
        "chol_utau_ulgy_quench_yhi"
    ],
    chol_utau_ul_quench_ylo=DEFAULT_SFH_PDF_QUENCH_PARAMS["chol_utau_ul_quench_ylo"],
    chol_utau_ul_quench_yhi=DEFAULT_SFH_PDF_QUENCH_PARAMS["chol_utau_ul_quench_yhi"],
    chol_uqt_ulgm_quench_ylo=DEFAULT_SFH_PDF_QUENCH_PARAMS["chol_uqt_ulgm_quench_ylo"],
    chol_uqt_ulgm_quench_yhi=DEFAULT_SFH_PDF_QUENCH_PARAMS["chol_uqt_ulgm_quench_yhi"],
    chol_uqt_ulgy_quench_ylo=DEFAULT_SFH_PDF_QUENCH_PARAMS["chol_uqt_ulgy_quench_ylo"],
    chol_uqt_ulgy_quench_yhi=DEFAULT_SFH_PDF_QUENCH_PARAMS["chol_uqt_ulgy_quench_yhi"],
    chol_uqt_ul_quench_ylo=DEFAULT_SFH_PDF_QUENCH_PARAMS["chol_uqt_ul_quench_ylo"],
    chol_uqt_ul_quench_yhi=DEFAULT_SFH_PDF_QUENCH_PARAMS["chol_uqt_ul_quench_yhi"],
    chol_uqt_utau_quench_ylo=DEFAULT_SFH_PDF_QUENCH_PARAMS["chol_uqt_utau_quench_ylo"],
    chol_uqt_utau_quench_yhi=DEFAULT_SFH_PDF_QUENCH_PARAMS["chol_uqt_utau_quench_yhi"],
    chol_uqs_ulgm_quench_ylo=DEFAULT_SFH_PDF_QUENCH_PARAMS["chol_uqs_ulgm_quench_ylo"],
    chol_uqs_ulgm_quench_yhi=DEFAULT_SFH_PDF_QUENCH_PARAMS["chol_uqs_ulgm_quench_yhi"],
    chol_uqs_ulgy_quench_ylo=DEFAULT_SFH_PDF_QUENCH_PARAMS["chol_uqs_ulgy_quench_ylo"],
    chol_uqs_ulgy_quench_yhi=DEFAULT_SFH_PDF_QUENCH_PARAMS["chol_uqs_ulgy_quench_yhi"],
    chol_uqs_ul_quench_ylo=DEFAULT_SFH_PDF_QUENCH_PARAMS["chol_uqs_ul_quench_ylo"],
    chol_uqs_ul_quench_yhi=DEFAULT_SFH_PDF_QUENCH_PARAMS["chol_uqs_ul_quench_yhi"],
    chol_uqs_utau_quench_ylo=DEFAULT_SFH_PDF_QUENCH_PARAMS["chol_uqs_utau_quench_ylo"],
    chol_uqs_utau_quench_yhi=DEFAULT_SFH_PDF_QUENCH_PARAMS["chol_uqs_utau_quench_yhi"],
    chol_uqs_uqt_quench_ylo=DEFAULT_SFH_PDF_QUENCH_PARAMS["chol_uqs_uqt_quench_ylo"],
    chol_uqs_uqt_quench_yhi=DEFAULT_SFH_PDF_QUENCH_PARAMS["chol_uqs_uqt_quench_yhi"],
    chol_udrop_ulgm_quench_ylo=DEFAULT_SFH_PDF_QUENCH_PARAMS[
        "chol_udrop_ulgm_quench_ylo"
    ],
    chol_udrop_ulgm_quench_yhi=DEFAULT_SFH_PDF_QUENCH_PARAMS[
        "chol_udrop_ulgm_quench_yhi"
    ],
    chol_udrop_ulgy_quench_ylo=DEFAULT_SFH_PDF_QUENCH_PARAMS[
        "chol_udrop_ulgy_quench_ylo"
    ],
    chol_udrop_ulgy_quench_yhi=DEFAULT_SFH_PDF_QUENCH_PARAMS[
        "chol_udrop_ulgy_quench_yhi"
    ],
    chol_udrop_ul_quench_ylo=DEFAULT_SFH_PDF_QUENCH_PARAMS["chol_udrop_ul_quench_ylo"],
    chol_udrop_ul_quench_yhi=DEFAULT_SFH_PDF_QUENCH_PARAMS["chol_udrop_ul_quench_yhi"],
    chol_udrop_utau_quench_ylo=DEFAULT_SFH_PDF_QUENCH_PARAMS[
        "chol_udrop_utau_quench_ylo"
    ],
    chol_udrop_utau_quench_yhi=DEFAULT_SFH_PDF_QUENCH_PARAMS[
        "chol_udrop_utau_quench_yhi"
    ],
    chol_udrop_uqt_quench_ylo=DEFAULT_SFH_PDF_QUENCH_PARAMS[
        "chol_udrop_uqt_quench_ylo"
    ],
    chol_udrop_uqt_quench_yhi=DEFAULT_SFH_PDF_QUENCH_PARAMS[
        "chol_udrop_uqt_quench_yhi"
    ],
    chol_udrop_uqs_quench_ylo=DEFAULT_SFH_PDF_QUENCH_PARAMS[
        "chol_udrop_uqs_quench_ylo"
    ],
    chol_udrop_uqs_quench_yhi=DEFAULT_SFH_PDF_QUENCH_PARAMS[
        "chol_udrop_uqs_quench_yhi"
    ],
    chol_urej_ulgm_quench_ylo=DEFAULT_SFH_PDF_QUENCH_PARAMS[
        "chol_urej_ulgm_quench_ylo"
    ],
    chol_urej_ulgm_quench_yhi=DEFAULT_SFH_PDF_QUENCH_PARAMS[
        "chol_urej_ulgm_quench_yhi"
    ],
    chol_urej_ulgy_quench_ylo=DEFAULT_SFH_PDF_QUENCH_PARAMS[
        "chol_urej_ulgy_quench_ylo"
    ],
    chol_urej_ulgy_quench_yhi=DEFAULT_SFH_PDF_QUENCH_PARAMS[
        "chol_urej_ulgy_quench_yhi"
    ],
    chol_urej_ul_quench_ylo=DEFAULT_SFH_PDF_QUENCH_PARAMS["chol_urej_ul_quench_ylo"],
    chol_urej_ul_quench_yhi=DEFAULT_SFH_PDF_QUENCH_PARAMS["chol_urej_ul_quench_yhi"],
    chol_urej_utau_quench_ylo=DEFAULT_SFH_PDF_QUENCH_PARAMS[
        "chol_urej_utau_quench_ylo"
    ],
    chol_urej_utau_quench_yhi=DEFAULT_SFH_PDF_QUENCH_PARAMS[
        "chol_urej_utau_quench_yhi"
    ],
    chol_urej_uqt_quench_ylo=DEFAULT_SFH_PDF_QUENCH_PARAMS["chol_urej_uqt_quench_ylo"],
    chol_urej_uqt_quench_yhi=DEFAULT_SFH_PDF_QUENCH_PARAMS["chol_urej_uqt_quench_yhi"],
    chol_urej_uqs_quench_ylo=DEFAULT_SFH_PDF_QUENCH_PARAMS["chol_urej_uqs_quench_ylo"],
    chol_urej_uqs_quench_yhi=DEFAULT_SFH_PDF_QUENCH_PARAMS["chol_urej_uqs_quench_yhi"],
    chol_urej_udrop_quench_ylo=DEFAULT_SFH_PDF_QUENCH_PARAMS[
        "chol_urej_udrop_quench_ylo"
    ],
    chol_urej_udrop_quench_yhi=DEFAULT_SFH_PDF_QUENCH_PARAMS[
        "chol_urej_udrop_quench_yhi"
    ],
):
    ulgm_ulgm = chol_ulgm_ulgm_quench_vs_lgm0(
        lgm, chol_ulgm_ulgm_quench_ylo, chol_ulgm_ulgm_quench_yhi
    )
    ulgy_ulgy = chol_ulgy_ulgy_quench_vs_lgm0(
        lgm, chol_ulgy_ulgy_quench_ylo, chol_ulgy_ulgy_quench_yhi
    )
    ul_ul = chol_ul_ul_quench_vs_lgm0(lgm, chol_ul_ul_quench_ylo, chol_ul_ul_quench_yhi)
    utau_utau = chol_utau_utau_quench_vs_lgm0(
        lgm, chol_utau_utau_quench_ylo, chol_utau_utau_quench_yhi
    )
    uqt_uqt = chol_uqt_uqt_quench_vs_lgm0(
        lgm, chol_uqt_uqt_quench_ylo, chol_uqt_uqt_quench_yhi
    )
    uqs_uqs = chol_uqs_uqs_quench_vs_lgm0(
        lgm, chol_uqs_uqs_quench_ylo, chol_uqs_uqs_quench_yhi
    )
    udrop_udrop = chol_udrop_udrop_quench_vs_lgm0(
        lgm, chol_udrop_udrop_quench_ylo, chol_udrop_udrop_quench_yhi
    )
    urej_urej = chol_urej_urej_quench_vs_lgm0(
        lgm, chol_urej_urej_quench_ylo, chol_urej_urej_quench_yhi
    )
    ulgy_ulgm = chol_ulgy_ulgm_quench_vs_lgm0(
        lgm, chol_ulgy_ulgm_quench_ylo, chol_ulgy_ulgm_quench_yhi
    )
    ul_ulgm = chol_ul_ulgm_quench_vs_lgm0(
        lgm, chol_ul_ulgm_quench_ylo, chol_ul_ulgm_quench_yhi
    )
    ul_ulgy = chol_ul_ulgy_quench_vs_lgm0(
        lgm, chol_ul_ulgy_quench_ylo, chol_ul_ulgy_quench_yhi
    )
    utau_ulgm = chol_utau_ulgm_quench_vs_lgm0(
        lgm, chol_utau_ulgm_quench_ylo, chol_utau_ulgm_quench_yhi
    )
    utau_ulgy = chol_utau_ulgy_quench_vs_lgm0(
        lgm, chol_utau_ulgy_quench_ylo, chol_utau_ulgy_quench_yhi
    )
    utau_ul = chol_utau_ul_quench_vs_lgm0(
        lgm, chol_utau_ul_quench_ylo, chol_utau_ul_quench_yhi
    )
    uqt_ulgm = chol_uqt_ulgm_quench_vs_lgm0(
        lgm, chol_uqt_ulgm_quench_ylo, chol_uqt_ulgm_quench_yhi
    )
    uqt_ulgy = chol_uqt_ulgy_quench_vs_lgm0(
        lgm, chol_uqt_ulgy_quench_ylo, chol_uqt_ulgy_quench_yhi
    )
    uqt_ul = chol_uqt_ul_quench_vs_lgm0(
        lgm, chol_uqt_ul_quench_ylo, chol_uqt_ul_quench_yhi
    )
    uqt_utau = chol_uqt_utau_quench_vs_lgm0(
        lgm, chol_uqt_utau_quench_ylo, chol_uqt_utau_quench_yhi
    )
    uqs_ulgm = chol_uqs_ulgm_quench_vs_lgm0(
        lgm, chol_uqs_ulgm_quench_ylo, chol_uqs_ulgm_quench_yhi
    )
    uqs_ulgy = chol_uqs_ulgy_quench_vs_lgm0(
        lgm, chol_uqs_ulgy_quench_ylo, chol_uqs_ulgy_quench_yhi
    )
    uqs_ul = chol_uqs_ul_quench_vs_lgm0(
        lgm, chol_uqs_ul_quench_ylo, chol_uqs_ul_quench_yhi
    )
    uqs_utau = chol_uqs_utau_quench_vs_lgm0(
        lgm, chol_uqs_utau_quench_ylo, chol_uqs_utau_quench_yhi
    )
    uqs_uqt = chol_uqs_uqt_quench_vs_lgm0(
        lgm, chol_uqs_uqt_quench_ylo, chol_uqs_uqt_quench_yhi
    )
    udrop_ulgm = chol_udrop_ulgm_quench_vs_lgm0(
        lgm, chol_udrop_ulgm_quench_ylo, chol_udrop_ulgm_quench_yhi
    )
    udrop_ulgy = chol_udrop_ulgy_quench_vs_lgm0(
        lgm, chol_udrop_ulgy_quench_ylo, chol_udrop_ulgy_quench_yhi
    )
    udrop_ul = chol_udrop_ul_quench_vs_lgm0(
        lgm, chol_udrop_ul_quench_ylo, chol_udrop_ul_quench_yhi
    )
    udrop_utau = chol_udrop_utau_quench_vs_lgm0(
        lgm, chol_udrop_utau_quench_ylo, chol_udrop_utau_quench_yhi
    )
    udrop_uqt = chol_udrop_uqt_quench_vs_lgm0(
        lgm, chol_udrop_uqt_quench_ylo, chol_udrop_uqt_quench_yhi
    )
    udrop_uqs = chol_udrop_uqs_quench_vs_lgm0(
        lgm, chol_udrop_uqs_quench_ylo, chol_udrop_uqs_quench_yhi
    )
    urej_ulgm = chol_urej_ulgm_quench_vs_lgm0(
        lgm, chol_urej_ulgm_quench_ylo, chol_urej_ulgm_quench_yhi
    )
    urej_ulgy = chol_urej_ulgy_quench_vs_lgm0(
        lgm, chol_urej_ulgy_quench_ylo, chol_urej_ulgy_quench_yhi
    )
    urej_ul = chol_urej_ul_quench_vs_lgm0(
        lgm, chol_urej_ul_quench_ylo, chol_urej_ul_quench_yhi
    )
    urej_utau = chol_urej_utau_quench_vs_lgm0(
        lgm, chol_urej_utau_quench_ylo, chol_urej_utau_quench_yhi
    )
    urej_uqt = chol_urej_uqt_quench_vs_lgm0(
        lgm, chol_urej_uqt_quench_ylo, chol_urej_uqt_quench_yhi
    )
    urej_uqs = chol_urej_uqs_quench_vs_lgm0(
        lgm, chol_urej_uqs_quench_ylo, chol_urej_uqs_quench_yhi
    )
    urej_udrop = chol_urej_udrop_quench_vs_lgm0(
        lgm, chol_urej_udrop_quench_ylo, chol_urej_udrop_quench_yhi
    )

    chol_params = (
        ulgm_ulgm,
        ulgy_ulgy,
        ul_ul,
        utau_utau,
        uqt_uqt,
        uqs_uqs,
        udrop_udrop,
        urej_urej,
        ulgy_ulgm,
        ul_ulgm,
        ul_ulgy,
        utau_ulgm,
        utau_ulgy,
        utau_ul,
        uqt_ulgm,
        uqt_ulgy,
        uqt_ul,
        uqt_utau,
        uqs_ulgm,
        uqs_ulgy,
        uqs_ul,
        uqs_utau,
        uqs_uqt,
        udrop_ulgm,
        udrop_ulgy,
        udrop_ul,
        udrop_utau,
        udrop_uqt,
        udrop_uqs,
        urej_ulgm,
        urej_ulgy,
        urej_ul,
        urej_utau,
        urej_uqt,
        urej_uqs,
        urej_udrop,
    )

    return chol_params
