"""Model of a quenched galaxy population calibrated to SMDPL halos."""

from collections import OrderedDict, namedtuple

from jax import jit as jjit
from jax import numpy as jnp

from ..utils import _sigmoid

from diffmah.utils import get_cholesky_from_params

TODAY = 13.8
LGT0 = jnp.log10(TODAY)

LGM_X0, LGM_K = 13.0, 0.5

DEFAULT_SFH_PDF_QUENCH_PDICT = OrderedDict(
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
QseqMassOnlyParams = namedtuple("Params", list(DEFAULT_SFH_PDF_QUENCH_PDICT.keys()))
DEFAULT_SFH_PDF_QUENCH_PARAMS = QseqMassOnlyParams(**DEFAULT_SFH_PDF_QUENCH_PDICT)

_UPNAMES = ["u_" + key for key in DEFAULT_SFH_PDF_QUENCH_PDICT.keys()]
QseqMassOnlyUParams = namedtuple("QseqMassOnlyUParams", _UPNAMES)


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
def _get_cov_scalar(chol_params):
    chol = get_cholesky_from_params(jnp.array(chol_params))
    cov = jnp.dot(chol, chol.T)
    return cov


@jjit
def _get_mean_u_params_qseq(params, lgm):
    ulgm = _fun_Mcrit(lgm, params.mean_ulgm_quench_ylo, params.mean_ulgm_quench_yhi)
    ulgy = _fun(lgm, params.mean_ulgy_quench_ylo, params.mean_ulgy_quench_yhi)
    ul = _fun(lgm, params.mean_ul_quench_ylo, params.mean_ul_quench_yhi)
    utau = _fun(lgm, params.mean_utau_quench_ylo, params.mean_utau_quench_yhi)
    uqt = _fun(lgm, params.mean_uqt_quench_ylo, params.mean_uqt_quench_yhi)
    uqs = _fun(lgm, params.mean_uqs_quench_ylo, params.mean_uqs_quench_yhi)
    udrop = _fun(lgm, params.mean_udrop_quench_ylo, params.mean_udrop_quench_yhi)
    urej = _fun(lgm, params.mean_urej_quench_ylo, params.mean_urej_quench_yhi)
    return ulgm, ulgy, ul, utau, uqt, uqs, udrop, urej


@jjit
def _get_chol_u_params_qseq(params, lgm):
    ulgm_ulgm = _fun_chol_diag(
        lgm, params.chol_ulgm_ulgm_quench_ylo, params.chol_ulgm_ulgm_quench_yhi
    )
    ulgy_ulgy = _fun_chol_diag(
        lgm, params.chol_ulgy_ulgy_quench_ylo, params.chol_ulgy_ulgy_quench_yhi
    )
    ul_ul = _fun_chol_diag(
        lgm, params.chol_ul_ul_quench_ylo, params.chol_ul_ul_quench_yhi
    )
    utau_utau = _fun_chol_diag(
        lgm, params.chol_utau_utau_quench_ylo, params.chol_utau_utau_quench_yhi
    )
    uqt_uqt = _fun_chol_diag(
        lgm, params.chol_uqt_uqt_quench_ylo, params.chol_uqt_uqt_quench_yhi
    )
    uqs_uqs = _fun_chol_diag(
        lgm, params.chol_uqs_uqs_quench_ylo, params.chol_uqs_uqs_quench_yhi
    )
    udrop_udrop = _fun_chol_diag(
        lgm, params.chol_udrop_udrop_quench_ylo, params.chol_udrop_udrop_quench_yhi
    )
    urej_urej = _fun_chol_diag(
        lgm, params.chol_urej_urej_quench_ylo, params.chol_urej_urej_quench_yhi
    )
    ulgy_ulgm = _fun(
        lgm, params.chol_ulgy_ulgm_quench_ylo, params.chol_ulgy_ulgm_quench_yhi
    )
    ul_ulgm = _fun(lgm, params.chol_ul_ulgm_quench_ylo, params.chol_ul_ulgm_quench_yhi)
    ul_ulgy = _fun(lgm, params.chol_ul_ulgy_quench_ylo, params.chol_ul_ulgy_quench_yhi)
    utau_ulgm = _fun(
        lgm, params.chol_utau_ulgm_quench_ylo, params.chol_utau_ulgm_quench_yhi
    )
    utau_ulgy = _fun(
        lgm, params.chol_utau_ulgy_quench_ylo, params.chol_utau_ulgy_quench_yhi
    )
    utau_ul = _fun(lgm, params.chol_utau_ul_quench_ylo, params.chol_utau_ul_quench_yhi)
    uqt_ulgm = _fun(
        lgm, params.chol_uqt_ulgm_quench_ylo, params.chol_uqt_ulgm_quench_yhi
    )
    uqt_ulgy = _fun(
        lgm, params.chol_uqt_ulgy_quench_ylo, params.chol_uqt_ulgy_quench_yhi
    )
    uqt_ul = _fun(lgm, params.chol_uqt_ul_quench_ylo, params.chol_uqt_ul_quench_yhi)
    uqt_utau = _fun(
        lgm, params.chol_uqt_utau_quench_ylo, params.chol_uqt_utau_quench_yhi
    )
    uqs_ulgm = _fun(
        lgm, params.chol_uqs_ulgm_quench_ylo, params.chol_uqs_ulgm_quench_yhi
    )
    uqs_ulgy = _fun(
        lgm, params.chol_uqs_ulgy_quench_ylo, params.chol_uqs_ulgy_quench_yhi
    )
    uqs_ul = _fun(lgm, params.chol_uqs_ul_quench_ylo, params.chol_uqs_ul_quench_yhi)
    uqs_utau = _fun(
        lgm, params.chol_uqs_utau_quench_ylo, params.chol_uqs_utau_quench_yhi
    )
    uqs_uqt = _fun(lgm, params.chol_uqs_uqt_quench_ylo, params.chol_uqs_uqt_quench_yhi)
    udrop_ulgm = _fun(
        lgm, params.chol_udrop_ulgm_quench_ylo, params.chol_udrop_ulgm_quench_yhi
    )
    udrop_ulgy = _fun(
        lgm, params.chol_udrop_ulgy_quench_ylo, params.chol_udrop_ulgy_quench_yhi
    )
    udrop_ul = _fun(
        lgm, params.chol_udrop_ul_quench_ylo, params.chol_udrop_ul_quench_yhi
    )
    udrop_utau = _fun(
        lgm, params.chol_udrop_utau_quench_ylo, params.chol_udrop_utau_quench_yhi
    )
    udrop_uqt = _fun(
        lgm, params.chol_udrop_uqt_quench_ylo, params.chol_udrop_uqt_quench_yhi
    )
    udrop_uqs = _fun(
        lgm, params.chol_udrop_uqs_quench_ylo, params.chol_udrop_uqs_quench_yhi
    )
    urej_ulgm = _fun(
        lgm, params.chol_urej_ulgm_quench_ylo, params.chol_urej_ulgm_quench_yhi
    )
    urej_ulgy = _fun(
        lgm, params.chol_urej_ulgy_quench_ylo, params.chol_urej_ulgy_quench_yhi
    )
    urej_ul = _fun(lgm, params.chol_urej_ul_quench_ylo, params.chol_urej_ul_quench_yhi)
    urej_utau = _fun(
        lgm, params.chol_urej_utau_quench_ylo, params.chol_urej_utau_quench_yhi
    )
    urej_uqt = _fun(
        lgm, params.chol_urej_uqt_quench_ylo, params.chol_urej_uqt_quench_yhi
    )
    urej_uqs = _fun(
        lgm, params.chol_urej_uqs_quench_ylo, params.chol_urej_uqs_quench_yhi
    )
    urej_udrop = _fun(
        lgm, params.chol_urej_udrop_quench_ylo, params.chol_urej_udrop_quench_yhi
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


@jjit
def _get_cov_qseq(params, lgm):
    chol_params = _get_chol_u_params_qseq(params, lgm)
    cov_qseq = _get_cov_scalar(chol_params)
    return cov_qseq


@jjit
def _frac_quench_vs_lgm0(params, lgm0):
    return _fun_fquench(
        lgm0,
        params.frac_quench_x0,
        params.frac_quench_k,
        params.frac_quench_ylo,
        params.frac_quench_yhi,
    )


@jjit
def get_bounded_qseq_massonly_params(u_params):
    return QseqMassOnlyParams(*u_params)


@jjit
def get_unbounded_qseq_massonly_params(params):
    return QseqMassOnlyUParams(*params)


DEFAULT_SFH_PDF_QUENCH_U_PARAMS = QseqMassOnlyUParams(
    *get_unbounded_qseq_massonly_params(DEFAULT_SFH_PDF_QUENCH_PARAMS)
)
