""""""

from collections import OrderedDict, namedtuple

from jax import jit as jjit
from jax import numpy as jnp

from ..utils import _sigmoid

TODAY = 13.8
LGT0 = jnp.log10(TODAY)

_LGM_X0, LGM_K = 13.0, 0.5

DEFAULT_AB_QSEQ_PDICT = OrderedDict(
    R_Fquench=0.994,
    R_ulgm_quench_ylo=-1.264,
    R_ulgm_quench_yhi=0.440,
    R_ulgy_quench_ylo=-0.165,
    R_ulgy_quench_yhi=-1.083,
    R_ul_quench_ylo=5.653,
    R_ul_quench_yhi=-4.126,
    R_utau_quench_ylo=5.613,
    R_utau_quench_yhi=-6.922,
    R_uqt_quench_ylo=0.190,
    R_uqt_quench_yhi=0.191,
    R_uqs_quench_ylo=2.124,
    R_uqs_quench_yhi=-2.699,
    R_udrop_quench_ylo=1.071,
    R_udrop_quench_yhi=-0.774,
    R_urej_quench_ylo=1.912,
    R_urej_quench_yhi=-1.761,
)
DEFAULT_AB_MAINSEQ_PDICT = OrderedDict(
    R_ulgm_mainseq_ylo=-0.558,
    R_ulgm_mainseq_yhi=0.575,
    R_ulgy_mainseq_ylo=-1.227,
    R_ulgy_mainseq_yhi=2.171,
    R_ul_mainseq_ylo=5.164,
    R_ul_mainseq_yhi=-3.384,
    R_utau_mainseq_ylo=5.828,
    R_utau_mainseq_yhi=-5.587,
)
ABMainseqParams = namedtuple("Params", list(DEFAULT_AB_MAINSEQ_PDICT.keys()))
DEFAULT_AB_MAINSEQ_PARAMS = ABMainseqParams(**DEFAULT_AB_MAINSEQ_PDICT)

_UPNAMES = ["u_" + key for key in DEFAULT_AB_MAINSEQ_PDICT.keys()]
ABMainseqUParams = namedtuple("ABMainseqUParams", _UPNAMES)

ABQseqParams = namedtuple("Params", list(DEFAULT_AB_QSEQ_PDICT.keys()))
DEFAULT_AB_QSEQ_PARAMS = ABQseqParams(**DEFAULT_AB_QSEQ_PDICT)

_UPNAMES = ["u_" + key for key in DEFAULT_AB_QSEQ_PDICT.keys()]
ABQseqUParams = namedtuple("ABQseqUParams", _UPNAMES)


@jjit
def _fun_Mcrit(x, ymin, ymax):
    return _sigmoid(x, 12.0, 4.0, ymin, ymax)


@jjit
def _get_slopes_qseq(params, lgm):
    R_ulgm = _fun_Mcrit(lgm, params.R_ulgm_quench_ylo, params.R_ulgm_quench_yhi)
    R_ulgy = _sigmoid(
        lgm, _LGM_X0, LGM_K, params.R_ulgy_quench_ylo, params.R_ulgy_quench_yhi
    )

    R_ul = _sigmoid(lgm, _LGM_X0, LGM_K, params.R_ul_quench_ylo, params.R_ul_quench_yhi)
    R_utau = _sigmoid(
        lgm, _LGM_X0, LGM_K, params.R_utau_quench_ylo, params.R_utau_quench_yhi
    )
    R_uqt = _sigmoid(
        lgm, _LGM_X0, LGM_K, params.R_uqt_quench_ylo, params.R_uqt_quench_yhi
    )
    R_uqs = _sigmoid(
        lgm, _LGM_X0, LGM_K, params.R_uqs_quench_ylo, params.R_uqs_quench_yhi
    )
    R_udrop = _sigmoid(
        lgm, _LGM_X0, LGM_K, params.R_udrop_quench_ylo, params.R_udrop_quench_yhi
    )
    R_urej = _sigmoid(
        lgm, _LGM_X0, LGM_K, params.R_urej_quench_ylo, params.R_urej_quench_yhi
    )

    slopes = (
        params.R_Fquench,
        R_ulgm,
        R_ulgy,
        R_ul,
        R_utau,
        R_uqt,
        R_uqs,
        R_udrop,
        R_urej,
    )

    return slopes


@jjit
def _get_slopes_mainseq(params, lgm):
    R_ulgm = _sigmoid(
        lgm, _LGM_X0, LGM_K, params.R_ulgm_mainseq_ylo, params.R_ulgm_mainseq_yhi
    )
    R_ulgy = _sigmoid(
        lgm, _LGM_X0, LGM_K, params.R_ulgy_mainseq_ylo, params.R_ulgy_mainseq_yhi
    )
    R_ul = _sigmoid(
        lgm, _LGM_X0, LGM_K, params.R_ul_mainseq_ylo, params.R_ul_mainseq_yhi
    )
    R_utau = _sigmoid(
        lgm, _LGM_X0, LGM_K, params.R_utau_mainseq_ylo, params.R_utau_mainseq_yhi
    )
    slopes = (R_ulgm, R_ulgy, R_ul, R_utau)
    return slopes


@jjit
def get_bounded_ab_mainseq_params(u_params):
    return ABMainseqParams(*u_params)


@jjit
def get_unbounded_ab_mainseq_params(params):
    return ABMainseqUParams(*params)


DEFAULT_AB_MAINSEQ_U_PARAMS = ABMainseqUParams(
    *get_unbounded_ab_mainseq_params(DEFAULT_AB_MAINSEQ_PARAMS)
)


@jjit
def get_bounded_ab_qseq_params(u_params):
    return ABQseqParams(*u_params)


@jjit
def get_unbounded_ab_qseq_params(params):
    return ABQseqUParams(*params)


DEFAULT_AB_QSEQ_U_PARAMS = ABQseqUParams(
    *get_unbounded_ab_qseq_params(DEFAULT_AB_QSEQ_PARAMS)
)
