from collections import OrderedDict
from jax import jit as jjit
from jax import numpy as jnp
from jax import vmap
from .utils import _sigmoid

TODAY = 13.8
LGT0 = jnp.log10(TODAY)

_LGM_X0, LGM_K = 13.0, 0.5

DEFAULT_R_QUENCH_PARAMS = OrderedDict(
    R_ulgm_quench_ylo=-0.15,
    R_ulgm_quench_yhi=-0.36,
    R_ulgy_quench_ylo=-7.31,
    R_ulgy_quench_yhi=8.04,
    R_ul_quench_ylo=-0.41,
    R_ul_quench_yhi=2.44,
    R_utau_quench_ylo=5.90,
    R_utau_quench_yhi=-8.93,
    R_uqt_quench_ylo=-0.03,
    R_uqt_quench_yhi=0.41,
    R_uqs_quench_ylo=7.86,
    R_uqs_quench_yhi=-10.64,
    R_udrop_quench_ylo=-4.83,
    R_udrop_quench_yhi=6.61,
    R_urej_quench_ylo=1.51,
    R_urej_quench_yhi=-0.57,
)

DEFAULT_R_MAINSEQ_PARAMS = OrderedDict(
    R_ulgm_mainseq_ylo=-0.88,
    R_ulgm_mainseq_yhi=0.99,
    R_ulgy_mainseq_ylo=14.02,
    R_ulgy_mainseq_yhi=-26.44,
    R_ul_mainseq_ylo=-0.49,
    R_ul_mainseq_yhi=6.91,
    R_utau_mainseq_ylo=27.14,
    R_utau_mainseq_yhi=-47.65,
)


# Helper functions
@jjit
def _fun_Mcrit(x, ymin, ymax):
    return _sigmoid(x, 12.0, 4.0, ymin, ymax)


@jjit
def _get_shift_to_PDF_mean_kern(p50, R):
    return R * (p50 - 0.5)


_g1 = vmap(_get_shift_to_PDF_mean_kern, in_axes=(0, None))
_get_shift_to_PDF_mean = jjit(vmap(_g1, in_axes=(None, 0)))


# Quenched functions
@jjit
def R_ulgm_quench_vs_lgm0(
    lgm0,
    R_ulgm_quench_ylo=DEFAULT_R_QUENCH_PARAMS["R_ulgm_quench_ylo"],
    R_ulgm_quench_yhi=DEFAULT_R_QUENCH_PARAMS["R_ulgm_quench_yhi"],
):
    return _fun_Mcrit(lgm0, R_ulgm_quench_ylo, R_ulgm_quench_yhi)


@jjit
def R_ulgy_quench_vs_lgm0(
    lgm0,
    R_ulgy_quench_ylo=DEFAULT_R_QUENCH_PARAMS["R_ulgy_quench_ylo"],
    R_ulgy_quench_yhi=DEFAULT_R_QUENCH_PARAMS["R_ulgy_quench_yhi"],
):
    return _sigmoid(lgm0, _LGM_X0, LGM_K, R_ulgy_quench_ylo, R_ulgy_quench_yhi)


@jjit
def R_ul_quench_vs_lgm0(
    lgm0,
    R_ul_quench_ylo=DEFAULT_R_QUENCH_PARAMS["R_ul_quench_ylo"],
    R_ul_quench_yhi=DEFAULT_R_QUENCH_PARAMS["R_ul_quench_yhi"],
):
    return _sigmoid(lgm0, _LGM_X0, LGM_K, R_ul_quench_ylo, R_ul_quench_yhi)


@jjit
def R_utau_quench_vs_lgm0(
    lgm0,
    R_utau_quench_ylo=DEFAULT_R_QUENCH_PARAMS["R_utau_quench_ylo"],
    R_utau_quench_yhi=DEFAULT_R_QUENCH_PARAMS["R_utau_quench_yhi"],
):
    return _sigmoid(lgm0, _LGM_X0, LGM_K, R_utau_quench_ylo, R_utau_quench_yhi)


@jjit
def R_uqt_quench_vs_lgm0(
    lgm0,
    R_uqt_quench_ylo=DEFAULT_R_QUENCH_PARAMS["R_uqt_quench_ylo"],
    R_uqt_quench_yhi=DEFAULT_R_QUENCH_PARAMS["R_uqt_quench_yhi"],
):
    return _sigmoid(lgm0, _LGM_X0, LGM_K, R_uqt_quench_ylo, R_uqt_quench_yhi)


@jjit
def R_uqs_quench_vs_lgm0(
    lgm0,
    R_uqs_quench_ylo=DEFAULT_R_QUENCH_PARAMS["R_uqs_quench_ylo"],
    R_uqs_quench_yhi=DEFAULT_R_QUENCH_PARAMS["R_uqs_quench_yhi"],
):
    return _sigmoid(lgm0, _LGM_X0, LGM_K, R_uqs_quench_ylo, R_uqs_quench_yhi)


@jjit
def R_udrop_quench_vs_lgm0(
    lgm0,
    R_udrop_quench_ylo=DEFAULT_R_QUENCH_PARAMS["R_udrop_quench_ylo"],
    R_udrop_quench_yhi=DEFAULT_R_QUENCH_PARAMS["R_udrop_quench_yhi"],
):
    return _sigmoid(lgm0, _LGM_X0, LGM_K, R_udrop_quench_ylo, R_udrop_quench_yhi)


@jjit
def R_urej_quench_vs_lgm0(
    lgm0,
    R_urej_quench_ylo=DEFAULT_R_QUENCH_PARAMS["R_urej_quench_ylo"],
    R_urej_quench_yhi=DEFAULT_R_QUENCH_PARAMS["R_urej_quench_yhi"],
):
    return _sigmoid(lgm0, _LGM_X0, LGM_K, R_urej_quench_ylo, R_urej_quench_yhi)


@jjit
def _get_slopes_quench(
    lgm,
    R_ulgm_quench_ylo=DEFAULT_R_QUENCH_PARAMS["R_ulgm_quench_ylo"],
    R_ulgm_quench_yhi=DEFAULT_R_QUENCH_PARAMS["R_ulgm_quench_yhi"],
    R_ulgy_quench_ylo=DEFAULT_R_QUENCH_PARAMS["R_ulgy_quench_ylo"],
    R_ulgy_quench_yhi=DEFAULT_R_QUENCH_PARAMS["R_ulgy_quench_yhi"],
    R_ul_quench_ylo=DEFAULT_R_QUENCH_PARAMS["R_ul_quench_ylo"],
    R_ul_quench_yhi=DEFAULT_R_QUENCH_PARAMS["R_ul_quench_yhi"],
    R_utau_quench_ylo=DEFAULT_R_QUENCH_PARAMS["R_utau_quench_ylo"],
    R_utau_quench_yhi=DEFAULT_R_QUENCH_PARAMS["R_utau_quench_yhi"],
    R_uqt_quench_ylo=DEFAULT_R_QUENCH_PARAMS["R_uqt_quench_ylo"],
    R_uqt_quench_yhi=DEFAULT_R_QUENCH_PARAMS["R_uqt_quench_yhi"],
    R_uqs_quench_ylo=DEFAULT_R_QUENCH_PARAMS["R_uqs_quench_ylo"],
    R_uqs_quench_yhi=DEFAULT_R_QUENCH_PARAMS["R_uqs_quench_yhi"],
    R_udrop_quench_ylo=DEFAULT_R_QUENCH_PARAMS["R_udrop_quench_ylo"],
    R_udrop_quench_yhi=DEFAULT_R_QUENCH_PARAMS["R_udrop_quench_yhi"],
    R_urej_quench_ylo=DEFAULT_R_QUENCH_PARAMS["R_urej_quench_ylo"],
    R_urej_quench_yhi=DEFAULT_R_QUENCH_PARAMS["R_urej_quench_yhi"],
):
    R_ulgm = R_ulgm_quench_vs_lgm0(lgm, R_ulgm_quench_ylo, R_ulgm_quench_yhi)
    R_ulgy = R_ulgy_quench_vs_lgm0(lgm, R_ulgy_quench_ylo, R_ulgy_quench_yhi)
    R_ul = R_ul_quench_vs_lgm0(lgm, R_ul_quench_ylo, R_ul_quench_yhi)
    R_utau = R_utau_quench_vs_lgm0(lgm, R_utau_quench_ylo, R_utau_quench_yhi)
    R_uqt = R_uqt_quench_vs_lgm0(lgm, R_uqt_quench_ylo, R_uqt_quench_yhi)
    R_uqs = R_uqs_quench_vs_lgm0(lgm, R_uqs_quench_ylo, R_uqs_quench_yhi)
    R_udrop = R_udrop_quench_vs_lgm0(lgm, R_udrop_quench_ylo, R_udrop_quench_yhi)
    R_urej = R_urej_quench_vs_lgm0(lgm, R_urej_quench_ylo, R_urej_quench_yhi)

    slopes = (
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


# Main Sequence functions
@jjit
def R_ulgm_mainseq_vs_lgm0(
    lgm0,
    R_ulgm_mainseq_ylo=DEFAULT_R_MAINSEQ_PARAMS["R_ulgm_mainseq_ylo"],
    R_ulgm_mainseq_yhi=DEFAULT_R_MAINSEQ_PARAMS["R_ulgm_mainseq_yhi"],
):
    return _sigmoid(lgm0, _LGM_X0, LGM_K, R_ulgm_mainseq_ylo, R_ulgm_mainseq_yhi)


@jjit
def R_ulgy_mainseq_vs_lgm0(
    lgm0,
    R_ulgy_mainseq_ylo=DEFAULT_R_MAINSEQ_PARAMS["R_ulgy_mainseq_ylo"],
    R_ulgy_mainseq_yhi=DEFAULT_R_MAINSEQ_PARAMS["R_ulgy_mainseq_yhi"],
):
    return _sigmoid(lgm0, _LGM_X0, LGM_K, R_ulgy_mainseq_ylo, R_ulgy_mainseq_yhi)


@jjit
def R_ul_mainseq_vs_lgm0(
    lgm0,
    R_ul_mainseq_ylo=DEFAULT_R_MAINSEQ_PARAMS["R_ul_mainseq_ylo"],
    R_ul_mainseq_yhi=DEFAULT_R_MAINSEQ_PARAMS["R_ul_mainseq_yhi"],
):
    return _sigmoid(lgm0, _LGM_X0, LGM_K, R_ul_mainseq_ylo, R_ul_mainseq_yhi)


@jjit
def R_utau_mainseq_vs_lgm0(
    lgm0,
    R_utau_mainseq_ylo=DEFAULT_R_MAINSEQ_PARAMS["R_utau_mainseq_ylo"],
    R_utau_mainseq_yhi=DEFAULT_R_MAINSEQ_PARAMS["R_utau_mainseq_yhi"],
):
    return _sigmoid(lgm0, _LGM_X0, LGM_K, R_utau_mainseq_ylo, R_utau_mainseq_yhi)


@jjit
def _get_slopes_mainseq(
    lgm,
    R_ulgm_mainseq_ylo=DEFAULT_R_MAINSEQ_PARAMS["R_ulgm_mainseq_ylo"],
    R_ulgm_mainseq_yhi=DEFAULT_R_MAINSEQ_PARAMS["R_ulgm_mainseq_yhi"],
    R_ulgy_mainseq_ylo=DEFAULT_R_MAINSEQ_PARAMS["R_ulgy_mainseq_ylo"],
    R_ulgy_mainseq_yhi=DEFAULT_R_MAINSEQ_PARAMS["R_ulgy_mainseq_yhi"],
    R_ul_mainseq_ylo=DEFAULT_R_MAINSEQ_PARAMS["R_ul_mainseq_ylo"],
    R_ul_mainseq_yhi=DEFAULT_R_MAINSEQ_PARAMS["R_ul_mainseq_yhi"],
    R_utau_mainseq_ylo=DEFAULT_R_MAINSEQ_PARAMS["R_utau_mainseq_ylo"],
    R_utau_mainseq_yhi=DEFAULT_R_MAINSEQ_PARAMS["R_utau_mainseq_yhi"],
):
    R_ulgm = R_ulgm_mainseq_vs_lgm0(lgm, R_ulgm_mainseq_ylo, R_ulgm_mainseq_yhi)
    R_ulgy = R_ulgy_mainseq_vs_lgm0(lgm, R_ulgy_mainseq_ylo, R_ulgy_mainseq_yhi)
    R_ul = R_ul_mainseq_vs_lgm0(lgm, R_ul_mainseq_ylo, R_ul_mainseq_yhi)
    R_utau = R_utau_mainseq_vs_lgm0(lgm, R_utau_mainseq_ylo, R_utau_mainseq_yhi)

    slopes = (
        R_ulgm,
        R_ulgy,
        R_ul,
        R_utau,
    )

    return slopes
