from collections import OrderedDict
from jax import jit as jjit
from jax import numpy as jnp
from jax import vmap

TODAY = 13.8
LGT0 = jnp.log10(TODAY)

_LGM_X0, LGM_K = 13.0, 0.5

DEFAULT_R_QUENCH_PARAMS = OrderedDict(
    R_ulgm_quench_ylo=-0.71,
    R_ulgm_quench_yhi=0.43,
    R_ulgy_quench_ylo=-1.02,
    R_ulgy_quench_yhi=0.69,
    R_ul_quench_ylo=-0.19,
    R_ul_quench_yhi=0.86,
    R_utau_quench_ylo=0.83,
    R_utau_quench_yhi=-1.40,
    R_uqt_quench_ylo=0.35,
    R_uqt_quench_yhi=0.16,
    R_uqs_quench_ylo=0.77,
    R_uqs_quench_yhi=-1.06,
    R_udrop_quench_ylo=-0.56,
    R_udrop_quench_yhi=1.10,
    R_urej_quench_ylo=-0.30,
    R_urej_quench_yhi=0.85,
)

DEFAULT_R_MAINSEQ_PARAMS = OrderedDict(
    R_ulgm_mainseq_ylo=-0.71,
    R_ulgm_mainseq_yhi=0.43,
    R_ulgy_mainseq_ylo=-1.02,
    R_ulgy_mainseq_yhi=0.69,
    R_ul_mainseq_ylo=-0.19,
    R_ul_mainseq_yhi=0.86,
    R_utau_mainseq_ylo=0.83,
    R_utau_mainseq_yhi=-1.40,
)


# Helper functions
@jjit
def _sigmoid(x, logtc, k, ymin, ymax):
    height_diff = ymax - ymin
    return ymin + height_diff / (1.0 + jnp.exp(-k * (x - logtc)))


@jjit
def _inverse_sigmoid(y, x0=0, k=1, ymin=-1, ymax=1):
    lnarg = (ymax - ymin) / (y - ymin) - 1
    return x0 - jnp.log(lnarg) / k


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
