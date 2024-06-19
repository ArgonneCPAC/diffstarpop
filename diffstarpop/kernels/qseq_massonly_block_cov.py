"""Model of a quenched galaxy population calibrated to SMDPL halos."""

from collections import OrderedDict, namedtuple

from diffmah.utils import get_cholesky_from_params
from jax import jit as jjit
from jax import numpy as jnp
from jax import vmap

from ..utils import _inverse_sigmoid, _sigmoid, covariance_from_correlation

TODAY = 13.8
LGT0 = jnp.log10(TODAY)

LGM_X0, LGM_K = 12.5, 1.0
LGMCRIT_K = 4.0
BOUNDING_K = 0.1
RHO_BOUNDS = (-0.3, 0.3)

SFH_PDF_QUENCH_MU_PDICT = OrderedDict(
    mean_lgmhalo_x0=12.5,
    mean_ulgm_quench_ylo=11.540,
    mean_ulgm_quench_yhi=12.080,
    mean_ulgy_quench_ylo=-0.481,
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
)
SFH_PDF_QUENCH_MU_BOUNDS_PDICT = OrderedDict(
    mean_lgmhalo_x0=(11.5, 13.5),
    mean_ulgm_quench_ylo=(11.5, 13.0),
    mean_ulgm_quench_yhi=(11.5, 13.0),
    mean_ulgy_quench_ylo=(-1.0, 0.0),
    mean_ulgy_quench_yhi=(-1.0, 0.0),
    mean_ul_quench_ylo=(-0.5, 0.5),
    mean_ul_quench_yhi=(-0.5, 0.5),
    mean_utau_quench_ylo=(-5.0, 20.0),
    mean_utau_quench_yhi=(-5.0, 20.0),
    mean_uqt_quench_ylo=(2.0, 0.0),
    mean_uqt_quench_yhi=(2.0, 0.0),
    mean_uqs_quench_ylo=(-2.0, 2.0),
    mean_uqs_quench_yhi=(-2.0, 2.0),
    mean_udrop_quench_ylo=(-3.0, -1.0),
    mean_udrop_quench_yhi=(-3.0, -1.0),
    mean_urej_quench_ylo=(-3.0, 1.0),
    mean_urej_quench_yhi=(-3.0, 1.0),
)

SFH_PDF_QUENCH_COV_MS_BLOCK_PDICT = OrderedDict(
    std_ulgm_quench_ylo=0.2,
    std_ulgm_quench_yhi=0.45,
    std_ulgy_quench_ylo=0.45,
    std_ulgy_quench_yhi=0.7,
    std_ul_quench_ylo=2.5,
    std_ul_quench_yhi=0.5,
    std_utau_quench_ylo=3.5,
    std_utau_quench_yhi=6,
    rho_ulgy_ulgm_quench_ylo=0.0,
    rho_ulgy_ulgm_quench_yhi=0.0,
    rho_ul_ulgm_quench_ylo=0.0,
    rho_ul_ulgm_quench_yhi=0.0,
    rho_ul_ulgy_quench_ylo=0.0,
    rho_ul_ulgy_quench_yhi=0.0,
    rho_utau_ulgm_quench_ylo=0.0,
    rho_utau_ulgm_quench_yhi=0.0,
    rho_utau_ulgy_quench_ylo=0.0,
    rho_utau_ulgy_quench_yhi=0.0,
    rho_utau_ul_quench_ylo=0.0,
    rho_utau_ul_quench_yhi=0.0,
)
SFH_PDF_QUENCH_COV_MS_BLOCK_BOUNDS_PDICT = OrderedDict(
    std_ulgm_quench_ylo=(0.1, 1.0),
    std_ulgm_quench_yhi=(0.1, 1.0),
    std_ulgy_quench_ylo=(0.1, 1.0),
    std_ulgy_quench_yhi=(0.1, 1.0),
    std_ul_quench_ylo=(0.5, 3.0),
    std_ul_quench_yhi=(0.5, 3.0),
    std_utau_quench_ylo=(1.0, 8.0),
    std_utau_quench_yhi=(1.0, 8.0),
    rho_ulgy_ulgm_quench_ylo=RHO_BOUNDS,
    rho_ulgy_ulgm_quench_yhi=RHO_BOUNDS,
    rho_ul_ulgm_quench_ylo=RHO_BOUNDS,
    rho_ul_ulgm_quench_yhi=RHO_BOUNDS,
    rho_ul_ulgy_quench_ylo=RHO_BOUNDS,
    rho_ul_ulgy_quench_yhi=RHO_BOUNDS,
    rho_utau_ulgm_quench_ylo=RHO_BOUNDS,
    rho_utau_ulgm_quench_yhi=RHO_BOUNDS,
    rho_utau_ulgy_quench_ylo=RHO_BOUNDS,
    rho_utau_ulgy_quench_yhi=RHO_BOUNDS,
    rho_utau_ul_quench_ylo=RHO_BOUNDS,
    rho_utau_ul_quench_yhi=RHO_BOUNDS,
)

SFH_PDF_QUENCH_COV_Q_BLOCK_PDICT = OrderedDict(
    std_uqt_quench_ylo=0.275,
    std_uqt_quench_yhi=0.1,
    std_uqs_quench_ylo=0.45,
    std_uqs_quench_yhi=0.65,
    std_udrop_quench_ylo=0.5,
    std_udrop_quench_yhi=0.5,
    std_urej_quench_ylo=0.3,
    std_urej_quench_yhi=0.75,
    rho_uqs_uqt_quench_ylo=0.0,
    rho_uqs_uqt_quench_yhi=0.0,
    rho_udrop_uqt_quench_ylo=0.0,
    rho_udrop_uqt_quench_yhi=0.0,
    rho_udrop_uqs_quench_ylo=0.0,
    rho_udrop_uqs_quench_yhi=0.0,
    rho_urej_uqt_quench_ylo=0.0,
    rho_urej_uqt_quench_yhi=0.0,
    rho_urej_uqs_quench_ylo=0.0,
    rho_urej_uqs_quench_yhi=0.0,
    rho_urej_udrop_quench_ylo=0.0,
    rho_urej_udrop_quench_yhi=0.0,
)
SFH_PDF_QUENCH_COV_Q_BLOCK_BOUNDS_PDICT = OrderedDict(
    std_uqt_quench_ylo=(0.1, 0.5),
    std_uqt_quench_yhi=(0.1, 0.5),
    std_uqs_quench_ylo=(0.1, 1.0),
    std_uqs_quench_yhi=(0.1, 1.0),
    std_udrop_quench_ylo=(0.1, 1.0),
    std_udrop_quench_yhi=(0.1, 1.0),
    std_urej_quench_ylo=(0.1, 1.0),
    std_urej_quench_yhi=(0.1, 1.0),
    rho_uqs_uqt_quench_ylo=RHO_BOUNDS,
    rho_uqs_uqt_quench_yhi=RHO_BOUNDS,
    rho_udrop_uqt_quench_ylo=RHO_BOUNDS,
    rho_udrop_uqt_quench_yhi=RHO_BOUNDS,
    rho_udrop_uqs_quench_ylo=RHO_BOUNDS,
    rho_udrop_uqs_quench_yhi=RHO_BOUNDS,
    rho_urej_uqt_quench_ylo=RHO_BOUNDS,
    rho_urej_uqt_quench_yhi=RHO_BOUNDS,
    rho_urej_uqs_quench_ylo=RHO_BOUNDS,
    rho_urej_uqs_quench_yhi=RHO_BOUNDS,
    rho_urej_udrop_quench_ylo=RHO_BOUNDS,
    rho_urej_udrop_quench_yhi=RHO_BOUNDS,
)
DEFAULT_SFH_PDF_QUENCH_MS_BLOCK_PDICT = OrderedDict(
    mean_ulgm_quench_ylo=11.540,
    mean_ulgm_quench_yhi=12.080,
    mean_ulgy_quench_ylo=0.481,
    mean_ulgy_quench_yhi=-0.223,
    mean_ul_quench_ylo=-1.274,
    mean_ul_quench_yhi=1.766,
    mean_utau_quench_ylo=55.480,
    mean_utau_quench_yhi=-66.540,
    chol_ulgm_ulgm_quench_ylo=-1.645,
    chol_ulgm_ulgm_quench_yhi=0.010,
    chol_ulgy_ulgy_quench_ylo=-1.125,
    chol_ulgy_ulgy_quench_yhi=-0.530,
    chol_ul_ul_quench_ylo=-0.701,
    chol_ul_ul_quench_yhi=0.544,
    chol_utau_utau_quench_ylo=0.833,
    chol_utau_utau_quench_yhi=1.100,
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
)
DEFAULT_SFH_PDF_QUENCH_Q_BLOCK_PDICT = OrderedDict(
    mean_uqt_quench_ylo=1.744,
    mean_uqt_quench_yhi=0.042,
    mean_uqs_quench_ylo=-2.979,
    mean_uqs_quench_yhi=3.520,
    mean_udrop_quench_ylo=-0.508,
    mean_udrop_quench_yhi=-3.785,
    mean_urej_quench_ylo=2.139,
    mean_urej_quench_yhi=-3.043,
    chol_uqt_uqt_quench_ylo=-1.001,
    chol_uqt_uqt_quench_yhi=-1.228,
    chol_uqs_uqs_quench_ylo=-0.814,
    chol_uqs_uqs_quench_yhi=-0.560,
    chol_udrop_udrop_quench_ylo=-0.612,
    chol_udrop_udrop_quench_yhi=-0.824,
    chol_urej_urej_quench_ylo=0.560,
    chol_urej_urej_quench_yhi=-1.103,
    chol_uqs_uqt_quench_ylo=-0.395,
    chol_uqs_uqt_quench_yhi=-0.508,
    chol_udrop_uqt_quench_ylo=2.323,
    chol_udrop_uqt_quench_yhi=3.009,
    chol_udrop_uqs_quench_ylo=1.021,
    chol_udrop_uqs_quench_yhi=-0.074,
    chol_urej_uqt_quench_ylo=-0.508,
    chol_urej_uqt_quench_yhi=0.758,
    chol_urej_uqs_quench_ylo=1.561,
    chol_urej_uqs_quench_yhi=2.030,
    chol_urej_udrop_quench_ylo=-1.445,
    chol_urej_udrop_quench_yhi=-2.245,
)
DEFAULT_SFH_PDF_QUENCH_BLOCK_PDICT = OrderedDict(
    frac_quench_x0=11.860,
    frac_quench_k=4.5,
    frac_quench_ylo=0.05,
    frac_quench_yhi=0.95,
)
DEFAULT_SFH_PDF_FRAC_QUENCH_PDICT = OrderedDict(
    frac_quench_x0=11.860,
    frac_quench_k=4.5,
    frac_quench_ylo=0.05,
    frac_quench_yhi=0.95,
)
DEFAULT_SFH_PDF_FRAC_QUENCH_BOUNDS_PDICT = OrderedDict(
    frac_quench_x0=(11.0, 13.0),
    frac_quench_k=(1.0, 5.0),
    frac_quench_ylo=(0.0, 0.5),
    frac_quench_yhi=(0.25, 1.0),
)

DEFAULT_SFH_PDF_QUENCH_BLOCK_PDICT.update(DEFAULT_SFH_PDF_QUENCH_MS_BLOCK_PDICT)
DEFAULT_SFH_PDF_QUENCH_BLOCK_PDICT.update(DEFAULT_SFH_PDF_QUENCH_Q_BLOCK_PDICT)

SFH_PDF_QUENCH_PDICT = DEFAULT_SFH_PDF_FRAC_QUENCH_PDICT.copy()
SFH_PDF_QUENCH_PDICT.update(SFH_PDF_QUENCH_MU_PDICT)
SFH_PDF_QUENCH_PDICT.update(SFH_PDF_QUENCH_COV_MS_BLOCK_PDICT)
SFH_PDF_QUENCH_PDICT.update(SFH_PDF_QUENCH_COV_Q_BLOCK_PDICT)

SFH_PDF_QUENCH_BOUNDS_PDICT = DEFAULT_SFH_PDF_FRAC_QUENCH_BOUNDS_PDICT.copy()
SFH_PDF_QUENCH_BOUNDS_PDICT.update(SFH_PDF_QUENCH_MU_BOUNDS_PDICT)
SFH_PDF_QUENCH_BOUNDS_PDICT.update(SFH_PDF_QUENCH_COV_MS_BLOCK_BOUNDS_PDICT)
SFH_PDF_QUENCH_BOUNDS_PDICT.update(SFH_PDF_QUENCH_COV_Q_BLOCK_BOUNDS_PDICT)

QseqMassOnlyBlockParams = namedtuple(
    "QseqMassOnlyBlockParams", list(DEFAULT_SFH_PDF_QUENCH_BLOCK_PDICT.keys())
)
DEFAULT_SFH_PDF_QUENCH_BLOCK_PARAMS = QseqMassOnlyBlockParams(
    **DEFAULT_SFH_PDF_QUENCH_BLOCK_PDICT
)

_UPNAMES = ["u_" + key for key in DEFAULT_SFH_PDF_QUENCH_BLOCK_PDICT.keys()]
QseqMassOnlyBlockUParams = namedtuple("QseqMassOnlyBlockUParams", _UPNAMES)


QseqParams = namedtuple("QseqParams", list(SFH_PDF_QUENCH_PDICT.keys()))
SFH_PDF_QUENCH_PARAMS = QseqParams(**SFH_PDF_QUENCH_PDICT)
SFH_PDF_QUENCH_PBOUNDS = QseqParams(**SFH_PDF_QUENCH_BOUNDS_PDICT)

_UPNAMES = ["u_" + key for key in QseqParams._fields]
QseqUParams = namedtuple("QseqUParams", _UPNAMES)

FRAC_Q_BOUNDS_PDICT = OrderedDict(
    frac_quench_x0=(10.0, 15.0),
    frac_quench_k=(1.0, 5.0),
    frac_quench_ylo=(0.0, 1.0),
    frac_quench_yhi=(0.0, 1.0),
)
FRAC_Q_PNAMES = list(FRAC_Q_BOUNDS_PDICT.keys())
FracQParams = namedtuple("FracQParams", FRAC_Q_PNAMES)
_FRAC_Q_UPNAMES = ["u_" + key for key in FRAC_Q_PNAMES]
FracQUParams = namedtuple("FracQUParams", _FRAC_Q_UPNAMES)

FRAC_Q_BOUNDS = FracQParams(**FRAC_Q_BOUNDS_PDICT)


@jjit
def _get_qseq_means_and_covs_scalar(params, lgm):
    _res = _get_mean_u_params_qseq(params, lgm)
    mu_ms = jnp.array(_res[:4])
    mu_q = jnp.array(_res[4:])
    cov_ms = _get_cov_qseq_ms_block(params, lgm)
    cov_q = _get_cov_qseq_q_block(params, lgm)
    frac_q = _frac_quench_vs_lgm0(params, lgm)
    return mu_ms, cov_ms, mu_q, cov_q, frac_q


_M = (None, 0)
_get_qseq_means_and_covs_vmap = jjit(vmap(_get_qseq_means_and_covs_scalar, in_axes=_M))


@jjit
def _fun(x, ymin, ymax):
    return _sigmoid(x, LGM_X0, LGM_K, ymin, ymax)


@jjit
def _fun_Mcrit(x, ymin, ymax):
    return _sigmoid(x, 12.0, LGMCRIT_K, ymin, ymax)


@jjit
def _fun_chol_diag(x, ymin, ymax):
    _res = 10 ** _fun(x, ymin, ymax)
    return _res


@jjit
def _fun_fquench(x, x0, k, ymin, ymax):
    fquench = _sigmoid(x, x0, k, ymin, ymax)
    return fquench


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
def _get_mean_u_params_qseq_ms_block(params, lgm):
    ulgm = _sigmoid(
        lgm,
        params.mean_lgmhalo_x0,
        LGMCRIT_K,
        params.mean_ulgm_quench_ylo,
        params.mean_ulgm_quench_yhi,
    )
    ulgy = _sigmoid(
        lgm,
        params.mean_lgmhalo_x0,
        LGM_K,
        params.mean_ulgy_quench_ylo,
        params.mean_ulgy_quench_yhi,
    )
    ul = _sigmoid(
        lgm,
        params.mean_lgmhalo_x0,
        LGM_K,
        params.mean_ul_quench_ylo,
        params.mean_ul_quench_yhi,
    )
    utau = _sigmoid(
        lgm,
        params.mean_lgmhalo_x0,
        LGM_K,
        params.mean_utau_quench_ylo,
        params.mean_utau_quench_yhi,
    )

    return (ulgm, ulgy, ul, utau)


@jjit
def _get_mean_u_params_qseq_q_block(params, lgm):
    uqt = _sigmoid(
        lgm,
        params.mean_lgmhalo_x0,
        LGM_K,
        params.mean_uqt_quench_ylo,
        params.mean_uqt_quench_yhi,
    )
    uqs = _sigmoid(
        lgm,
        params.mean_lgmhalo_x0,
        LGM_K,
        params.mean_uqs_quench_ylo,
        params.mean_uqs_quench_yhi,
    )
    udrop = _sigmoid(
        lgm,
        params.mean_lgmhalo_x0,
        LGM_K,
        params.mean_udrop_quench_ylo,
        params.mean_udrop_quench_yhi,
    )
    urej = _sigmoid(
        lgm,
        params.mean_lgmhalo_x0,
        LGM_K,
        params.mean_urej_quench_ylo,
        params.mean_urej_quench_yhi,
    )
    return uqt, uqs, udrop, urej


@jjit
def _get_cov_params_qseq_ms_block(params, lgm):
    std_ulgm = _sigmoid(
        lgm,
        params.mean_lgmhalo_x0,
        LGM_K,
        params.std_ulgm_quench_ylo,
        params.std_ulgm_quench_yhi,
    )
    std_ulgy = _sigmoid(
        lgm,
        params.mean_lgmhalo_x0,
        LGM_K,
        params.std_ulgy_quench_ylo,
        params.std_ulgy_quench_yhi,
    )

    std_ul = _sigmoid(
        lgm,
        params.mean_lgmhalo_x0,
        LGM_K,
        params.std_ul_quench_ylo,
        params.std_ul_quench_yhi,
    )
    std_utau = _sigmoid(
        lgm,
        params.mean_lgmhalo_x0,
        LGM_K,
        params.std_utau_quench_ylo,
        params.std_utau_quench_yhi,
    )

    rho_ulgy_ulgm = _sigmoid(
        lgm,
        params.mean_lgmhalo_x0,
        LGM_K,
        params.rho_ulgy_ulgm_quench_ylo,
        params.rho_ulgy_ulgm_quench_yhi,
    )

    rho_ul_ulgm = _sigmoid(
        lgm,
        params.mean_lgmhalo_x0,
        LGM_K,
        params.rho_ul_ulgm_quench_ylo,
        params.rho_ul_ulgm_quench_yhi,
    )

    rho_ul_ulgy = _sigmoid(
        lgm,
        params.mean_lgmhalo_x0,
        LGM_K,
        params.rho_ul_ulgy_quench_ylo,
        params.rho_ul_ulgy_quench_yhi,
    )

    rho_utau_ulgm = _sigmoid(
        lgm,
        params.mean_lgmhalo_x0,
        LGM_K,
        params.rho_utau_ulgm_quench_ylo,
        params.rho_utau_ulgm_quench_yhi,
    )

    rho_utau_ulgy = _sigmoid(
        lgm,
        params.mean_lgmhalo_x0,
        LGM_K,
        params.rho_utau_ulgy_quench_ylo,
        params.rho_utau_ulgy_quench_yhi,
    )

    rho_utau_ul = _sigmoid(
        lgm,
        params.mean_lgmhalo_x0,
        LGM_K,
        params.rho_utau_ul_quench_ylo,
        params.rho_utau_ul_quench_yhi,
    )

    diags = std_ulgm, std_ulgy, std_ul, std_utau
    off_diags = (
        rho_ulgy_ulgm,
        rho_ul_ulgm,
        rho_ul_ulgy,
        rho_utau_ulgm,
        rho_utau_ulgy,
        rho_utau_ul,
    )
    return diags, off_diags


@jjit
def _get_cov_params_qseq_q_block(params, lgm):
    std_uqt = _sigmoid(
        lgm,
        params.mean_lgmhalo_x0,
        LGM_K,
        params.std_uqt_quench_ylo,
        params.std_uqt_quench_yhi,
    )

    std_uqs = _sigmoid(
        lgm,
        params.mean_lgmhalo_x0,
        LGM_K,
        params.std_uqs_quench_ylo,
        params.std_uqs_quench_yhi,
    )

    std_udrop = _sigmoid(
        lgm,
        params.mean_lgmhalo_x0,
        LGM_K,
        params.std_udrop_quench_ylo,
        params.std_udrop_quench_yhi,
    )

    std_urej = _sigmoid(
        lgm,
        params.mean_lgmhalo_x0,
        LGM_K,
        params.std_urej_quench_ylo,
        params.std_urej_quench_yhi,
    )
    diags = std_uqt, std_uqs, std_udrop, std_urej

    rho_uqs_uqt = _sigmoid(
        lgm,
        params.mean_lgmhalo_x0,
        LGM_K,
        params.rho_uqs_uqt_quench_ylo,
        params.rho_uqs_uqt_quench_yhi,
    )

    rho_udrop_uqs = _sigmoid(
        lgm,
        params.mean_lgmhalo_x0,
        LGM_K,
        params.rho_udrop_uqs_quench_ylo,
        params.rho_udrop_uqs_quench_yhi,
    )

    rho_udrop_uqt = _sigmoid(
        lgm,
        params.mean_lgmhalo_x0,
        LGM_K,
        params.rho_udrop_uqt_quench_ylo,
        params.rho_udrop_uqt_quench_yhi,
    )

    rho_urej_uqt = _sigmoid(
        lgm,
        params.mean_lgmhalo_x0,
        LGM_K,
        params.rho_urej_uqt_quench_ylo,
        params.rho_urej_uqt_quench_yhi,
    )

    rho_urej_uqs = _sigmoid(
        lgm,
        params.mean_lgmhalo_x0,
        LGM_K,
        params.rho_urej_uqs_quench_ylo,
        params.rho_urej_uqs_quench_yhi,
    )

    rho_urej_udrop = _sigmoid(
        lgm,
        params.mean_lgmhalo_x0,
        LGM_K,
        params.rho_urej_udrop_quench_ylo,
        params.rho_urej_udrop_quench_yhi,
    )
    off_diags = (
        rho_uqs_uqt,
        rho_udrop_uqt,
        rho_udrop_uqs,
        rho_urej_uqt,
        rho_urej_uqs,
        rho_urej_udrop,
    )

    return diags, off_diags


@jjit
def _get_covariance_qseq_q_block(params, lgm):
    diags, off_diags = _get_cov_params_qseq_q_block(params, lgm)
    ones = jnp.ones(len(diags))
    x = jnp.array((*ones, *off_diags))
    M = get_cholesky_from_params(x)
    corr_matrix = jnp.where(M == 0, M.T, M)
    cov_qseq_q_block = covariance_from_correlation(corr_matrix, jnp.array(diags))
    return cov_qseq_q_block


@jjit
def _get_covariance_qseq_ms_block(params, lgm):
    diags, off_diags = _get_cov_params_qseq_ms_block(params, lgm)
    ones = jnp.ones(len(diags))
    x = jnp.array((*ones, *off_diags))
    M = get_cholesky_from_params(x)
    corr_matrix = jnp.where(M == 0, M.T, M)
    cov_qseq_ms_block = covariance_from_correlation(corr_matrix, jnp.array(diags))
    return cov_qseq_ms_block


@jjit
def _get_chol_u_params_qseq_ms_block(params, lgm):
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

    return (
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


@jjit
def _get_chol_u_params_qseq_q_block(params, lgm):
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
    uqs_uqt = _fun(lgm, params.chol_uqs_uqt_quench_ylo, params.chol_uqs_uqt_quench_yhi)
    udrop_uqt = _fun(
        lgm, params.chol_udrop_uqt_quench_ylo, params.chol_udrop_uqt_quench_yhi
    )
    udrop_uqs = _fun(
        lgm, params.chol_udrop_uqs_quench_ylo, params.chol_udrop_uqs_quench_yhi
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
    return (
        uqt_uqt,
        uqs_uqs,
        udrop_udrop,
        urej_urej,
        uqs_uqt,
        udrop_uqt,
        udrop_uqs,
        urej_uqt,
        urej_uqs,
        urej_udrop,
    )


@jjit
def _get_cov_qseq_ms_block(params, lgm):
    chol_params_ms_block = _get_chol_u_params_qseq_ms_block(params, lgm)
    cov_qseq_ms_block = _get_cov_scalar(chol_params_ms_block)
    return cov_qseq_ms_block


@jjit
def _get_cov_qseq_q_block(params, lgm):
    chol_params_q_block = _get_chol_u_params_qseq_q_block(params, lgm)
    cov_qseq_q_block = _get_cov_scalar(chol_params_q_block)
    return cov_qseq_q_block


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
def _get_p_from_u_p_scalar(u_p, bounds):
    lo, hi = bounds
    p0 = 0.5 * (lo + hi)
    p = _sigmoid(u_p, p0, BOUNDING_K, lo, hi)
    return p


@jjit
def _get_u_p_from_p_scalar(p, bounds):
    lo, hi = bounds
    p0 = 0.5 * (lo + hi)
    u_p = _inverse_sigmoid(p, p0, BOUNDING_K, lo, hi)
    return u_p


_get_p_from_u_p_vmap = jjit(vmap(_get_p_from_u_p_scalar, in_axes=(0, 0)))
_get_u_p_from_p_vmap = jjit(vmap(_get_u_p_from_p_scalar, in_axes=(0, 0)))


@jjit
def get_bounded_qseq_params(u_params):
    u_params = jnp.array(
        [getattr(u_params, u_pname) for u_pname in QseqUParams._fields]
    )
    params = _get_p_from_u_p_vmap(
        jnp.array(u_params), jnp.array(SFH_PDF_QUENCH_PBOUNDS)
    )
    return QseqParams(*params)


def get_unbounded_qseq_params(params):
    params = jnp.array([getattr(params, pname) for pname in QseqParams._fields])
    u_params = _get_u_p_from_p_vmap(
        jnp.array(params), jnp.array(SFH_PDF_QUENCH_PBOUNDS)
    )
    return QseqUParams(*u_params)


@jjit
def get_bounded_qseq_massonly_params(u_params):
    fq_u_params = jnp.array([getattr(u_params, u_pname) for u_pname in _FRAC_Q_UPNAMES])
    fq_params = _get_p_from_u_p_vmap(fq_u_params, jnp.array(FRAC_Q_BOUNDS))
    params = QseqMassOnlyBlockParams(*u_params)
    params = params._replace(frac_quench_x0=fq_params[0])
    params = params._replace(frac_quench_k=fq_params[1])
    params = params._replace(frac_quench_ylo=fq_params[2])
    params = params._replace(frac_quench_yhi=fq_params[3])
    return params


@jjit
def get_unbounded_qseq_massonly_params(params):
    fq_params = jnp.array([getattr(params, pname) for pname in FRAC_Q_PNAMES])
    fq_u_params = _get_u_p_from_p_vmap(fq_params, jnp.array(FRAC_Q_BOUNDS))

    pnames = DEFAULT_SFH_PDF_QUENCH_BLOCK_PARAMS._fields
    params = jnp.array([getattr(params, pname) for pname in pnames])
    u_params = QseqMassOnlyBlockUParams(*params)
    u_params = u_params._replace(u_frac_quench_x0=fq_u_params[0])
    u_params = u_params._replace(u_frac_quench_k=fq_u_params[1])
    u_params = u_params._replace(u_frac_quench_ylo=fq_u_params[2])
    u_params = u_params._replace(u_frac_quench_yhi=fq_u_params[3])
    return u_params


DEFAULT_SFH_PDF_QUENCH_BLOCK_U_PARAMS = QseqMassOnlyBlockUParams(
    *get_unbounded_qseq_massonly_params(DEFAULT_SFH_PDF_QUENCH_BLOCK_PARAMS)
)

SFH_PDF_QUENCH_U_PARAMS = QseqUParams(*get_unbounded_qseq_params(SFH_PDF_QUENCH_PARAMS))
