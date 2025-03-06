"""Model of a quenched galaxy population calibrated to SMDPL halos."""

from collections import OrderedDict, namedtuple

from diffmah.utils import get_cholesky_from_params
from jax import jit as jjit
from jax import numpy as jnp
from jax import vmap

from ..utils import (
    _inverse_sigmoid,
    _sigmoid,
    covariance_from_correlation,
    smoothly_clipped_line,
)


TODAY = 13.8
LGT0 = jnp.log10(TODAY)

LGM_X0 = 12.5
LGM_K = 2.0
LGMCRIT_K = 4.0
BOUNDING_K = 0.1
RHO_BOUNDS = (-0.3, 0.3)

SFH_PDF_QUENCH_MU_PDICT = OrderedDict(
    mean_ulgm_int=12.16,
    mean_ulgm_slp=0.11,
    mean_ulgy_int=-0.04,
    mean_ulgy_slp=-0.17,
    mean_ul_int=0.09,
    mean_ul_slp=0.22,
    mean_utau_int=1.36,
    mean_utau_slp=-12.57,
    mean_uqt_int=0.93,
    mean_uqt_slp=-0.19,
    mean_uqs_int=-0.36,
    mean_uqs_slp=0.68,
    mean_udrop_int=-1.84,
    mean_udrop_slp=-0.44,
    mean_urej_int=0.02,
    mean_urej_slp=-1.09,
)
SFH_PDF_QUENCH_MU_BOUNDS_PDICT = OrderedDict(
    mean_ulgm_int=(11.0, 13.0),
    mean_ulgm_slp=(-20.0, 20.0),
    mean_ulgy_int=(-1.0, 3.5),
    mean_ulgy_slp=(-20.0, 20.0),
    mean_ul_int=(-3.0, 5.0),
    mean_ul_slp=(-20.0, 20.0),
    mean_utau_int=(-25.0, 50.0),
    mean_utau_slp=(-20.0, 20.0),
    mean_uqt_int=(0.0, 2.0),
    mean_uqt_slp=(-20.0, 20.0),
    mean_uqs_int=(-5.0, 2.0),
    mean_uqs_slp=(-20.0, 20.0),
    mean_udrop_int=(-3.0, 2.0),
    mean_udrop_slp=(-20.0, 20.0),
    mean_urej_int=(-10.0, 2.0),
    mean_urej_slp=(-20.0, 20.0),
)

SFH_PDF_QUENCH_COV_MS_BLOCK_PDICT = OrderedDict(
    std_ulgm_int=0.22,
    std_ulgm_slp=0.11,
    std_ulgy_int=0.32,
    std_ulgy_slp=0.03,
    std_ul_int=0.32,
    std_ul_slp=0.10,
    std_utau_int=6.85,
    std_utau_slp=0.96,
    rho_ulgy_ulgm_int=0.001,
    rho_ulgy_ulgm_slp=0.001,
    rho_ul_ulgm_int=0.001,
    rho_ul_ulgm_slp=0.001,
    rho_ul_ulgy_int=0.001,
    rho_ul_ulgy_slp=0.001,
    rho_utau_ulgm_int=0.001,
    rho_utau_ulgm_slp=0.001,
    rho_utau_ulgy_int=0.001,
    rho_utau_ulgy_slp=0.001,
    rho_utau_ul_int=0.001,
    rho_utau_ul_slp=0.001,
)
SFH_PDF_QUENCH_COV_MS_BLOCK_BOUNDS_PDICT = OrderedDict(
    std_ulgm_int=(0.01, 1.0),
    std_ulgm_slp=(-1.00, 1.0),
    std_ulgy_int=(0.01, 1.0),
    std_ulgy_slp=(-1.00, 1.0),
    std_ul_int=(0.01, 1.0),
    std_ul_slp=(-1.00, 1.0),
    std_utau_int=(1.00, 12.0),
    std_utau_slp=(-3.00, 3.0),
    rho_ulgy_ulgm_int=(-20.0, 20.0),
    rho_ulgy_ulgm_slp=(-20.0, 20.0),
    rho_ul_ulgm_int=(-20.0, 20.0),
    rho_ul_ulgm_slp=(-20.0, 20.0),
    rho_ul_ulgy_int=(-20.0, 20.0),
    rho_ul_ulgy_slp=(-20.0, 20.0),
    rho_utau_ulgm_int=(-20.0, 20.0),
    rho_utau_ulgm_slp=(-20.0, 20.0),
    rho_utau_ulgy_int=(-20.0, 20.0),
    rho_utau_ulgy_slp=(-20.0, 20.0),
    rho_utau_ul_int=(-20.0, 20.0),
    rho_utau_ul_slp=(-20.0, 20.0),
)

SFH_PDF_QUENCH_COV_Q_BLOCK_PDICT = OrderedDict(
    std_uqt_int=0.09,
    std_uqt_slp=0.02,
    std_uqs_int=0.55,
    std_uqs_slp=0.02,
    std_udrop_int=0.73,
    std_udrop_slp=-0.07,
    std_urej_int=1.12,
    std_urej_slp=-0.26,
    rho_uqs_uqt_int=0.001,
    rho_uqs_uqt_slp=0.001,
    rho_udrop_uqt_int=0.001,
    rho_udrop_uqt_slp=0.001,
    rho_udrop_uqs_int=0.001,
    rho_udrop_uqs_slp=0.001,
    rho_urej_uqt_int=0.001,
    rho_urej_uqt_slp=0.001,
    rho_urej_uqs_int=0.001,
    rho_urej_uqs_slp=0.001,
    rho_urej_udrop_int=0.001,
    rho_urej_udrop_slp=0.001,
)
SFH_PDF_QUENCH_COV_Q_BLOCK_BOUNDS_PDICT = OrderedDict(
    std_uqt_int=(0.01, 0.5),
    std_uqt_slp=(-1.00, 1.0),
    std_uqs_int=(0.01, 1.0),
    std_uqs_slp=(-1.00, 1.0),
    std_udrop_int=(0.01, 2.0),
    std_udrop_slp=(-1.00, 1.0),
    std_urej_int=(0.01, 2.0),
    std_urej_slp=(-1.00, 1.0),
    rho_uqs_uqt_int=(-20.0, 20.0),
    rho_uqs_uqt_slp=(-20.0, 20.0),
    rho_udrop_uqt_int=(-20.0, 20.0),
    rho_udrop_uqt_slp=(-20.0, 20.0),
    rho_udrop_uqs_int=(-20.0, 20.0),
    rho_udrop_uqs_slp=(-20.0, 20.0),
    rho_urej_uqt_int=(-20.0, 20.0),
    rho_urej_uqt_slp=(-20.0, 20.0),
    rho_urej_uqs_int=(-20.0, 20.0),
    rho_urej_uqs_slp=(-20.0, 20.0),
    rho_urej_udrop_int=(-20.0, 20.0),
    rho_urej_udrop_slp=(-20.0, 20.0),
)
SFH_PDF_FRAC_QUENCH_PDICT = OrderedDict(
    frac_quench_x0=12.07,
    frac_quench_k=2.87,
    frac_quench_ylo=0.01,
    frac_quench_yhi=0.99,
)
SFH_PDF_FRAC_QUENCH_BOUNDS_PDICT = OrderedDict(
    frac_quench_x0=(10.0, 13.0),
    frac_quench_k=(0.01, 5.0),
    frac_quench_ylo=(0.0, 0.5),
    frac_quench_yhi=(0.5, 1.0),
)

BOUNDING_MEAN_VALS_PDICT = OrderedDict(
    mean_ulgm=(11.0, 13.0),
    mean_ulgy=(-1.0, 3.5),
    mean_ul=(-3.0, 5.0),
    mean_utau=(-25.0, 50.0),
    mean_uqt=(0.0, 2.0),
    mean_uqs=(-5.0, 2.0),
    mean_udrop=(-3.0, 2.0),
    mean_urej=(-10.0, 2.0),
)

BOUNDING_STD_VALS_PDICT = OrderedDict(
    std_ulgm=(0.01, 1.0),
    std_ulgy=(0.01, 1.0),
    std_ul=(0.01, 1.0),
    std_utau=(1.00, 12.0),
    std_uqt=(0.01, 0.5),
    std_uqs=(0.01, 1.0),
    std_udrop=(0.01, 2.0),
    std_urej=(0.01, 2.0),
)

BOUNDING_RHO_VALS_PDICT = OrderedDict(
    rho_ulgy_ulgm=RHO_BOUNDS,
    rho_ul_ulgm=RHO_BOUNDS,
    rho_ul_ulgy=RHO_BOUNDS,
    rho_utau_ulgm=RHO_BOUNDS,
    rho_utau_ulgy=RHO_BOUNDS,
    rho_utau_ul=RHO_BOUNDS,
    rho_uqs_uqt=RHO_BOUNDS,
    rho_udrop_uqt=RHO_BOUNDS,
    rho_udrop_uqs=RHO_BOUNDS,
    rho_urej_uqt=RHO_BOUNDS,
    rho_urej_uqs=RHO_BOUNDS,
    rho_urej_udrop=RHO_BOUNDS,
)

SFH_PDF_QUENCH_PDICT = SFH_PDF_FRAC_QUENCH_PDICT.copy()
SFH_PDF_QUENCH_PDICT.update(SFH_PDF_QUENCH_MU_PDICT)
SFH_PDF_QUENCH_PDICT.update(SFH_PDF_QUENCH_COV_MS_BLOCK_PDICT)
SFH_PDF_QUENCH_PDICT.update(SFH_PDF_QUENCH_COV_Q_BLOCK_PDICT)

SFH_PDF_QUENCH_BOUNDS_PDICT = SFH_PDF_FRAC_QUENCH_BOUNDS_PDICT.copy()
SFH_PDF_QUENCH_BOUNDS_PDICT.update(SFH_PDF_QUENCH_MU_BOUNDS_PDICT)
SFH_PDF_QUENCH_BOUNDS_PDICT.update(SFH_PDF_QUENCH_COV_MS_BLOCK_BOUNDS_PDICT)
SFH_PDF_QUENCH_BOUNDS_PDICT.update(SFH_PDF_QUENCH_COV_Q_BLOCK_BOUNDS_PDICT)

BOUNDING_VALS_PDICT = BOUNDING_MEAN_VALS_PDICT.copy()
BOUNDING_VALS_PDICT.update(BOUNDING_STD_VALS_PDICT.copy())
BOUNDING_VALS_PDICT.update(BOUNDING_RHO_VALS_PDICT.copy())

QseqParams = namedtuple("QseqParams", list(SFH_PDF_QUENCH_PDICT.keys()))
SFH_PDF_QUENCH_PARAMS = QseqParams(**SFH_PDF_QUENCH_PDICT)
SFH_PDF_QUENCH_PBOUNDS = QseqParams(**SFH_PDF_QUENCH_BOUNDS_PDICT)

_UPNAMES = ["u_" + key for key in QseqParams._fields]
QseqUParams = namedtuple("QseqUParams", _UPNAMES)

BoundingParams = namedtuple("BoundingParams", list(BOUNDING_VALS_PDICT.keys()))
BOUNDING_VALS = BoundingParams(**BOUNDING_VALS_PDICT)


def line_model(x, y0, m, y_lo, y_hi):
    return smoothly_clipped_line(x, LGM_X0, y0, m, y_lo, y_hi)


@jjit
def _sfh_pdf_scalar_kernel(params, logmp0):
    frac_quench = _frac_quench_vs_logmp0(params, logmp0)

    mu = _get_mean_u_params(params, logmp0)
    cov_qseq_ms_block = _get_covariance_qseq_ms_block(params, logmp0)
    cov_qseq_q_block = _get_covariance_qseq_q_block(params, logmp0)

    return (
        frac_quench,
        mu,
        cov_qseq_ms_block,
        cov_qseq_q_block,
    )


@jjit
def _get_mean_u_params(params, logmp0):

    ulgm = line_model(
        logmp0,
        params.mean_ulgm_int,
        params.mean_ulgm_slp,
        *BOUNDING_VALS.mean_ulgm,
    )

    ulgy = line_model(
        logmp0,
        params.mean_ulgy_int,
        params.mean_ulgy_slp,
        *BOUNDING_VALS.mean_ulgy,
    )

    ul = line_model(
        logmp0,
        params.mean_ul_int,
        params.mean_ul_slp,
        *BOUNDING_VALS.mean_ul,
    )

    utau = line_model(
        logmp0,
        params.mean_utau_int,
        params.mean_utau_slp,
        *BOUNDING_VALS.mean_utau,
    )

    uqt = line_model(
        logmp0,
        params.mean_uqt_int,
        params.mean_uqt_slp,
        *BOUNDING_VALS.mean_uqt,
    )

    uqs = line_model(
        logmp0,
        params.mean_uqs_int,
        params.mean_uqs_slp,
        *BOUNDING_VALS.mean_uqs,
    )

    udrop = line_model(
        logmp0,
        params.mean_udrop_int,
        params.mean_udrop_slp,
        *BOUNDING_VALS.mean_udrop,
    )

    urej = line_model(
        logmp0,
        params.mean_urej_int,
        params.mean_urej_slp,
        *BOUNDING_VALS.mean_urej,
    )
    return (ulgm, ulgy, ul, utau, uqt, uqs, udrop, urej)


@jjit
def _get_cov_params_qseq_ms_block(params, logmp0):

    std_ulgm = line_model(
        logmp0,
        params.std_ulgm_int,
        params.std_ulgm_slp,
        *BOUNDING_VALS.std_ulgm,
    )

    std_ulgy = line_model(
        logmp0,
        params.std_ulgy_int,
        params.std_ulgy_slp,
        *BOUNDING_VALS.std_ulgy,
    )

    std_ul = line_model(
        logmp0,
        params.std_ul_int,
        params.std_ul_slp,
        *BOUNDING_VALS.std_ul,
    )

    std_utau = line_model(
        logmp0,
        params.std_utau_int,
        params.std_utau_slp,
        *BOUNDING_VALS.std_utau,
    )

    rho_ulgy_ulgm = line_model(
        logmp0,
        params.rho_ulgy_ulgm_int,
        params.rho_ulgy_ulgm_slp,
        *BOUNDING_VALS.rho_ulgy_ulgm,
    )

    rho_ul_ulgm = line_model(
        logmp0,
        params.rho_ul_ulgm_int,
        params.rho_ul_ulgm_slp,
        *BOUNDING_VALS.rho_ul_ulgm,
    )

    rho_ul_ulgy = line_model(
        logmp0,
        params.rho_ul_ulgy_int,
        params.rho_ul_ulgy_slp,
        *BOUNDING_VALS.rho_ul_ulgy,
    )

    rho_utau_ulgm = line_model(
        logmp0,
        params.rho_utau_ulgm_int,
        params.rho_utau_ulgm_slp,
        *BOUNDING_VALS.rho_utau_ulgm,
    )

    rho_utau_ulgy = line_model(
        logmp0,
        params.rho_utau_ulgy_int,
        params.rho_utau_ulgy_slp,
        *BOUNDING_VALS.rho_utau_ulgy,
    )

    rho_utau_ul = line_model(
        logmp0,
        params.rho_utau_ul_int,
        params.rho_utau_ul_slp,
        *BOUNDING_VALS.rho_utau_ul,
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
def _get_cov_params_qseq_q_block(params, logmp0):

    std_uqt = line_model(
        logmp0,
        params.std_uqt_int,
        params.std_uqt_slp,
        *BOUNDING_VALS.std_uqt,
    )

    std_uqs = line_model(
        logmp0,
        params.std_uqs_int,
        params.std_uqs_slp,
        *BOUNDING_VALS.std_uqs,
    )

    std_udrop = line_model(
        logmp0,
        params.std_udrop_int,
        params.std_udrop_slp,
        *BOUNDING_VALS.std_udrop,
    )

    std_urej = line_model(
        logmp0,
        params.std_urej_int,
        params.std_urej_slp,
        *BOUNDING_VALS.std_urej,
    )

    rho_uqs_uqt = line_model(
        logmp0,
        params.rho_uqs_uqt_int,
        params.rho_uqs_uqt_slp,
        *BOUNDING_VALS.rho_uqs_uqt,
    )

    rho_udrop_uqt = line_model(
        logmp0,
        params.rho_udrop_uqt_int,
        params.rho_udrop_uqt_slp,
        *BOUNDING_VALS.rho_udrop_uqt,
    )

    rho_udrop_uqs = line_model(
        logmp0,
        params.rho_udrop_uqs_int,
        params.rho_udrop_uqs_slp,
        *BOUNDING_VALS.rho_udrop_uqs,
    )

    rho_urej_uqt = line_model(
        logmp0,
        params.rho_urej_uqt_int,
        params.rho_urej_uqt_slp,
        *BOUNDING_VALS.rho_urej_uqt,
    )

    rho_urej_uqs = line_model(
        logmp0,
        params.rho_urej_uqs_int,
        params.rho_urej_uqs_slp,
        *BOUNDING_VALS.rho_urej_uqs,
    )

    rho_urej_udrop = line_model(
        logmp0,
        params.rho_urej_udrop_int,
        params.rho_urej_udrop_slp,
        *BOUNDING_VALS.rho_urej_udrop,
    )

    diags = std_uqt, std_uqs, std_udrop, std_urej
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
def _get_covariance_qseq_q_block(params, logmp0):
    diags, off_diags = _get_cov_params_qseq_q_block(params, logmp0)
    ones = jnp.ones(len(diags))
    x = jnp.array((*ones, *off_diags))
    M = get_cholesky_from_params(x)
    corr_matrix = jnp.where(M == 0, M.T, M)
    cov_qseq_q_block = covariance_from_correlation(corr_matrix, jnp.array(diags))
    return cov_qseq_q_block


@jjit
def _get_covariance_qseq_ms_block(params, logmp0):
    diags, off_diags = _get_cov_params_qseq_ms_block(params, logmp0)
    ones = jnp.ones(len(diags))
    x = jnp.array((*ones, *off_diags))
    M = get_cholesky_from_params(x)
    corr_matrix = jnp.where(M == 0, M.T, M)
    cov_qseq_ms_block = covariance_from_correlation(corr_matrix, jnp.array(diags))
    return cov_qseq_ms_block


@jjit
def _frac_quench_vs_logmp0(params, logmp0):
    frac_q = _sigmoid(
        logmp0,
        params.frac_quench_x0,
        params.frac_quench_k,
        params.frac_quench_ylo,
        params.frac_quench_yhi,
    )
    return frac_q


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
def get_bounded_sfh_pdf_params(u_params):
    u_params = jnp.array(
        [getattr(u_params, u_pname) for u_pname in QseqUParams._fields]
    )
    params = _get_p_from_u_p_vmap(
        jnp.array(u_params), jnp.array(SFH_PDF_QUENCH_PBOUNDS)
    )
    return QseqParams(*params)


def get_unbounded_sfh_pdf_params(params):
    params = jnp.array([getattr(params, pname) for pname in QseqParams._fields])
    u_params = _get_u_p_from_p_vmap(
        jnp.array(params), jnp.array(SFH_PDF_QUENCH_PBOUNDS)
    )
    return QseqUParams(*u_params)


SFH_PDF_QUENCH_U_PARAMS = QseqUParams(
    *get_unbounded_sfh_pdf_params(SFH_PDF_QUENCH_PARAMS)
)
