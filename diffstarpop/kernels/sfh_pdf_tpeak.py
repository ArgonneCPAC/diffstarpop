"""Model of a quenched galaxy population calibrated to SMDPL halos."""

from collections import OrderedDict, namedtuple

from diffmah.utils import get_cholesky_from_params
from jax import jit as jjit
from jax import numpy as jnp
from jax import vmap

from ..utils import _inverse_sigmoid, _sigmoid, covariance_from_correlation

TODAY = 13.8
LGT0 = jnp.log10(TODAY)

LGM_X0, LGM_K = 12.5, 2.0
LGMCRIT_K = 4.0
BOUNDING_K = 0.1
RHO_BOUNDS = (-0.3, 0.3)

SFH_PDF_QUENCH_MU_PDICT = OrderedDict(
    mean_lgmhalo_x0=13.146,
    mean_ulgm_ms_ylo=11.871,
    mean_ulgm_ms_yhi=11.152,
    mean_ulgy_ms_ylo=-0.481,
    mean_ulgy_ms_yhi=2.322,
    mean_ul_ms_ylo=0.543,
    mean_ul_ms_yhi=-4.995,
    mean_utau_ms_ylo=9.437,
    mean_utau_ms_yhi=-6.089,
    mean_ulgm_quench_ylo=11.962,
    mean_ulgm_quench_yhi=12.074,
    mean_ulgy_quench_ylo=5.400,
    mean_ulgy_quench_yhi=-0.592,
    mean_ul_quench_ylo=-0.041,
    mean_ul_quench_yhi=-0.809,
    mean_utau_quench_ylo=11.339,
    mean_utau_quench_yhi=-4.601,
    mean_uqt_quench_ylo=1.010,
    mean_uqt_quench_yhi=0.186,
    mean_uqs_quench_ylo=-0.952,
    mean_uqs_quench_yhi=0.509,
    mean_udrop_quench_ylo=-2.971,
    mean_udrop_quench_yhi=-1.822,
    mean_urej_quench_ylo=0.008,
    mean_urej_quench_yhi=-8.250,
)
SFH_PDF_QUENCH_MU_BOUNDS_PDICT = OrderedDict(
    mean_lgmhalo_x0=(11.5, 13.5),
    mean_ulgm_ms_ylo=(11.0, 13.0),
    mean_ulgm_ms_yhi=(11.0, 13.0),
    mean_ulgy_ms_ylo=(-1.0, 1.5),
    mean_ulgy_ms_yhi=(-1.0, 2.5),
    mean_ul_ms_ylo=(-3.0, 5.0),
    mean_ul_ms_yhi=(-5.0, 2.5),
    mean_utau_ms_ylo=(-25.0, 50.0),
    mean_utau_ms_yhi=(-25.0, 50.0),
    mean_ulgm_quench_ylo=(11.5, 13.0),
    mean_ulgm_quench_yhi=(11.5, 13.0),
    mean_ulgy_quench_ylo=(0.0, 5.5),
    mean_ulgy_quench_yhi=(-2.0, 0.5),
    mean_ul_quench_ylo=(-1.0, 3.0),
    mean_ul_quench_yhi=(-10.0, 3.0),
    mean_utau_quench_ylo=(-25.0, 50.0),
    mean_utau_quench_yhi=(-25.0, 50.0),
    mean_uqt_quench_ylo=(0.0, 2.0),
    mean_uqt_quench_yhi=(0.0, 2.0),
    mean_uqs_quench_ylo=(-5.0, 2.0),
    mean_uqs_quench_yhi=(-5.0, 2.0),
    mean_udrop_quench_ylo=(-3.0, 2.0),
    mean_udrop_quench_yhi=(-3.0, 2.0),
    mean_urej_quench_ylo=(-10.0, 2.0),
    mean_urej_quench_yhi=(-10.0, 2.0),
)

SFH_PDF_QUENCH_COV_MS_BLOCK_PDICT = OrderedDict(
    std_ulgm_quench_ylo=0.257,
    std_ulgm_quench_yhi=0.249,
    std_ulgy_quench_ylo=0.316,
    std_ulgy_quench_yhi=0.120,
    std_ul_quench_ylo=0.368,
    std_ul_quench_yhi=0.259,
    std_utau_quench_ylo=1.752,
    std_utau_quench_yhi=2.324,
    rho_ulgy_ulgm_quench_ylo=0.028,
    rho_ulgy_ulgm_quench_yhi=-0.296,
    rho_ul_ulgm_quench_ylo=-0.160,
    rho_ul_ulgm_quench_yhi=-0.281,
    rho_ul_ulgy_quench_ylo=0.209,
    rho_ul_ulgy_quench_yhi=0.256,
    rho_utau_ulgm_quench_ylo=-0.278,
    rho_utau_ulgm_quench_yhi=0.293,
    rho_utau_ulgy_quench_ylo=0.286,
    rho_utau_ulgy_quench_yhi=0.296,
    rho_utau_ul_quench_ylo=-0.298,
    rho_utau_ul_quench_yhi=-0.106,
)
SFH_PDF_QUENCH_COV_MS_BLOCK_BOUNDS_PDICT = OrderedDict(
    std_ulgm_quench_ylo=(0.1, 1.0),
    std_ulgm_quench_yhi=(0.1, 1.0),
    std_ulgy_quench_ylo=(0.1, 1.0),
    std_ulgy_quench_yhi=(0.1, 1.0),
    std_ul_quench_ylo=(0.25, 3.0),
    std_ul_quench_yhi=(0.25, 3.0),
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
    std_uqt_quench_ylo=0.100,
    std_uqt_quench_yhi=0.063,
    std_uqs_quench_ylo=0.117,
    std_uqs_quench_yhi=0.803,
    std_udrop_quench_ylo=0.422,
    std_udrop_quench_yhi=0.692,
    std_urej_quench_ylo=0.427,
    std_urej_quench_yhi=0.944,
    rho_uqs_uqt_quench_ylo=0.221,
    rho_uqs_uqt_quench_yhi=0.178,
    rho_udrop_uqt_quench_ylo=0.011,
    rho_udrop_uqt_quench_yhi=-0.070,
    rho_udrop_uqs_quench_ylo=-0.039,
    rho_udrop_uqs_quench_yhi=0.107,
    rho_urej_uqt_quench_ylo=-0.214,
    rho_urej_uqt_quench_yhi=-0.142,
    rho_urej_uqs_quench_ylo=0.234,
    rho_urej_uqs_quench_yhi=0.238,
    rho_urej_udrop_quench_ylo=0.216,
    rho_urej_udrop_quench_yhi=0.259,
)
SFH_PDF_QUENCH_COV_Q_BLOCK_BOUNDS_PDICT = OrderedDict(
    std_uqt_quench_ylo=(0.01, 0.5),
    std_uqt_quench_yhi=(0.01, 0.5),
    std_uqs_quench_ylo=(0.01, 1.0),
    std_uqs_quench_yhi=(0.01, 1.0),
    std_udrop_quench_ylo=(0.01, 1.0),
    std_udrop_quench_yhi=(0.01, 1.0),
    std_urej_quench_ylo=(0.01, 1.0),
    std_urej_quench_yhi=(0.01, 1.0),
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
SFH_PDF_FRAC_QUENCH_PDICT = OrderedDict(
    frac_quench_x0=12.176,
    frac_quench_k=2.576,
    frac_quench_ylo=0.077,
    frac_quench_yhi=0.728,
)
SFH_PDF_FRAC_QUENCH_BOUNDS_PDICT = OrderedDict(
    frac_quench_x0=(11.0, 13.0),
    frac_quench_k=(1.0, 5.0),
    frac_quench_ylo=(0.0, 0.5),
    frac_quench_yhi=(0.5, 1.0),
)

SFH_PDF_QUENCH_PDICT = SFH_PDF_FRAC_QUENCH_PDICT.copy()
SFH_PDF_QUENCH_PDICT.update(SFH_PDF_QUENCH_MU_PDICT)
SFH_PDF_QUENCH_PDICT.update(SFH_PDF_QUENCH_COV_MS_BLOCK_PDICT)
SFH_PDF_QUENCH_PDICT.update(SFH_PDF_QUENCH_COV_Q_BLOCK_PDICT)

SFH_PDF_QUENCH_BOUNDS_PDICT = SFH_PDF_FRAC_QUENCH_BOUNDS_PDICT.copy()
SFH_PDF_QUENCH_BOUNDS_PDICT.update(SFH_PDF_QUENCH_MU_BOUNDS_PDICT)
SFH_PDF_QUENCH_BOUNDS_PDICT.update(SFH_PDF_QUENCH_COV_MS_BLOCK_BOUNDS_PDICT)
SFH_PDF_QUENCH_BOUNDS_PDICT.update(SFH_PDF_QUENCH_COV_Q_BLOCK_BOUNDS_PDICT)


QseqParams = namedtuple("QseqParams", list(SFH_PDF_QUENCH_PDICT.keys()))
SFH_PDF_QUENCH_PARAMS = QseqParams(**SFH_PDF_QUENCH_PDICT)
SFH_PDF_QUENCH_PBOUNDS = QseqParams(**SFH_PDF_QUENCH_BOUNDS_PDICT)

_UPNAMES = ["u_" + key for key in QseqParams._fields]
QseqUParams = namedtuple("QseqUParams", _UPNAMES)


@jjit
def _sfh_pdf_scalar_kernel(params, logmp0):
    frac_quench = _frac_quench_vs_logmp0(params, logmp0)

    mu_mseq = _get_mean_u_params_mseq(params, logmp0)

    mu_qseq_ms_block = _get_mean_u_params_qseq_ms_block(params, logmp0)
    cov_qseq_ms_block = _get_covariance_qseq_ms_block(params, logmp0)

    mu_qseq_q_block = _get_mean_u_params_qseq_q_block(params, logmp0)
    cov_qseq_q_block = _get_covariance_qseq_q_block(params, logmp0)

    return (
        frac_quench,
        mu_mseq,
        mu_qseq_ms_block,
        cov_qseq_ms_block,
        mu_qseq_q_block,
        cov_qseq_q_block,
    )


@jjit
def _get_mean_u_params_mseq(params, logmp0):
    ulgm = _sigmoid(
        logmp0,
        params.mean_lgmhalo_x0,
        LGMCRIT_K,
        params.mean_ulgm_ms_ylo,
        params.mean_ulgm_ms_yhi,
    )
    ulgy = _sigmoid(
        logmp0,
        params.mean_lgmhalo_x0,
        LGM_K,
        params.mean_ulgy_ms_ylo,
        params.mean_ulgy_ms_yhi,
    )
    ul = _sigmoid(
        logmp0,
        params.mean_lgmhalo_x0,
        LGM_K,
        params.mean_ul_ms_ylo,
        params.mean_ul_ms_yhi,
    )
    utau = _sigmoid(
        logmp0,
        params.mean_lgmhalo_x0,
        LGM_K,
        params.mean_utau_ms_ylo,
        params.mean_utau_ms_yhi,
    )

    return (ulgm, ulgy, ul, utau)


@jjit
def _get_mean_u_params_qseq_ms_block(params, logmp0):
    ulgm = _sigmoid(
        logmp0,
        params.mean_lgmhalo_x0,
        LGMCRIT_K,
        params.mean_ulgm_quench_ylo,
        params.mean_ulgm_quench_yhi,
    )
    ulgy = _sigmoid(
        logmp0,
        params.mean_lgmhalo_x0,
        LGM_K,
        params.mean_ulgy_quench_ylo,
        params.mean_ulgy_quench_yhi,
    )
    ul = _sigmoid(
        logmp0,
        params.mean_lgmhalo_x0,
        LGM_K,
        params.mean_ul_quench_ylo,
        params.mean_ul_quench_yhi,
    )
    utau = _sigmoid(
        logmp0,
        params.mean_lgmhalo_x0,
        LGM_K,
        params.mean_utau_quench_ylo,
        params.mean_utau_quench_yhi,
    )

    return (ulgm, ulgy, ul, utau)


@jjit
def _get_mean_u_params_qseq_q_block(params, logmp0):
    uqt = _sigmoid(
        logmp0,
        params.mean_lgmhalo_x0,
        LGM_K,
        params.mean_uqt_quench_ylo,
        params.mean_uqt_quench_yhi,
    )
    uqs = _sigmoid(
        logmp0,
        params.mean_lgmhalo_x0,
        LGM_K,
        params.mean_uqs_quench_ylo,
        params.mean_uqs_quench_yhi,
    )
    udrop = _sigmoid(
        logmp0,
        params.mean_lgmhalo_x0,
        LGM_K,
        params.mean_udrop_quench_ylo,
        params.mean_udrop_quench_yhi,
    )
    urej = _sigmoid(
        logmp0,
        params.mean_lgmhalo_x0,
        LGM_K,
        params.mean_urej_quench_ylo,
        params.mean_urej_quench_yhi,
    )
    return uqt, uqs, udrop, urej


@jjit
def _get_cov_params_qseq_ms_block(params, logmp0):
    std_ulgm = _sigmoid(
        logmp0,
        params.mean_lgmhalo_x0,
        LGM_K,
        params.std_ulgm_quench_ylo,
        params.std_ulgm_quench_yhi,
    )
    std_ulgy = _sigmoid(
        logmp0,
        params.mean_lgmhalo_x0,
        LGM_K,
        params.std_ulgy_quench_ylo,
        params.std_ulgy_quench_yhi,
    )

    std_ul = _sigmoid(
        logmp0,
        params.mean_lgmhalo_x0,
        LGM_K,
        params.std_ul_quench_ylo,
        params.std_ul_quench_yhi,
    )
    std_utau = _sigmoid(
        logmp0,
        params.mean_lgmhalo_x0,
        LGM_K,
        params.std_utau_quench_ylo,
        params.std_utau_quench_yhi,
    )

    rho_ulgy_ulgm = _sigmoid(
        logmp0,
        params.mean_lgmhalo_x0,
        LGM_K,
        params.rho_ulgy_ulgm_quench_ylo,
        params.rho_ulgy_ulgm_quench_yhi,
    )

    rho_ul_ulgm = _sigmoid(
        logmp0,
        params.mean_lgmhalo_x0,
        LGM_K,
        params.rho_ul_ulgm_quench_ylo,
        params.rho_ul_ulgm_quench_yhi,
    )

    rho_ul_ulgy = _sigmoid(
        logmp0,
        params.mean_lgmhalo_x0,
        LGM_K,
        params.rho_ul_ulgy_quench_ylo,
        params.rho_ul_ulgy_quench_yhi,
    )

    rho_utau_ulgm = _sigmoid(
        logmp0,
        params.mean_lgmhalo_x0,
        LGM_K,
        params.rho_utau_ulgm_quench_ylo,
        params.rho_utau_ulgm_quench_yhi,
    )

    rho_utau_ulgy = _sigmoid(
        logmp0,
        params.mean_lgmhalo_x0,
        LGM_K,
        params.rho_utau_ulgy_quench_ylo,
        params.rho_utau_ulgy_quench_yhi,
    )

    rho_utau_ul = _sigmoid(
        logmp0,
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
def _get_cov_params_qseq_q_block(params, logmp0):
    std_uqt = _sigmoid(
        logmp0,
        params.mean_lgmhalo_x0,
        LGM_K,
        params.std_uqt_quench_ylo,
        params.std_uqt_quench_yhi,
    )

    std_uqs = _sigmoid(
        logmp0,
        params.mean_lgmhalo_x0,
        LGM_K,
        params.std_uqs_quench_ylo,
        params.std_uqs_quench_yhi,
    )

    std_udrop = _sigmoid(
        logmp0,
        params.mean_lgmhalo_x0,
        LGM_K,
        params.std_udrop_quench_ylo,
        params.std_udrop_quench_yhi,
    )

    std_urej = _sigmoid(
        logmp0,
        params.mean_lgmhalo_x0,
        LGM_K,
        params.std_urej_quench_ylo,
        params.std_urej_quench_yhi,
    )
    diags = std_uqt, std_uqs, std_udrop, std_urej

    rho_uqs_uqt = _sigmoid(
        logmp0,
        params.mean_lgmhalo_x0,
        LGM_K,
        params.rho_uqs_uqt_quench_ylo,
        params.rho_uqs_uqt_quench_yhi,
    )

    rho_udrop_uqs = _sigmoid(
        logmp0,
        params.mean_lgmhalo_x0,
        LGM_K,
        params.rho_udrop_uqs_quench_ylo,
        params.rho_udrop_uqs_quench_yhi,
    )

    rho_udrop_uqt = _sigmoid(
        logmp0,
        params.mean_lgmhalo_x0,
        LGM_K,
        params.rho_udrop_uqt_quench_ylo,
        params.rho_udrop_uqt_quench_yhi,
    )

    rho_urej_uqt = _sigmoid(
        logmp0,
        params.mean_lgmhalo_x0,
        LGM_K,
        params.rho_urej_uqt_quench_ylo,
        params.rho_urej_uqt_quench_yhi,
    )

    rho_urej_uqs = _sigmoid(
        logmp0,
        params.mean_lgmhalo_x0,
        LGM_K,
        params.rho_urej_uqs_quench_ylo,
        params.rho_urej_uqs_quench_yhi,
    )

    rho_urej_udrop = _sigmoid(
        logmp0,
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
