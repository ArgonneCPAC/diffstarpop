"""Model of a quenched galaxy population calibrated to SMDPL halos."""

from collections import OrderedDict, namedtuple

from diffmah.utils import get_cholesky_from_params
from jax import jit as jjit
from jax import numpy as jnp
from jax import vmap

from ..utils import _inverse_sigmoid, _sigmoid, covariance_from_correlation

TODAY = 13.8
LGT0 = jnp.log10(TODAY)


K = 0.5 # K=0.5 makes the sigmoid approximately a line.
X0 = 12.5 # With K=0.5 the precise value of X0 is not too important. It can be held fixed.
LGMCRIT_K = 4.0 # With LGMCRIT_K the parameter lgm has an evolution that quickly flattens after some value.
BOUNDING_K = 0.1
RHO_BOUNDS = (-0.3, 0.3)

SFH_PDF_QUENCH_MU_PDICT = OrderedDict(
    mean_ulgm_ms_ylo=11.92,
    mean_ulgm_ms_yhi=11.40,
    mean_ulgy_ms_ylo=-0.37,
    mean_ulgy_ms_yhi=1.87,
    mean_ul_ms_ylo=-1.86,
    mean_ul_ms_yhi=-3.77,
    mean_utau_ms_ylo=11.40,
    mean_utau_ms_yhi=-4.30,
    mean_ulgm_quench_ylo=12.06,
    mean_ulgm_quench_yhi=12.23,
    mean_ulgy_quench_ylo=4.09,
    mean_ulgy_quench_yhi=-1.41,
    mean_ul_quench_ylo=0.03,
    mean_ul_quench_yhi=-6.87,
    mean_utau_quench_ylo=16.78,
    mean_utau_quench_yhi=-5.58,
    mean_uqt_quench_ylo=1.3,
    mean_uqt_quench_yhi=0.5,
    mean_uqs_quench_ylo=-0.5,
    mean_uqs_quench_yhi=0.5,
    mean_udrop_quench_ylo=-2.0,
    mean_udrop_quench_yhi=-2.0,
    mean_urej_quench_ylo=0.5,
    mean_urej_quench_yhi=-2.5,
)
SFH_PDF_QUENCH_MU_BOUNDS_PDICT = OrderedDict(
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
    frac_quench_x0=12.01,
    frac_quench_k=1.75,
    frac_quench_ylo=-0.63,
    frac_quench_yhi=0.99,
)
SFH_PDF_FRAC_QUENCH_BOUNDS_PDICT = OrderedDict(
    frac_quench_x0=(11.0, 13.0),
    frac_quench_k=(1.0, 5.0),
    frac_quench_ylo=(-1.0, 0.5),
    frac_quench_yhi=(0.5, 1.5),
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
def _sfh_pdf_scalar_kernel(params, lgm):
    frac_quench = _frac_quench_vs_lgm0(params, lgm)

    mu_mseq = _get_mean_u_params_mseq(params, lgm)

    mu_qseq_ms_block = _get_mean_u_params_qseq_ms_block(params, lgm)
    cov_qseq_ms_block = _get_covariance_qseq_ms_block(params, lgm)

    mu_qseq_q_block = _get_mean_u_params_qseq_q_block(params, lgm)
    cov_qseq_q_block = _get_covariance_qseq_q_block(params, lgm)

    return (
        frac_quench,
        mu_mseq,
        mu_qseq_ms_block,
        cov_qseq_ms_block,
        mu_qseq_q_block,
        cov_qseq_q_block,
    )


@jjit
def _get_mean_u_params_mseq(params, lgm):
    ulgm = _sigmoid(
        lgm,
        X0,
        LGMCRIT_K,
        params.mean_ulgm_ms_ylo,
        params.mean_ulgm_ms_yhi,
    )
    ulgy = _sigmoid(
        lgm,
        X0,
        K,
        params.mean_ulgy_ms_ylo,
        params.mean_ulgy_ms_yhi,
    )
    ul = _sigmoid(
        lgm,
        X0,
        K,
        params.mean_ul_ms_ylo,
        params.mean_ul_ms_yhi,
    )
    utau = _sigmoid(
        lgm,
        X0,
        K,
        params.mean_utau_ms_ylo,
        params.mean_utau_ms_yhi,
    )

    return (ulgm, ulgy, ul, utau)


@jjit
def _get_mean_u_params_qseq_ms_block(params, lgm):
    ulgm = _sigmoid(
        lgm,
        X0,
        LGMCRIT_K,
        params.mean_ulgm_quench_ylo,
        params.mean_ulgm_quench_yhi,
    )
    ulgy = _sigmoid(
        lgm,
        X0,
        K,
        params.mean_ulgy_quench_ylo,
        params.mean_ulgy_quench_yhi,
    )
    ul = _sigmoid(
        lgm,
        X0,
        K,
        params.mean_ul_quench_ylo,
        params.mean_ul_quench_yhi,
    )
    utau = _sigmoid(
        lgm,
        X0,
        K,
        params.mean_utau_quench_ylo,
        params.mean_utau_quench_yhi,
    )

    return (ulgm, ulgy, ul, utau)


@jjit
def _get_mean_u_params_qseq_q_block(params, lgm):
    uqt = _sigmoid(
        lgm,
        X0,
        K,
        params.mean_uqt_quench_ylo,
        params.mean_uqt_quench_yhi,
    )
    uqs = _sigmoid(
        lgm,
        X0,
        K,
        params.mean_uqs_quench_ylo,
        params.mean_uqs_quench_yhi,
    )
    udrop = _sigmoid(
        lgm,
        X0,
        K,
        params.mean_udrop_quench_ylo,
        params.mean_udrop_quench_yhi,
    )
    urej = _sigmoid(
        lgm,
        X0,
        K,
        params.mean_urej_quench_ylo,
        params.mean_urej_quench_yhi,
    )
    return uqt, uqs, udrop, urej


@jjit
def _get_cov_params_qseq_ms_block(params, lgm):
    std_ulgm = _sigmoid(
        lgm,
        X0,
        K,
        params.std_ulgm_quench_ylo,
        params.std_ulgm_quench_yhi,
    )
    std_ulgy = _sigmoid(
        lgm,
        X0,
        K,
        params.std_ulgy_quench_ylo,
        params.std_ulgy_quench_yhi,
    )

    std_ul = _sigmoid(
        lgm,
        X0,
        K,
        params.std_ul_quench_ylo,
        params.std_ul_quench_yhi,
    )
    std_utau = _sigmoid(
        lgm,
        X0,
        K,
        params.std_utau_quench_ylo,
        params.std_utau_quench_yhi,
    )

    rho_ulgy_ulgm = _sigmoid(
        lgm,
        X0,
        K,
        params.rho_ulgy_ulgm_quench_ylo,
        params.rho_ulgy_ulgm_quench_yhi,
    )

    rho_ul_ulgm = _sigmoid(
        lgm,
        X0,
        K,
        params.rho_ul_ulgm_quench_ylo,
        params.rho_ul_ulgm_quench_yhi,
    )

    rho_ul_ulgy = _sigmoid(
        lgm,
        X0,
        K,
        params.rho_ul_ulgy_quench_ylo,
        params.rho_ul_ulgy_quench_yhi,
    )

    rho_utau_ulgm = _sigmoid(
        lgm,
        X0,
        K,
        params.rho_utau_ulgm_quench_ylo,
        params.rho_utau_ulgm_quench_yhi,
    )

    rho_utau_ulgy = _sigmoid(
        lgm,
        X0,
        K,
        params.rho_utau_ulgy_quench_ylo,
        params.rho_utau_ulgy_quench_yhi,
    )

    rho_utau_ul = _sigmoid(
        lgm,
        X0,
        K,
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
        X0,
        K,
        params.std_uqt_quench_ylo,
        params.std_uqt_quench_yhi,
    )

    std_uqs = _sigmoid(
        lgm,
        X0,
        K,
        params.std_uqs_quench_ylo,
        params.std_uqs_quench_yhi,
    )

    std_udrop = _sigmoid(
        lgm,
        X0,
        K,
        params.std_udrop_quench_ylo,
        params.std_udrop_quench_yhi,
    )

    std_urej = _sigmoid(
        lgm,
        X0,
        K,
        params.std_urej_quench_ylo,
        params.std_urej_quench_yhi,
    )
    diags = std_uqt, std_uqs, std_udrop, std_urej

    rho_uqs_uqt = _sigmoid(
        lgm,
        X0,
        K,
        params.rho_uqs_uqt_quench_ylo,
        params.rho_uqs_uqt_quench_yhi,
    )

    rho_udrop_uqs = _sigmoid(
        lgm,
        X0,
        K,
        params.rho_udrop_uqs_quench_ylo,
        params.rho_udrop_uqs_quench_yhi,
    )

    rho_udrop_uqt = _sigmoid(
        lgm,
        X0,
        K,
        params.rho_udrop_uqt_quench_ylo,
        params.rho_udrop_uqt_quench_yhi,
    )

    rho_urej_uqt = _sigmoid(
        lgm,
        X0,
        K,
        params.rho_urej_uqt_quench_ylo,
        params.rho_urej_uqt_quench_yhi,
    )

    rho_urej_uqs = _sigmoid(
        lgm,
        X0,
        K,
        params.rho_urej_uqs_quench_ylo,
        params.rho_urej_uqs_quench_yhi,
    )

    rho_urej_udrop = _sigmoid(
        lgm,
        X0,
        K,
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
def _frac_quench_vs_lgm0(params, lgm0):
    frac_q = _sigmoid(
        lgm0,
        params.frac_quench_x0,
        params.frac_quench_k,
        params.frac_quench_ylo,
        params.frac_quench_yhi,
    )
    frac_q = jnp.clip(frac_q, a_min=0.0, a_max=1.0)
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
