from collections import OrderedDict, namedtuple

import typing
from jax import numpy as jnp

from ..satquenchpop_model import (
    DEFAULT_SATQUENCHPOP_PARAMS,
)
from ..defaults_tpeak_line_sepms_satfrac import get_unbounded_diffstarpop_params

SFH_PDF_QUENCH_MU_PDICT = OrderedDict(
    [
        ("mean_ulgm_mseq_int", 11.672),
        ("mean_ulgm_mseq_slp", -0.692),
        ("mean_ulgy_mseq_int", 0.013),
        ("mean_ulgy_mseq_slp", 0.588),
        ("mean_ul_mseq_int", 3.198),
        ("mean_ul_mseq_slp", 6.787),
        ("mean_utau_mseq_int", -5.884),
        ("mean_utau_mseq_slp", -12.386),
        ("mean_ulgm_qseq_int", 11.660),
        ("mean_ulgm_qseq_slp", 0.254),
        ("mean_ulgy_qseq_int", 0.897),
        ("mean_ulgy_qseq_slp", -0.161),
        ("mean_ul_qseq_int", -0.886),
        ("mean_ul_qseq_slp", -1.355),
        ("mean_utau_qseq_int", 3.435),
        ("mean_utau_qseq_slp", -7.466),
        ("mean_uqt_int", 0.869),
        ("mean_uqt_slp", -0.547),
        ("mean_uqs_int", -0.289),
        ("mean_uqs_slp", -0.589),
        ("mean_udrop_int", -2.708),
        ("mean_udrop_slp", 0.941),
        ("mean_urej_int", -3.126),
        ("mean_urej_slp", 1.176),
    ]
)

SFH_PDF_QUENCH_COV_MS_BLOCK_PDICT = OrderedDict(
    [
        ("std_ulgm_mseq_int", 0.035),
        ("std_ulgm_mseq_slp", -0.003),
        ("std_ulgy_mseq_int", 0.116),
        ("std_ulgy_mseq_slp", -0.084),
        ("std_ul_mseq_int", 0.065),
        ("std_ul_mseq_slp", -0.429),
        ("std_utau_mseq_int", 1.168),
        ("std_utau_mseq_slp", -2.183),
        ("std_ulgm_qseq_int", 0.146),
        ("std_ulgm_qseq_slp", -0.037),
        ("std_ulgy_qseq_int", 0.099),
        ("std_ulgy_qseq_slp", 0.065),
        ("std_ul_qseq_int", 0.055),
        ("std_ul_qseq_slp", -0.039),
        ("std_utau_qseq_int", 1.276),
        ("std_utau_qseq_slp", 0.310),
    ]
)

SFH_PDF_QUENCH_COV_Q_BLOCK_PDICT = OrderedDict(
    [
        ("std_uqt_int", 0.085),
        ("std_uqt_slp", -0.002),
        ("std_uqs_int", 0.523),
        ("std_uqs_slp", -0.755),
        ("std_udrop_int", 0.984),
        ("std_udrop_slp", -0.793),
        ("std_urej_int", 0.090),
        ("std_urej_slp", 0.805),
    ]
)

SFH_PDF_FRAC_QUENCH_PDICT = OrderedDict(
    [
        ("frac_quench_cen_x0", 12.078),
        ("frac_quench_cen_k", 3.766),
        ("frac_quench_cen_ylo", 0.001),
        ("frac_quench_cen_yhi", 0.990),
        ("frac_quench_sat_x0", 12.067),
        ("frac_quench_sat_k", 1.766),
        ("frac_quench_sat_ylo", 0.005),
        ("frac_quench_sat_yhi", 0.998),
    ]
)

SFH_PDF_QUENCH_PDICT = SFH_PDF_FRAC_QUENCH_PDICT.copy()
SFH_PDF_QUENCH_PDICT.update(SFH_PDF_QUENCH_MU_PDICT)
SFH_PDF_QUENCH_PDICT.update(SFH_PDF_QUENCH_COV_MS_BLOCK_PDICT)
SFH_PDF_QUENCH_PDICT.update(SFH_PDF_QUENCH_COV_Q_BLOCK_PDICT)

QseqParams = namedtuple("QseqParams", list(SFH_PDF_QUENCH_PDICT.keys()))
SFH_PDF_QUENCH_PARAMS = QseqParams(**SFH_PDF_QUENCH_PDICT)
_UPNAMES = ["u_" + key for key in QseqParams._fields]
QseqUParams = namedtuple("QseqUParams", _UPNAMES)


# Define a namedtuple container for the params of each component
class DiffstarPopParams(typing.NamedTuple):
    sfh_pdf_cens_params: jnp.array
    satquench_params: jnp.array


DIFFSTARPOP_FITS_TNG_DIFFSTARPOP_PARAMS = DiffstarPopParams(
    SFH_PDF_QUENCH_PARAMS, DEFAULT_SATQUENCHPOP_PARAMS
)

_U_PNAMES = ["u_" + key for key in DIFFSTARPOP_FITS_TNG_DIFFSTARPOP_PARAMS._fields]
DiffstarPopUParams = namedtuple("DiffstarPopUParams", _U_PNAMES)

DIFFSTARPOP_FITS_TNG_DIFFSTARPOP_U_PARAMS = get_unbounded_diffstarpop_params(
    DIFFSTARPOP_FITS_TNG_DIFFSTARPOP_PARAMS
)
