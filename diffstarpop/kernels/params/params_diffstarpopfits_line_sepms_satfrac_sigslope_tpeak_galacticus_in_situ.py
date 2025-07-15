from collections import OrderedDict, namedtuple

import typing
from jax import numpy as jnp

from ..satquenchpop_model import (
    DEFAULT_SATQUENCHPOP_PARAMS,
)
from ..defaults_tpeak_line_sepms_satfrac_sigslope_tpeak import (
    get_unbounded_diffstarpop_params,
)

SFH_PDF_QUENCH_MU_PDICT = OrderedDict(
    [
        ("mean_ulgm_mseq_xtp", 12.116),
        ("mean_ulgm_mseq_ytp", 11.437),
        ("mean_ulgm_mseq_lo", 1.820),
        ("mean_ulgm_mseq_hi", -0.048),
        ("mean_ulgy_mseq_int", -0.995),
        ("mean_ulgy_mseq_slp", 0.013),
        ("mean_ul_mseq_int", -2.968),
        ("mean_ul_mseq_slp", -3.511),
        ("mean_utau_mseq_int", 6.120),
        ("mean_utau_mseq_slp", -1.875),
        ("mean_ulgm_qseq_xtp", 12.166),
        ("mean_ulgm_qseq_ytp", 11.415),
        ("mean_ulgm_qseq_lo", 1.073),
        ("mean_ulgm_qseq_hi", -0.375),
        ("mean_ulgy_qseq_int", -0.813),
        ("mean_ulgy_qseq_slp", 0.395),
        ("mean_ul_qseq_int", 4.191),
        ("mean_ul_qseq_slp", 0.413),
        ("mean_utau_qseq_int", 3.897),
        ("mean_utau_qseq_slp", -4.438),
        ("mean_uqt_int", 0.874),
        ("mean_uqt_slp", 0.031),
        ("mean_uqs_int", -4.744),
        ("mean_uqs_slp", 11.155),
        ("mean_udrop_int", -1.317),
        ("mean_udrop_slp", 0.417),
        ("mean_urej_int", -8.153),
        ("mean_urej_slp", 0.529),
    ]
)

SFH_PDF_QUENCH_COV_MS_BLOCK_PDICT = OrderedDict(
    [
        ("std_ulgm_mseq_int", 0.188),
        ("std_ulgm_mseq_slp", 0.064),
        ("std_ulgy_mseq_int", 0.015),
        ("std_ulgy_mseq_slp", -0.013),
        ("std_ul_mseq_int", 0.247),
        ("std_ul_mseq_slp", 0.975),
        ("std_utau_mseq_int", 1.116),
        ("std_utau_mseq_slp", -0.520),
        ("std_ulgm_qseq_int", 0.047),
        ("std_ulgm_qseq_slp", -0.043),
        ("std_ulgy_qseq_int", 0.016),
        ("std_ulgy_qseq_slp", 0.089),
        ("std_ul_qseq_int", 0.018),
        ("std_ul_qseq_slp", -0.624),
        ("std_utau_qseq_int", 1.071),
        ("std_utau_qseq_slp", -2.880),
    ]
)

SFH_PDF_QUENCH_COV_Q_BLOCK_PDICT = OrderedDict(
    [
        ("std_uqt_int", 0.131),
        ("std_uqt_slp", 0.037),
        ("std_uqs_int", 0.831),
        ("std_uqs_slp", -0.697),
        ("std_udrop_int", 0.904),
        ("std_udrop_slp", 0.136),
        ("std_urej_int", 0.128),
        ("std_urej_slp", 0.086),
    ]
)

SFH_PDF_FRAC_QUENCH_PDICT = OrderedDict(
    [
        ("frac_quench_cen_x0_tpeak", 6.499),
        ("frac_quench_cen_k_tpeak", 2.284),
        ("frac_quench_cen_x0_ylotpeak", 11.100),
        ("frac_quench_cen_x0_yhitpeak", 12.389),
        ("frac_quench_cen_ylo_ylotpeak", 0.990),
        ("frac_quench_cen_ylo_yhitpeak", 0.025),
        ("frac_quench_cen_k", 4.995),
        ("frac_quench_cen_yhi", 0.852),
        ("frac_quench_sat_x0_tpeak", 5.886),
        ("frac_quench_sat_k_tpeak", 9.842),
        ("frac_quench_sat_x0_ylotpeak", 11.895),
        ("frac_quench_sat_x0_yhitpeak", 11.120),
        ("frac_quench_sat_ylo_ylotpeak", 0.999),
        ("frac_quench_sat_ylo_yhitpeak", 0.001),
        ("frac_quench_sat_k", 4.997),
        ("frac_quench_sat_yhi", 0.669),
    ]
)

DELTA_UQT_PDICT = OrderedDict(
    [
        ("delta_uqt_x0", 1.011),
        ("delta_uqt_k", 0.209),
        ("delta_uqt_ylo", -0.928),
        ("delta_uqt_yhi", 0.134),
        ("delta_uqt_slope", -0.058),
    ]
)
SFH_PDF_QUENCH_PDICT = SFH_PDF_FRAC_QUENCH_PDICT.copy()
SFH_PDF_QUENCH_PDICT.update(SFH_PDF_QUENCH_MU_PDICT)
SFH_PDF_QUENCH_PDICT.update(SFH_PDF_QUENCH_COV_MS_BLOCK_PDICT)
SFH_PDF_QUENCH_PDICT.update(SFH_PDF_QUENCH_COV_Q_BLOCK_PDICT)

SFH_PDF_QUENCH_PDICT.update(DELTA_UQT_PDICT)

QseqParams = namedtuple("QseqParams", list(SFH_PDF_QUENCH_PDICT.keys()))
SFH_PDF_QUENCH_PARAMS = QseqParams(**SFH_PDF_QUENCH_PDICT)
_UPNAMES = ["u_" + key for key in QseqParams._fields]
QseqUParams = namedtuple("QseqUParams", _UPNAMES)


# Define a namedtuple container for the params of each component
class DiffstarPopParams(typing.NamedTuple):
    sfh_pdf_cens_params: jnp.array
    satquench_params: jnp.array


DIFFSTARPOP_FITS_GALACTICUS_IN_DIFFSTARPOP_PARAMS = DiffstarPopParams(
    SFH_PDF_QUENCH_PARAMS, DEFAULT_SATQUENCHPOP_PARAMS
)

_U_PNAMES = [
    "u_" + key for key in DIFFSTARPOP_FITS_GALACTICUS_IN_DIFFSTARPOP_PARAMS._fields
]
DiffstarPopUParams = namedtuple("DiffstarPopUParams", _U_PNAMES)

DIFFSTARPOP_FITS_GALACTICUS_IN_DIFFSTARPOP_U_PARAMS = get_unbounded_diffstarpop_params(
    DIFFSTARPOP_FITS_GALACTICUS_IN_DIFFSTARPOP_PARAMS
)
