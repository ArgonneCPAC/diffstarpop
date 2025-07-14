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
        ("mean_ulgm_mseq_xtp", 12.183),
        ("mean_ulgm_mseq_ytp", 11.212),
        ("mean_ulgm_mseq_lo", 1.579),
        ("mean_ulgm_mseq_hi", -0.038),
        ("mean_ulgy_mseq_int", -0.548),
        ("mean_ulgy_mseq_slp", 0.331),
        ("mean_ul_mseq_int", -2.961),
        ("mean_ul_mseq_slp", 10.011),
        ("mean_utau_mseq_int", 5.661),
        ("mean_utau_mseq_slp", -1.496),
        ("mean_ulgm_qseq_xtp", 12.036),
        ("mean_ulgm_qseq_ytp", 11.344),
        ("mean_ulgm_qseq_lo", 1.266),
        ("mean_ulgm_qseq_hi", 0.031),
        ("mean_ulgy_qseq_int", -0.776),
        ("mean_ulgy_qseq_slp", 0.468),
        ("mean_ul_qseq_int", 1.947),
        ("mean_ul_qseq_slp", -2.716),
        ("mean_utau_qseq_int", 1.587),
        ("mean_utau_qseq_slp", -3.990),
        ("mean_uqt_int", 0.980),
        ("mean_uqt_slp", 0.157),
        ("mean_uqs_int", -2.064),
        ("mean_uqs_slp", 4.074),
        ("mean_udrop_int", -1.205),
        ("mean_udrop_slp", -0.111),
        ("mean_urej_int", -7.366),
        ("mean_urej_slp", 0.193),
    ]
)

SFH_PDF_QUENCH_COV_MS_BLOCK_PDICT = OrderedDict(
    [
        ("std_ulgm_mseq_int", 0.078),
        ("std_ulgm_mseq_slp", 0.149),
        ("std_ulgy_mseq_int", 0.059),
        ("std_ulgy_mseq_slp", 0.056),
        ("std_ul_mseq_int", 0.284),
        ("std_ul_mseq_slp", -0.960),
        ("std_utau_mseq_int", 1.020),
        ("std_utau_mseq_slp", 0.061),
        ("std_ulgm_qseq_int", 0.157),
        ("std_ulgm_qseq_slp", 0.039),
        ("std_ulgy_qseq_int", 0.020),
        ("std_ulgy_qseq_slp", 0.016),
        ("std_ul_qseq_int", 0.051),
        ("std_ul_qseq_slp", 0.088),
        ("std_utau_qseq_int", 2.138),
        ("std_utau_qseq_slp", -0.979),
    ]
)

SFH_PDF_QUENCH_COV_Q_BLOCK_PDICT = OrderedDict(
    [
        ("std_uqt_int", 0.142),
        ("std_uqt_slp", -0.016),
        ("std_uqs_int", 0.492),
        ("std_uqs_slp", -0.968),
        ("std_udrop_int", 0.383),
        ("std_udrop_slp", -0.208),
        ("std_urej_int", 0.239),
        ("std_urej_slp", -0.039),
    ]
)

SFH_PDF_FRAC_QUENCH_PDICT = OrderedDict(
    [
        ("frac_quench_cen_x0_tpeak", 6.491),
        ("frac_quench_cen_k_tpeak", 2.281),
        ("frac_quench_cen_x0_ylotpeak", 11.100),
        ("frac_quench_cen_x0_yhitpeak", 12.610),
        ("frac_quench_cen_ylo_ylotpeak", 0.990),
        ("frac_quench_cen_ylo_yhitpeak", 0.022),
        ("frac_quench_cen_k", 4.992),
        ("frac_quench_cen_yhi", 0.999),
        ("frac_quench_sat_x0_tpeak", 5.989),
        ("frac_quench_sat_k_tpeak", 5.303),
        ("frac_quench_sat_x0_ylotpeak", 12.344),
        ("frac_quench_sat_x0_yhitpeak", 11.450),
        ("frac_quench_sat_ylo_ylotpeak", 0.999),
        ("frac_quench_sat_ylo_yhitpeak", 0.001),
        ("frac_quench_sat_k", 4.876),
        ("frac_quench_sat_yhi", 0.647),
    ]
)

DELTA_UQT_PDICT = OrderedDict(
    [
        ("delta_uqt_x0", 1.002),
        ("delta_uqt_k", 0.530),
        ("delta_uqt_ylo", -0.715),
        ("delta_uqt_yhi", 0.034),
        ("delta_uqt_slope", -0.057),
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


DIFFSTARPOP_FITS_GALACTICUS_INPLUSEX_DIFFSTARPOP_PARAMS = DiffstarPopParams(
    SFH_PDF_QUENCH_PARAMS, DEFAULT_SATQUENCHPOP_PARAMS
)

_U_PNAMES = [
    "u_" + key
    for key in DIFFSTARPOP_FITS_GALACTICUS_INPLUSEX_DIFFSTARPOP_PARAMS._fields
]
DiffstarPopUParams = namedtuple("DiffstarPopUParams", _U_PNAMES)

DIFFSTARPOP_FITS_GALACTICUS_INPLUSEX_DIFFSTARPOP_U_PARAMS = (
    get_unbounded_diffstarpop_params(
        DIFFSTARPOP_FITS_GALACTICUS_INPLUSEX_DIFFSTARPOP_PARAMS
    )
)
