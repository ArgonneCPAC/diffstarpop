from collections import OrderedDict, namedtuple

import typing
from jax import numpy as jnp

from ..satquenchpop_model import (
    DEFAULT_SATQUENCHPOP_PARAMS,
)
from ..defaults_tpeak_line_sepms_satfrac import get_unbounded_diffstarpop_params

SFH_PDF_QUENCH_MU_PDICT = OrderedDict(
    [
        ("mean_ulgm_mseq_int", 11.485),
        ("mean_ulgm_mseq_slp", 0.688),
        ("mean_ulgy_mseq_int", -0.994),
        ("mean_ulgy_mseq_slp", 0.007),
        ("mean_ul_mseq_int", -2.924),
        ("mean_ul_mseq_slp", -3.414),
        ("mean_utau_mseq_int", 6.395),
        ("mean_utau_mseq_slp", 0.107),
        ("mean_ulgm_qseq_int", 11.556),
        ("mean_ulgm_qseq_slp", -1.080),
        ("mean_ulgy_qseq_int", -0.952),
        ("mean_ulgy_qseq_slp", 0.629),
        ("mean_ul_qseq_int", 0.567),
        ("mean_ul_qseq_slp", 0.628),
        ("mean_utau_qseq_int", 1.148),
        ("mean_utau_qseq_slp", 1.201),
        ("mean_uqt_int", 1.070),
        ("mean_uqt_slp", -0.213),
        ("mean_uqs_int", -3.980),
        ("mean_uqs_slp", 4.487),
        ("mean_udrop_int", -1.215),
        ("mean_udrop_slp", 0.207),
        ("mean_urej_int", -8.293),
        ("mean_urej_slp", -0.395),
    ]
)

SFH_PDF_QUENCH_COV_MS_BLOCK_PDICT = OrderedDict(
    [
        ("std_ulgm_mseq_int", 0.015),
        ("std_ulgm_mseq_slp", -0.447),
        ("std_ulgy_mseq_int", 0.014),
        ("std_ulgy_mseq_slp", -0.187),
        ("std_ul_mseq_int", 0.072),
        ("std_ul_mseq_slp", 0.100),
        ("std_utau_mseq_int", 1.037),
        ("std_utau_mseq_slp", 0.066),
        ("std_ulgm_qseq_int", 0.292),
        ("std_ulgm_qseq_slp", -0.997),
        ("std_ulgy_qseq_int", 0.028),
        ("std_ulgy_qseq_slp", 0.067),
        ("std_ul_qseq_int", 0.044),
        ("std_ul_qseq_slp", 0.572),
        ("std_utau_qseq_int", 3.156),
        ("std_utau_qseq_slp", -1.253),
    ]
)

SFH_PDF_QUENCH_COV_Q_BLOCK_PDICT = OrderedDict(
    [
        ("std_uqt_int", 0.103),
        ("std_uqt_slp", -0.189),
        ("std_uqs_int", 0.310),
        ("std_uqs_slp", -0.620),
        ("std_udrop_int", 0.140),
        ("std_udrop_slp", 0.634),
        ("std_urej_int", 0.153),
        ("std_urej_slp", 0.062),
    ]
)

SFH_PDF_FRAC_QUENCH_PDICT = OrderedDict(
    [
        ("frac_quench_cen_x0", 12.513),
        ("frac_quench_cen_k", 4.919),
        ("frac_quench_cen_ylo", 0.001),
        ("frac_quench_cen_yhi", 0.998),
        ("frac_quench_sat_x0", 11.860),
        ("frac_quench_sat_k", 0.021),
        ("frac_quench_sat_ylo", 0.521),
        ("frac_quench_sat_yhi", 0.999),
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
