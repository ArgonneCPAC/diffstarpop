from collections import OrderedDict, namedtuple

import typing
from jax import numpy as jnp

from ..satquenchpop_model import (
    DEFAULT_SATQUENCHPOP_PARAMS,
)
from ..defaults_tpeak_line_sepms_satfrac import get_unbounded_diffstarpop_params

SFH_PDF_QUENCH_MU_PDICT = OrderedDict(
    [
        ("mean_ulgm_mseq_int", 11.700),
        ("mean_ulgm_mseq_slp", 0.726),
        ("mean_ulgy_mseq_int", -0.447),
        ("mean_ulgy_mseq_slp", -0.168),
        ("mean_ul_mseq_int", 0.552),
        ("mean_ul_mseq_slp", 0.352),
        ("mean_utau_mseq_int", 4.393),
        ("mean_utau_mseq_slp", -4.443),
        ("mean_ulgm_qseq_int", 11.700),
        ("mean_ulgm_qseq_slp", 2.380),
        ("mean_ulgy_qseq_int", -0.307),
        ("mean_ulgy_qseq_slp", 0.424),
        ("mean_ul_qseq_int", 0.534),
        ("mean_ul_qseq_slp", 0.483),
        ("mean_utau_qseq_int", 6.554),
        ("mean_utau_qseq_slp", 5.760),
        ("mean_uqt_int", 0.992),
        ("mean_uqt_slp", 0.010),
        ("mean_uqs_int", 0.546),
        ("mean_uqs_slp", -4.458),
        ("mean_udrop_int", -1.181),
        ("mean_udrop_slp", -0.090),
        ("mean_urej_int", -0.027),
        ("mean_urej_slp", -1.202),
    ]
)

SFH_PDF_QUENCH_COV_MS_BLOCK_PDICT = OrderedDict(
    [
        ("std_ulgm_mseq_int", 0.387),
        ("std_ulgm_mseq_slp", -0.195),
        ("std_ulgy_mseq_int", 0.391),
        ("std_ulgy_mseq_slp", 0.053),
        ("std_ul_mseq_int", 0.458),
        ("std_ul_mseq_slp", 0.035),
        ("std_utau_mseq_int", 2.951),
        ("std_utau_mseq_slp", 1.931),
        ("std_ulgm_qseq_int", 0.551),
        ("std_ulgm_qseq_slp", -0.900),
        ("std_ulgy_qseq_int", 0.389),
        ("std_ulgy_qseq_slp", -0.078),
        ("std_ul_qseq_int", 0.389),
        ("std_ul_qseq_slp", -0.097),
        ("std_utau_qseq_int", 2.987),
        ("std_utau_qseq_slp", 0.004),
    ]
)

SFH_PDF_QUENCH_COV_Q_BLOCK_PDICT = OrderedDict(
    [
        ("std_uqt_int", 0.129),
        ("std_uqt_slp", 0.001),
        ("std_uqs_int", 0.900),
        ("std_uqs_slp", 0.899),
        ("std_udrop_int", 0.609),
        ("std_udrop_slp", -0.176),
        ("std_urej_int", 0.876),
        ("std_urej_slp", -0.299),
    ]
)

SFH_PDF_FRAC_QUENCH_PDICT = OrderedDict(
    [
        ("frac_quench_cen_x0", 10.200),
        ("frac_quench_cen_k", 0.200),
        ("frac_quench_cen_ylo", 0.998),
        ("frac_quench_cen_yhi", 0.002),
        ("frac_quench_sat_x0", 12.592),
        ("frac_quench_sat_k", 0.843),
        ("frac_quench_sat_ylo", 0.002),
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


DIFFSTARFITS_GALACTICUS_INPLUSEX_DIFFSTARPOP_PARAMS = DiffstarPopParams(
    SFH_PDF_QUENCH_PARAMS, DEFAULT_SATQUENCHPOP_PARAMS
)

_U_PNAMES = [
    "u_" + key for key in DIFFSTARFITS_GALACTICUS_INPLUSEX_DIFFSTARPOP_PARAMS._fields
]
DiffstarPopUParams = namedtuple("DiffstarPopUParams", _U_PNAMES)

DIFFSTARFITS_GALACTICUS_INPLUSEX_DIFFSTARPOP_U_PARAMS = (
    get_unbounded_diffstarpop_params(
        DIFFSTARFITS_GALACTICUS_INPLUSEX_DIFFSTARPOP_PARAMS
    )
)
