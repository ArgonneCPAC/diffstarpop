from collections import OrderedDict, namedtuple

import typing
from jax import numpy as jnp

from ..satquenchpop_model import (
    DEFAULT_SATQUENCHPOP_PARAMS,
)
from ..defaults_tpeak_line_sepms_satfrac import get_unbounded_diffstarpop_params

SFH_PDF_QUENCH_MU_PDICT = OrderedDict(
    [
        ("mean_ulgm_mseq_int", 11.514),
        ("mean_ulgm_mseq_slp", -6.434),
        ("mean_ulgy_mseq_int", -0.917),
        ("mean_ulgy_mseq_slp", 0.817),
        ("mean_ul_mseq_int", -0.590),
        ("mean_ul_mseq_slp", -0.253),
        ("mean_utau_mseq_int", -19.840),
        ("mean_utau_mseq_slp", -1.523),
        ("mean_ulgm_qseq_int", 11.089),
        ("mean_ulgm_qseq_slp", 0.105),
        ("mean_ulgy_qseq_int", -0.291),
        ("mean_ulgy_qseq_slp", 0.738),
        ("mean_ul_qseq_int", 2.205),
        ("mean_ul_qseq_slp", 7.218),
        ("mean_utau_qseq_int", 3.799),
        ("mean_utau_qseq_slp", -3.558),
        ("mean_uqt_int", 1.112),
        ("mean_uqt_slp", 0.030),
        ("mean_uqs_int", -1.200),
        ("mean_uqs_slp", 4.908),
        ("mean_udrop_int", -1.603),
        ("mean_udrop_slp", 0.053),
        ("mean_urej_int", 0.754),
        ("mean_urej_slp", 1.091),
    ]
)

SFH_PDF_QUENCH_COV_MS_BLOCK_PDICT = OrderedDict(
    [
        ("std_ulgm_mseq_int", 0.055),
        ("std_ulgm_mseq_slp", -0.045),
        ("std_ulgy_mseq_int", 0.030),
        ("std_ulgy_mseq_slp", -0.010),
        ("std_ul_mseq_int", 0.209),
        ("std_ul_mseq_slp", -0.008),
        ("std_utau_mseq_int", 2.470),
        ("std_utau_mseq_slp", 1.716),
        ("std_ulgm_qseq_int", 0.142),
        ("std_ulgm_qseq_slp", 0.024),
        ("std_ulgy_qseq_int", 0.037),
        ("std_ulgy_qseq_slp", -0.235),
        ("std_ul_qseq_int", 0.042),
        ("std_ul_qseq_slp", 0.051),
        ("std_utau_qseq_int", 1.427),
        ("std_utau_qseq_slp", 1.355),
    ]
)

SFH_PDF_QUENCH_COV_Q_BLOCK_PDICT = OrderedDict(
    [
        ("std_uqt_int", 0.151),
        ("std_uqt_slp", -0.035),
        ("std_uqs_int", 0.156),
        ("std_uqs_slp", -0.035),
        ("std_udrop_int", 0.097),
        ("std_udrop_slp", -0.030),
        ("std_urej_int", 0.299),
        ("std_urej_slp", 0.450),
    ]
)

SFH_PDF_FRAC_QUENCH_PDICT = OrderedDict(
    [
        ("frac_quench_cen_x0", 12.485),
        ("frac_quench_cen_k", 0.031),
        ("frac_quench_cen_ylo", 0.999),
        ("frac_quench_cen_yhi", 0.839),
        ("frac_quench_sat_x0", 11.942),
        ("frac_quench_sat_k", 4.190),
        ("frac_quench_sat_ylo", 0.001),
        ("frac_quench_sat_yhi", 0.988),
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
