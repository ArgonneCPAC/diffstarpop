from collections import OrderedDict, namedtuple

import typing
from jax import numpy as jnp

from ..satquenchpop_model import (
    DEFAULT_SATQUENCHPOP_PARAMS,
)
from ..defaults_tpeak_line_sepms_satfrac_sigslope import (
    get_unbounded_diffstarpop_params,
)

SFH_PDF_QUENCH_MU_PDICT = OrderedDict(
    [
        ("mean_ulgm_mseq_xtp", 12.143),
        ("mean_ulgm_mseq_ytp", 11.193),
        ("mean_ulgm_mseq_lo", 1.557),
        ("mean_ulgm_mseq_hi", -0.110),
        ("mean_ulgy_mseq_int", -0.444),
        ("mean_ulgy_mseq_slp", 0.391),
        ("mean_ul_mseq_int", -2.111),
        ("mean_ul_mseq_slp", 7.934),
        ("mean_utau_mseq_int", 5.517),
        ("mean_utau_mseq_slp", -1.559),
        ("mean_ulgm_qseq_xtp", 11.994),
        ("mean_ulgm_qseq_ytp", 11.320),
        ("mean_ulgm_qseq_lo", 1.430),
        ("mean_ulgm_qseq_hi", 0.041),
        ("mean_ulgy_qseq_int", -0.756),
        ("mean_ulgy_qseq_slp", 0.520),
        ("mean_ul_qseq_int", 0.878),
        ("mean_ul_qseq_slp", -2.730),
        ("mean_utau_qseq_int", 1.963),
        ("mean_utau_qseq_slp", -3.657),
        ("mean_uqt_int", 1.072),
        ("mean_uqt_slp", 0.051),
        ("mean_uqs_int", 0.729),
        ("mean_uqs_slp", 0.776),
        ("mean_udrop_int", -1.679),
        ("mean_udrop_slp", 0.269),
        ("mean_urej_int", -5.486),
        ("mean_urej_slp", 0.257),
    ]
)

SFH_PDF_QUENCH_COV_MS_BLOCK_PDICT = OrderedDict(
    [
        ("std_ulgm_mseq_int", 0.076),
        ("std_ulgm_mseq_slp", 0.299),
        ("std_ulgy_mseq_int", 0.101),
        ("std_ulgy_mseq_slp", 0.102),
        ("std_ul_mseq_int", 0.542),
        ("std_ul_mseq_slp", -0.742),
        ("std_utau_mseq_int", 1.142),
        ("std_utau_mseq_slp", 0.818),
        ("std_ulgm_qseq_int", 0.052),
        ("std_ulgm_qseq_slp", -0.027),
        ("std_ulgy_qseq_int", 0.043),
        ("std_ulgy_qseq_slp", 0.261),
        ("std_ul_qseq_int", 0.318),
        ("std_ul_qseq_slp", -0.197),
        ("std_utau_qseq_int", 1.487),
        ("std_utau_qseq_slp", -0.791),
    ]
)

SFH_PDF_QUENCH_COV_Q_BLOCK_PDICT = OrderedDict(
    [
        ("std_uqt_int", 0.197),
        ("std_uqt_slp", -0.060),
        ("std_uqs_int", 0.982),
        ("std_uqs_slp", -0.816),
        ("std_udrop_int", 0.371),
        ("std_udrop_slp", -0.155),
        ("std_urej_int", 0.256),
        ("std_urej_slp", -0.036),
    ]
)

SFH_PDF_FRAC_QUENCH_PDICT = OrderedDict(
    [
        ("frac_quench_cen_x0", 12.639),
        ("frac_quench_cen_k", 4.793),
        ("frac_quench_cen_ylo", 0.001),
        ("frac_quench_cen_yhi", 0.993),
        ("frac_quench_sat_x0", 12.341),
        ("frac_quench_sat_k", 0.030),
        ("frac_quench_sat_ylo", 0.410),
        ("frac_quench_sat_yhi", 0.899),
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
