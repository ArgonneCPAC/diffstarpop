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
        ("mean_ulgm_mseq_xtp", 11.888),
        ("mean_ulgm_mseq_ytp", 11.650),
        ("mean_ulgm_mseq_lo", 1.427),
        ("mean_ulgm_mseq_hi", 0.281),
        ("mean_ulgy_mseq_int", -0.353),
        ("mean_ulgy_mseq_slp", 0.210),
        ("mean_ul_mseq_int", -0.379),
        ("mean_ul_mseq_slp", -0.257),
        ("mean_utau_mseq_int", 0.660),
        ("mean_utau_mseq_slp", -10.507),
        ("mean_ulgm_qseq_xtp", 11.826),
        ("mean_ulgm_qseq_ytp", 11.704),
        ("mean_ulgm_qseq_lo", 2.431),
        ("mean_ulgm_qseq_hi", 0.406),
        ("mean_ulgy_qseq_int", 0.168),
        ("mean_ulgy_qseq_slp", 0.103),
        ("mean_ul_qseq_int", -0.893),
        ("mean_ul_qseq_slp", 2.134),
        ("mean_utau_qseq_int", 3.042),
        ("mean_utau_qseq_slp", -11.344),
        ("mean_uqt_int", 1.028),
        ("mean_uqt_slp", -0.085),
        ("mean_uqs_int", -0.056),
        ("mean_uqs_slp", 1.437),
        ("mean_udrop_int", -1.968),
        ("mean_udrop_slp", 0.930),
        ("mean_urej_int", -0.827),
        ("mean_urej_slp", -1.218),
    ]
)

SFH_PDF_QUENCH_COV_MS_BLOCK_PDICT = OrderedDict(
    [
        ("std_ulgm_mseq_int", 0.144),
        ("std_ulgm_mseq_slp", -0.263),
        ("std_ulgy_mseq_int", 0.129),
        ("std_ulgy_mseq_slp", 0.274),
        ("std_ul_mseq_int", 0.110),
        ("std_ul_mseq_slp", 0.107),
        ("std_utau_mseq_int", 2.866),
        ("std_utau_mseq_slp", 2.179),
        ("std_ulgm_qseq_int", 0.099),
        ("std_ulgm_qseq_slp", 0.003),
        ("std_ulgy_qseq_int", 0.257),
        ("std_ulgy_qseq_slp", 0.030),
        ("std_ul_qseq_int", 0.065),
        ("std_ul_qseq_slp", -0.079),
        ("std_utau_qseq_int", 2.222),
        ("std_utau_qseq_slp", 2.742),
    ]
)

SFH_PDF_QUENCH_COV_Q_BLOCK_PDICT = OrderedDict(
    [
        ("std_uqt_int", 0.135),
        ("std_uqt_slp", 0.077),
        ("std_uqs_int", 0.666),
        ("std_uqs_slp", -0.303),
        ("std_udrop_int", 0.467),
        ("std_udrop_slp", -0.464),
        ("std_urej_int", 0.610),
        ("std_urej_slp", -0.303),
    ]
)

SFH_PDF_FRAC_QUENCH_PDICT = OrderedDict(
    [
        ("frac_quench_cen_x0", 11.089),
        ("frac_quench_cen_k", 2.380),
        ("frac_quench_cen_ylo", 0.006),
        ("frac_quench_cen_yhi", 0.731),
        ("frac_quench_sat_x0", 11.699),
        ("frac_quench_sat_k", 1.414),
        ("frac_quench_sat_ylo", 0.198),
        ("frac_quench_sat_yhi", 0.872),
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


DIFFSTARPOP_FITS_SMDPL_DR1_DIFFSTARPOP_PARAMS = DiffstarPopParams(
    SFH_PDF_QUENCH_PARAMS, DEFAULT_SATQUENCHPOP_PARAMS
)

_U_PNAMES = [
    "u_" + key for key in DIFFSTARPOP_FITS_SMDPL_DR1_DIFFSTARPOP_PARAMS._fields
]
DiffstarPopUParams = namedtuple("DiffstarPopUParams", _U_PNAMES)

DIFFSTARPOP_FITS_SMDPL_DR1_DIFFSTARPOP_U_PARAMS = get_unbounded_diffstarpop_params(
    DIFFSTARPOP_FITS_SMDPL_DR1_DIFFSTARPOP_PARAMS
)
