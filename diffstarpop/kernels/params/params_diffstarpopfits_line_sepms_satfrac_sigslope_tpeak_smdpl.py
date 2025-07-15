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
        ("mean_ulgm_mseq_xtp", 12.793),
        ("mean_ulgm_mseq_ytp", 11.828),
        ("mean_ulgm_mseq_lo", 0.173),
        ("mean_ulgm_mseq_hi", -1.953),
        ("mean_ulgy_mseq_int", 0.155),
        ("mean_ulgy_mseq_slp", 0.907),
        ("mean_ul_mseq_int", -1.261),
        ("mean_ul_mseq_slp", -0.834),
        ("mean_utau_mseq_int", 5.297),
        ("mean_utau_mseq_slp", -5.961),
        ("mean_ulgm_qseq_xtp", 12.185),
        ("mean_ulgm_qseq_ytp", 12.011),
        ("mean_ulgm_qseq_lo", 1.661),
        ("mean_ulgm_qseq_hi", 0.069),
        ("mean_ulgy_qseq_int", 0.273),
        ("mean_ulgy_qseq_slp", -0.432),
        ("mean_ul_qseq_int", 0.202),
        ("mean_ul_qseq_slp", -0.335),
        ("mean_utau_qseq_int", 3.542),
        ("mean_utau_qseq_slp", -8.932),
        ("mean_uqt_int", 0.866),
        ("mean_uqt_slp", -0.273),
        ("mean_uqs_int", -0.183),
        ("mean_uqs_slp", 0.270),
        ("mean_udrop_int", -2.917),
        ("mean_udrop_slp", -0.172),
        ("mean_urej_int", -1.880),
        ("mean_urej_slp", 0.800),
    ]
)

SFH_PDF_QUENCH_COV_MS_BLOCK_PDICT = OrderedDict(
    [
        ("std_ulgm_mseq_int", 0.099),
        ("std_ulgm_mseq_slp", 0.287),
        ("std_ulgy_mseq_int", 0.317),
        ("std_ulgy_mseq_slp", 0.053),
        ("std_ul_mseq_int", 0.136),
        ("std_ul_mseq_slp", 0.147),
        ("std_utau_mseq_int", 2.344),
        ("std_utau_mseq_slp", 2.359),
        ("std_ulgm_qseq_int", 0.356),
        ("std_ulgm_qseq_slp", -0.010),
        ("std_ulgy_qseq_int", 0.207),
        ("std_ulgy_qseq_slp", -0.037),
        ("std_ul_qseq_int", 0.086),
        ("std_ul_qseq_slp", 0.681),
        ("std_utau_qseq_int", 1.877),
        ("std_utau_qseq_slp", 0.858),
    ]
)

SFH_PDF_QUENCH_COV_Q_BLOCK_PDICT = OrderedDict(
    [
        ("std_uqt_int", 0.121),
        ("std_uqt_slp", -0.056),
        ("std_uqs_int", 0.383),
        ("std_uqs_slp", -0.091),
        ("std_udrop_int", 1.395),
        ("std_udrop_slp", -0.905),
        ("std_urej_int", 1.159),
        ("std_urej_slp", -0.148),
    ]
)

SFH_PDF_FRAC_QUENCH_PDICT = OrderedDict(
    [
        ("frac_quench_cen_x0_tpeak", 10.914),
        ("frac_quench_cen_k_tpeak", 0.295),
        ("frac_quench_cen_x0_ylotpeak", 13.929),
        ("frac_quench_cen_x0_yhitpeak", 11.645),
        ("frac_quench_cen_ylo_ylotpeak", 0.650),
        ("frac_quench_cen_ylo_yhitpeak", 0.022),
        ("frac_quench_cen_k", 4.423),
        ("frac_quench_cen_yhi", 0.998),
        ("frac_quench_sat_x0_tpeak", 7.862),
        ("frac_quench_sat_k_tpeak", 8.136),
        ("frac_quench_sat_x0_ylotpeak", 12.913),
        ("frac_quench_sat_x0_yhitpeak", 11.951),
        ("frac_quench_sat_ylo_ylotpeak", 0.849),
        ("frac_quench_sat_ylo_yhitpeak", 0.001),
        ("frac_quench_sat_k", 4.952),
        ("frac_quench_sat_yhi", 0.996),
    ]
)

DELTA_UQT_PDICT = OrderedDict(
    [
        ("delta_uqt_x0", 5.056),
        ("delta_uqt_k", 0.302),
        ("delta_uqt_ylo", -0.399),
        ("delta_uqt_yhi", 0.091),
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


DIFFSTARPOP_FITS_SMDPL_DIFFSTARPOP_PARAMS = DiffstarPopParams(
    SFH_PDF_QUENCH_PARAMS, DEFAULT_SATQUENCHPOP_PARAMS
)

_U_PNAMES = ["u_" + key for key in DIFFSTARPOP_FITS_SMDPL_DIFFSTARPOP_PARAMS._fields]
DiffstarPopUParams = namedtuple("DiffstarPopUParams", _U_PNAMES)

DIFFSTARPOP_FITS_SMDPL_DIFFSTARPOP_U_PARAMS = get_unbounded_diffstarpop_params(
    DIFFSTARPOP_FITS_SMDPL_DIFFSTARPOP_PARAMS
)
