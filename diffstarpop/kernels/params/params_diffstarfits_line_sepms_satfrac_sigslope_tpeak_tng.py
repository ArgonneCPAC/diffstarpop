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
        ("mean_ulgm_mseq_xtp", 12.060),
        ("mean_ulgm_mseq_ytp", 11.953),
        ("mean_ulgm_mseq_lo", 1.095),
        ("mean_ulgm_mseq_hi", -0.070),
        ("mean_ulgy_mseq_int", 0.063),
        ("mean_ulgy_mseq_slp", 0.026),
        ("mean_ul_mseq_int", 0.205),
        ("mean_ul_mseq_slp", -0.021),
        ("mean_utau_mseq_int", -8.698),
        ("mean_utau_mseq_slp", -18.000),
        ("mean_ulgm_qseq_xtp", 11.766),
        ("mean_ulgm_qseq_ytp", 11.561),
        ("mean_ulgm_qseq_lo", 3.561),
        ("mean_ulgm_qseq_hi", 0.233),
        ("mean_ulgy_qseq_int", -0.017),
        ("mean_ulgy_qseq_slp", -0.162),
        ("mean_ul_qseq_int", -0.022),
        ("mean_ul_qseq_slp", 0.327),
        ("mean_utau_qseq_int", -7.602),
        ("mean_utau_qseq_slp", -17.929),
        ("mean_uqt_int", 0.940),
        ("mean_uqt_slp", -0.152),
        ("mean_uqs_int", 0.100),
        ("mean_uqs_slp", 0.058),
        ("mean_udrop_int", -1.882),
        ("mean_udrop_slp", 0.417),
        ("mean_urej_int", -0.674),
        ("mean_urej_slp", -0.059),
    ]
)

SFH_PDF_QUENCH_COV_MS_BLOCK_PDICT = OrderedDict(
    [
        ("std_ulgm_mseq_int", 0.250),
        ("std_ulgm_mseq_slp", 0.001),
        ("std_ulgy_mseq_int", 0.233),
        ("std_ulgy_mseq_slp", 0.020),
        ("std_ul_mseq_int", 0.346),
        ("std_ul_mseq_slp", 0.081),
        ("std_utau_mseq_int", 4.949),
        ("std_utau_mseq_slp", -2.700),
        ("std_ulgm_qseq_int", 0.259),
        ("std_ulgm_qseq_slp", -0.052),
        ("std_ulgy_qseq_int", 0.266),
        ("std_ulgy_qseq_slp", 0.007),
        ("std_ul_qseq_int", 0.335),
        ("std_ul_qseq_slp", 0.167),
        ("std_utau_qseq_int", 7.769),
        ("std_utau_qseq_slp", 2.700),
    ]
)

SFH_PDF_QUENCH_COV_Q_BLOCK_PDICT = OrderedDict(
    [
        ("std_uqt_int", 0.130),
        ("std_uqt_slp", 0.064),
        ("std_uqs_int", 0.500),
        ("std_uqs_slp", 0.087),
        ("std_udrop_int", 0.638),
        ("std_udrop_slp", -0.158),
        ("std_urej_int", 1.177),
        ("std_urej_slp", -0.253),
    ]
)

SFH_PDF_FRAC_QUENCH_PDICT = OrderedDict(
    [
        ("frac_quench_cen_x0_tpeak", 7.000),
        ("frac_quench_cen_k_tpeak", 2.000),
        ("frac_quench_cen_x0_ylotpeak", 12.433),
        ("frac_quench_cen_x0_yhitpeak", 12.050),
        ("frac_quench_cen_ylo_ylotpeak", 0.949),
        ("frac_quench_cen_ylo_yhitpeak", 0.010),
        ("frac_quench_cen_k", 3.848),
        ("frac_quench_cen_yhi", 0.971),
        ("frac_quench_sat_x0_tpeak", 7.000),
        ("frac_quench_sat_k_tpeak", 2.000),
        ("frac_quench_sat_x0_ylotpeak", 12.433),
        ("frac_quench_sat_x0_yhitpeak", 12.050),
        ("frac_quench_sat_ylo_ylotpeak", 0.949),
        ("frac_quench_sat_ylo_yhitpeak", 0.010),
        ("frac_quench_sat_k", 3.848),
        ("frac_quench_sat_yhi", 0.971),
    ]
)
DELTA_UQT_PDICT = OrderedDict(
    [
        ("delta_uqt_x0", 11.563),
        ("delta_uqt_k", 0.026),
        ("delta_uqt_ylo", -0.999),
        ("delta_uqt_yhi", 0.992),
        ("delta_uqt_slope", -0.025),
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


DIFFSTARFITS_TNG_DIFFSTARPOP_PARAMS = DiffstarPopParams(
    SFH_PDF_QUENCH_PARAMS, DEFAULT_SATQUENCHPOP_PARAMS
)

_U_PNAMES = ["u_" + key for key in DIFFSTARFITS_TNG_DIFFSTARPOP_PARAMS._fields]
DiffstarPopUParams = namedtuple("DiffstarPopUParams", _U_PNAMES)

DIFFSTARFITS_TNG_DIFFSTARPOP_U_PARAMS = get_unbounded_diffstarpop_params(
    DIFFSTARFITS_TNG_DIFFSTARPOP_PARAMS
)
