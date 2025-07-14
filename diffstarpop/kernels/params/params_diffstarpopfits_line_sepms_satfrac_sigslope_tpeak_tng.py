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
        ("mean_ulgm_mseq_xtp", 11.841),
        ("mean_ulgm_mseq_ytp", 12.125),
        ("mean_ulgm_mseq_lo", 1.139),
        ("mean_ulgm_mseq_hi", -0.408),
        ("mean_ulgy_mseq_int", -0.296),
        ("mean_ulgy_mseq_slp", 0.356),
        ("mean_ul_mseq_int", -0.811),
        ("mean_ul_mseq_slp", 0.021),
        ("mean_utau_mseq_int", -13.163),
        ("mean_utau_mseq_slp", -17.980),
        ("mean_ulgm_qseq_xtp", 11.585),
        ("mean_ulgm_qseq_ytp", 11.302),
        ("mean_ulgm_qseq_lo", 2.096),
        ("mean_ulgm_qseq_hi", 0.380),
        ("mean_ulgy_qseq_int", 0.258),
        ("mean_ulgy_qseq_slp", -0.074),
        ("mean_ul_qseq_int", -0.857),
        ("mean_ul_qseq_slp", -1.129),
        ("mean_utau_qseq_int", 1.433),
        ("mean_utau_qseq_slp", -8.669),
        ("mean_uqt_int", 0.878),
        ("mean_uqt_slp", -0.506),
        ("mean_uqs_int", -0.609),
        ("mean_uqs_slp", -0.095),
        ("mean_udrop_int", -2.791),
        ("mean_udrop_slp", 1.521),
        ("mean_urej_int", -2.043),
        ("mean_urej_slp", -0.581),
    ]
)

SFH_PDF_QUENCH_COV_MS_BLOCK_PDICT = OrderedDict(
    [
        ("std_ulgm_mseq_int", 0.185),
        ("std_ulgm_mseq_slp", 0.177),
        ("std_ulgy_mseq_int", 0.194),
        ("std_ulgy_mseq_slp", -0.034),
        ("std_ul_mseq_int", 0.181),
        ("std_ul_mseq_slp", 0.454),
        ("std_utau_mseq_int", 2.270),
        ("std_utau_mseq_slp", -1.304),
        ("std_ulgm_qseq_int", 0.134),
        ("std_ulgm_qseq_slp", 0.007),
        ("std_ulgy_qseq_int", 0.059),
        ("std_ulgy_qseq_slp", -0.081),
        ("std_ul_qseq_int", 0.088),
        ("std_ul_qseq_slp", 0.076),
        ("std_utau_qseq_int", 1.928),
        ("std_utau_qseq_slp", 0.404),
    ]
)

SFH_PDF_QUENCH_COV_Q_BLOCK_PDICT = OrderedDict(
    [
        ("std_uqt_int", 0.091),
        ("std_uqt_slp", -0.033),
        ("std_uqs_int", 0.605),
        ("std_uqs_slp", -0.269),
        ("std_udrop_int", 0.147),
        ("std_udrop_slp", 0.367),
        ("std_urej_int", 0.916),
        ("std_urej_slp", -0.795),
    ]
)

SFH_PDF_FRAC_QUENCH_PDICT = OrderedDict(
    [
        ("frac_quench_cen_x0_tpeak", 9.949),
        ("frac_quench_cen_k_tpeak", 0.072),
        ("frac_quench_cen_x0_ylotpeak", 12.456),
        ("frac_quench_cen_x0_yhitpeak", 11.907),
        ("frac_quench_cen_ylo_ylotpeak", 0.948),
        ("frac_quench_cen_ylo_yhitpeak", 0.018),
        ("frac_quench_cen_k", 4.917),
        ("frac_quench_cen_yhi", 0.998),
        ("frac_quench_sat_x0_tpeak", 4.558),
        ("frac_quench_sat_k_tpeak", 0.744),
        ("frac_quench_sat_x0_ylotpeak", 13.754),
        ("frac_quench_sat_x0_yhitpeak", 11.155),
        ("frac_quench_sat_ylo_ylotpeak", 0.078),
        ("frac_quench_sat_ylo_yhitpeak", 0.003),
        ("frac_quench_sat_k", 4.776),
        ("frac_quench_sat_yhi", 0.951),
    ]
)

DELTA_UQT_PDICT = OrderedDict(
    [
        ("delta_uqt_x0", 11.877),
        ("delta_uqt_k", 0.060),
        ("delta_uqt_ylo", -0.999),
        ("delta_uqt_yhi", 0.992),
        ("delta_uqt_slope", -0.039),
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


DIFFSTARPOP_FITS_TNG_DIFFSTARPOP_PARAMS = DiffstarPopParams(
    SFH_PDF_QUENCH_PARAMS, DEFAULT_SATQUENCHPOP_PARAMS
)

_U_PNAMES = ["u_" + key for key in DIFFSTARPOP_FITS_TNG_DIFFSTARPOP_PARAMS._fields]
DiffstarPopUParams = namedtuple("DiffstarPopUParams", _U_PNAMES)

DIFFSTARPOP_FITS_TNG_DIFFSTARPOP_U_PARAMS = get_unbounded_diffstarpop_params(
    DIFFSTARPOP_FITS_TNG_DIFFSTARPOP_PARAMS
)
