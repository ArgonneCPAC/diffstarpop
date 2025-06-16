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
        ("mean_ulgm_mseq_xtp", 12.174),
        ("mean_ulgm_mseq_ytp", 11.486),
        ("mean_ulgm_mseq_lo", 1.569),
        ("mean_ulgm_mseq_hi", -0.082),
        ("mean_ulgy_mseq_int", -0.952),
        ("mean_ulgy_mseq_slp", 0.036),
        ("mean_ul_mseq_int", -2.946),
        ("mean_ul_mseq_slp", -2.837),
        ("mean_utau_mseq_int", 6.191),
        ("mean_utau_mseq_slp", -1.874),
        ("mean_ulgm_qseq_xtp", 12.163),
        ("mean_ulgm_qseq_ytp", 11.498),
        ("mean_ulgm_qseq_lo", 0.961),
        ("mean_ulgm_qseq_hi", -0.394),
        ("mean_ulgy_qseq_int", -0.854),
        ("mean_ulgy_qseq_slp", 0.277),
        ("mean_ul_qseq_int", 0.005),
        ("mean_ul_qseq_slp", 2.992),
        ("mean_utau_qseq_int", 4.160),
        ("mean_utau_qseq_slp", -5.312),
        ("mean_uqt_int", 1.000),
        ("mean_uqt_slp", 0.090),
        ("mean_uqs_int", 0.911),
        ("mean_uqs_slp", 0.146),
        ("mean_udrop_int", -2.086),
        ("mean_udrop_slp", 0.784),
        ("mean_urej_int", -2.704),
        ("mean_urej_slp", 1.028),
    ]
)

SFH_PDF_QUENCH_COV_MS_BLOCK_PDICT = OrderedDict(
    [
        ("std_ulgm_mseq_int", 0.140),
        ("std_ulgm_mseq_slp", 0.185),
        ("std_ulgy_mseq_int", 0.037),
        ("std_ulgy_mseq_slp", -0.047),
        ("std_ul_mseq_int", 0.211),
        ("std_ul_mseq_slp", 0.850),
        ("std_utau_mseq_int", 1.023),
        ("std_utau_mseq_slp", -1.288),
        ("std_ulgm_qseq_int", 0.070),
        ("std_ulgm_qseq_slp", -0.035),
        ("std_ulgy_qseq_int", 0.017),
        ("std_ulgy_qseq_slp", 0.009),
        ("std_ul_qseq_int", 0.029),
        ("std_ul_qseq_slp", 0.030),
        ("std_utau_qseq_int", 1.109),
        ("std_utau_qseq_slp", 1.505),
    ]
)

SFH_PDF_QUENCH_COV_Q_BLOCK_PDICT = OrderedDict(
    [
        ("std_uqt_int", 0.216),
        ("std_uqt_slp", 0.060),
        ("std_uqs_int", 0.989),
        ("std_uqs_slp", -0.820),
        ("std_udrop_int", 0.670),
        ("std_udrop_slp", -0.082),
        ("std_urej_int", 0.348),
        ("std_urej_slp", 0.018),
    ]
)

SFH_PDF_FRAC_QUENCH_PDICT = OrderedDict(
    [
        ("frac_quench_cen_x0", 12.565),
        ("frac_quench_cen_k", 4.951),
        ("frac_quench_cen_ylo", 0.001),
        ("frac_quench_cen_yhi", 0.946),
        ("frac_quench_sat_x0", 11.703),
        ("frac_quench_sat_k", 0.016),
        ("frac_quench_sat_ylo", 0.385),
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
