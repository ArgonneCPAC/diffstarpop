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
        ("mean_ulgm_mseq_xtp", 11.728),
        ("mean_ulgm_mseq_ytp", 12.489),
        ("mean_ulgm_mseq_lo", -0.727),
        ("mean_ulgm_mseq_hi", 0.081),
        ("mean_ulgy_mseq_int", -0.344),
        ("mean_ulgy_mseq_slp", -5.352),
        ("mean_ul_mseq_int", -0.075),
        ("mean_ul_mseq_slp", -0.414),
        ("mean_utau_mseq_int", -16.625),
        ("mean_utau_mseq_slp", -17.261),
        ("mean_ulgm_qseq_xtp", 12.468),
        ("mean_ulgm_qseq_ytp", 12.092),
        ("mean_ulgm_qseq_lo", 0.951),
        ("mean_ulgm_qseq_hi", -0.363),
        ("mean_ulgy_qseq_int", 0.446),
        ("mean_ulgy_qseq_slp", 0.869),
        ("mean_ul_qseq_int", -0.029),
        ("mean_ul_qseq_slp", 0.632),
        ("mean_utau_qseq_int", 6.016),
        ("mean_utau_qseq_slp", -7.328),
        ("mean_uqt_int", 0.892),
        ("mean_uqt_slp", -0.257),
        ("mean_uqs_int", -0.249),
        ("mean_uqs_slp", 1.634),
        ("mean_udrop_int", -2.091),
        ("mean_udrop_slp", -1.146),
        ("mean_urej_int", -1.133),
        ("mean_urej_slp", 0.406),
    ]
)

SFH_PDF_QUENCH_COV_MS_BLOCK_PDICT = OrderedDict(
    [
        ("std_ulgm_mseq_int", 0.318),
        ("std_ulgm_mseq_slp", -0.318),
        ("std_ulgy_mseq_int", 0.368),
        ("std_ulgy_mseq_slp", -0.801),
        ("std_ul_mseq_int", 0.323),
        ("std_ul_mseq_slp", 0.367),
        ("std_utau_mseq_int", 5.917),
        ("std_utau_mseq_slp", 2.162),
        ("std_ulgm_qseq_int", 0.203),
        ("std_ulgm_qseq_slp", -0.013),
        ("std_ulgy_qseq_int", 0.398),
        ("std_ulgy_qseq_slp", 0.073),
        ("std_ul_qseq_int", 0.120),
        ("std_ul_qseq_slp", 0.791),
        ("std_utau_qseq_int", 1.672),
        ("std_utau_qseq_slp", 1.807),
    ]
)

SFH_PDF_QUENCH_COV_Q_BLOCK_PDICT = OrderedDict(
    [
        ("std_uqt_int", 0.106),
        ("std_uqt_slp", -0.023),
        ("std_uqs_int", 0.329),
        ("std_uqs_slp", 0.479),
        ("std_udrop_int", 1.030),
        ("std_udrop_slp", -0.540),
        ("std_urej_int", 1.550),
        ("std_urej_slp", 0.458),
    ]
)

SFH_PDF_FRAC_QUENCH_PDICT = OrderedDict(
    [
        ("frac_quench_cen_x0", 10.615),
        ("frac_quench_cen_k", 4.952),
        ("frac_quench_cen_ylo", 0.005),
        ("frac_quench_cen_yhi", 0.927),
        ("frac_quench_sat_x0", 12.981),
        ("frac_quench_sat_k", 4.922),
        ("frac_quench_sat_ylo", 0.650),
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


DIFFSTARPOP_FITS_SMDPL_DIFFSTARPOP_PARAMS = DiffstarPopParams(
    SFH_PDF_QUENCH_PARAMS, DEFAULT_SATQUENCHPOP_PARAMS
)

_U_PNAMES = ["u_" + key for key in DIFFSTARPOP_FITS_SMDPL_DIFFSTARPOP_PARAMS._fields]
DiffstarPopUParams = namedtuple("DiffstarPopUParams", _U_PNAMES)

DIFFSTARPOP_FITS_SMDPL_DIFFSTARPOP_U_PARAMS = get_unbounded_diffstarpop_params(
    DIFFSTARPOP_FITS_SMDPL_DIFFSTARPOP_PARAMS
)
