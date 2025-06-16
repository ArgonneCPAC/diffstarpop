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
        ("mean_ulgm_mseq_slp", 0.120),
        ("mean_ulgy_mseq_int", 0.056),
        ("mean_ulgy_mseq_slp", 0.044),
        ("mean_ul_mseq_int", -0.146),
        ("mean_ul_mseq_slp", 0.035),
        ("mean_utau_mseq_int", 0.887),
        ("mean_utau_mseq_slp", -12.097),
        ("mean_ulgm_qseq_int", 11.700),
        ("mean_ulgm_qseq_slp", 0.637),
        ("mean_ulgy_qseq_int", -0.007),
        ("mean_ulgy_qseq_slp", 0.004),
        ("mean_ul_qseq_int", -0.250),
        ("mean_ul_qseq_slp", 0.213),
        ("mean_utau_qseq_int", 0.772),
        ("mean_utau_qseq_slp", -12.111),
        ("mean_uqt_int", 0.970),
        ("mean_uqt_slp", -0.008),
        ("mean_uqs_int", -0.423),
        ("mean_uqs_slp", 0.139),
        ("mean_udrop_int", -1.297),
        ("mean_udrop_slp", 0.197),
        ("mean_urej_int", 0.013),
        ("mean_urej_slp", -0.370),
    ]
)

SFH_PDF_QUENCH_COV_MS_BLOCK_PDICT = OrderedDict(
    [
        ("std_ulgm_mseq_int", 0.250),
        ("std_ulgm_mseq_slp", -0.002),
        ("std_ulgy_mseq_int", 0.279),
        ("std_ulgy_mseq_slp", -0.048),
        ("std_ul_mseq_int", 0.287),
        ("std_ul_mseq_slp", 0.015),
        ("std_utau_mseq_int", 3.739),
        ("std_utau_mseq_slp", 1.254),
        ("std_ulgm_qseq_int", 0.244),
        ("std_ulgm_qseq_slp", -0.083),
        ("std_ulgy_qseq_int", 0.320),
        ("std_ulgy_qseq_slp", -0.089),
        ("std_ul_qseq_int", 0.252),
        ("std_ul_qseq_slp", 0.091),
        ("std_utau_qseq_int", 4.105),
        ("std_utau_qseq_slp", 1.401),
    ]
)

SFH_PDF_QUENCH_COV_Q_BLOCK_PDICT = OrderedDict(
    [
        ("std_uqt_int", 0.102),
        ("std_uqt_slp", 0.018),
        ("std_uqs_int", 0.604),
        ("std_uqs_slp", 0.009),
        ("std_udrop_int", 0.574),
        ("std_udrop_slp", -0.133),
        ("std_urej_int", 1.072),
        ("std_urej_slp", -0.149),
    ]
)

SFH_PDF_FRAC_QUENCH_PDICT = OrderedDict(
    [
        ("frac_quench_cen_x0", 11.965),
        ("frac_quench_cen_k", 4.390),
        ("frac_quench_cen_ylo", 0.002),
        ("frac_quench_cen_yhi", 0.782),
        ("frac_quench_sat_x0", 11.955),
        ("frac_quench_sat_k", 2.150),
        ("frac_quench_sat_ylo", 0.127),
        ("frac_quench_sat_yhi", 0.854),
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


DIFFSTARFITS_SMDPL_DR1_DIFFSTARPOP_PARAMS = DiffstarPopParams(
    SFH_PDF_QUENCH_PARAMS, DEFAULT_SATQUENCHPOP_PARAMS
)

_U_PNAMES = ["u_" + key for key in DIFFSTARFITS_SMDPL_DR1_DIFFSTARPOP_PARAMS._fields]
DiffstarPopUParams = namedtuple("DiffstarPopUParams", _U_PNAMES)

DIFFSTARFITS_SMDPL_DR1_DIFFSTARPOP_U_PARAMS = get_unbounded_diffstarpop_params(
    DIFFSTARFITS_SMDPL_DR1_DIFFSTARPOP_PARAMS
)
