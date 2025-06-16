from collections import OrderedDict, namedtuple

import typing
from jax import numpy as jnp

from ..satquenchpop_model import (
    DEFAULT_SATQUENCHPOP_PARAMS,
)
from ..defaults_tpeak_line_sepms_satfrac import get_unbounded_diffstarpop_params

SFH_PDF_QUENCH_MU_PDICT = OrderedDict(
    [
        ("mean_ulgm_mseq_int", 12.319),
        ("mean_ulgm_mseq_slp", -0.499),
        ("mean_ulgy_mseq_int", -0.232),
        ("mean_ulgy_mseq_slp", -5.394),
        ("mean_ul_mseq_int", 0.038),
        ("mean_ul_mseq_slp", -0.276),
        ("mean_utau_mseq_int", -10.342),
        ("mean_utau_mseq_slp", -14.223),
        ("mean_ulgm_qseq_int", 12.055),
        ("mean_ulgm_qseq_slp", -0.297),
        ("mean_ulgy_qseq_int", 0.373),
        ("mean_ulgy_qseq_slp", 0.959),
        ("mean_ul_qseq_int", 0.515),
        ("mean_ul_qseq_slp", 3.346),
        ("mean_utau_qseq_int", 5.805),
        ("mean_utau_qseq_slp", -6.292),
        ("mean_uqt_int", 0.919),
        ("mean_uqt_slp", -0.316),
        ("mean_uqs_int", -0.375),
        ("mean_uqs_slp", 0.654),
        ("mean_udrop_int", -2.401),
        ("mean_udrop_slp", -2.260),
        ("mean_urej_int", -0.887),
        ("mean_urej_slp", -0.260),
    ]
)

SFH_PDF_QUENCH_COV_MS_BLOCK_PDICT = OrderedDict(
    [
        ("std_ulgm_mseq_int", 0.423),
        ("std_ulgm_mseq_slp", -0.351),
        ("std_ulgy_mseq_int", 0.278),
        ("std_ulgy_mseq_slp", -0.200),
        ("std_ul_mseq_int", 0.153),
        ("std_ul_mseq_slp", 0.088),
        ("std_utau_mseq_int", 7.467),
        ("std_utau_mseq_slp", -0.121),
        ("std_ulgm_qseq_int", 0.294),
        ("std_ulgm_qseq_slp", -0.051),
        ("std_ulgy_qseq_int", 0.283),
        ("std_ulgy_qseq_slp", 0.038),
        ("std_ul_qseq_int", 0.232),
        ("std_ul_qseq_slp", -0.091),
        ("std_utau_qseq_int", 2.216),
        ("std_utau_qseq_slp", 1.185),
    ]
)

SFH_PDF_QUENCH_COV_Q_BLOCK_PDICT = OrderedDict(
    [
        ("std_uqt_int", 0.108),
        ("std_uqt_slp", -0.013),
        ("std_uqs_int", 0.204),
        ("std_uqs_slp", -0.368),
        ("std_udrop_int", 0.920),
        ("std_udrop_slp", 0.074),
        ("std_urej_int", 1.247),
        ("std_urej_slp", 0.351),
    ]
)

SFH_PDF_FRAC_QUENCH_PDICT = OrderedDict(
    [
        ("frac_quench_cen_x0", 11.141),
        ("frac_quench_cen_k", 2.585),
        ("frac_quench_cen_ylo", 0.151),
        ("frac_quench_cen_yhi", 0.998),
        ("frac_quench_sat_x0", 12.899),
        ("frac_quench_sat_k", 4.381),
        ("frac_quench_sat_ylo", 0.513),
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
