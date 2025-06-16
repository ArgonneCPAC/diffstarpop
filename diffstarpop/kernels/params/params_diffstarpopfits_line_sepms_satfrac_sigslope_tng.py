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
        ("mean_ulgm_mseq_xtp", 11.837),
        ("mean_ulgm_mseq_ytp", 11.804),
        ("mean_ulgm_mseq_lo", 0.638),
        ("mean_ulgm_mseq_hi", -0.438),
        ("mean_ulgy_mseq_int", 0.100),
        ("mean_ulgy_mseq_slp", 0.687),
        ("mean_ul_mseq_int", 2.446),
        ("mean_ul_mseq_slp", 4.032),
        ("mean_utau_mseq_int", -11.569),
        ("mean_utau_mseq_slp", -16.867),
        ("mean_ulgm_qseq_xtp", 11.394),
        ("mean_ulgm_qseq_ytp", 11.184),
        ("mean_ulgm_qseq_lo", 3.095),
        ("mean_ulgm_qseq_hi", 0.401),
        ("mean_ulgy_qseq_int", 0.427),
        ("mean_ulgy_qseq_slp", -0.115),
        ("mean_ul_qseq_int", -1.483),
        ("mean_ul_qseq_slp", -2.092),
        ("mean_utau_qseq_int", 2.843),
        ("mean_utau_qseq_slp", -9.511),
        ("mean_uqt_int", 0.881),
        ("mean_uqt_slp", -0.549),
        ("mean_uqs_int", -0.610),
        ("mean_uqs_slp", -0.013),
        ("mean_udrop_int", -2.809),
        ("mean_udrop_slp", 0.966),
        ("mean_urej_int", -2.333),
        ("mean_urej_slp", 0.735),
    ]
)

SFH_PDF_QUENCH_COV_MS_BLOCK_PDICT = OrderedDict(
    [
        ("std_ulgm_mseq_int", 0.072),
        ("std_ulgm_mseq_slp", 0.059),
        ("std_ulgy_mseq_int", 0.257),
        ("std_ulgy_mseq_slp", -0.052),
        ("std_ul_mseq_int", 0.088),
        ("std_ul_mseq_slp", 0.055),
        ("std_utau_mseq_int", 1.519),
        ("std_utau_mseq_slp", -1.619),
        ("std_ulgm_qseq_int", 0.136),
        ("std_ulgm_qseq_slp", -0.019),
        ("std_ulgy_qseq_int", 0.218),
        ("std_ulgy_qseq_slp", -0.147),
        ("std_ul_qseq_int", 0.148),
        ("std_ul_qseq_slp", 0.058),
        ("std_utau_qseq_int", 1.562),
        ("std_utau_qseq_slp", 1.331),
    ]
)

SFH_PDF_QUENCH_COV_Q_BLOCK_PDICT = OrderedDict(
    [
        ("std_uqt_int", 0.100),
        ("std_uqt_slp", -0.062),
        ("std_uqs_int", 0.386),
        ("std_uqs_slp", -0.252),
        ("std_udrop_int", 0.138),
        ("std_udrop_slp", 0.580),
        ("std_urej_int", 0.602),
        ("std_urej_slp", -0.729),
    ]
)

SFH_PDF_FRAC_QUENCH_PDICT = OrderedDict(
    [
        ("frac_quench_cen_x0", 12.056),
        ("frac_quench_cen_k", 4.946),
        ("frac_quench_cen_ylo", 0.007),
        ("frac_quench_cen_yhi", 0.992),
        ("frac_quench_sat_x0", 11.990),
        ("frac_quench_sat_k", 2.353),
        ("frac_quench_sat_ylo", 0.001),
        ("frac_quench_sat_yhi", 0.994),
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


DIFFSTARPOP_FITS_TNG_DIFFSTARPOP_PARAMS = DiffstarPopParams(
    SFH_PDF_QUENCH_PARAMS, DEFAULT_SATQUENCHPOP_PARAMS
)

_U_PNAMES = ["u_" + key for key in DIFFSTARPOP_FITS_TNG_DIFFSTARPOP_PARAMS._fields]
DiffstarPopUParams = namedtuple("DiffstarPopUParams", _U_PNAMES)

DIFFSTARPOP_FITS_TNG_DIFFSTARPOP_U_PARAMS = get_unbounded_diffstarpop_params(
    DIFFSTARPOP_FITS_TNG_DIFFSTARPOP_PARAMS
)
