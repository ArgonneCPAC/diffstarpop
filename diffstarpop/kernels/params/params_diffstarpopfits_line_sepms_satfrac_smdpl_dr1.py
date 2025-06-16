from collections import OrderedDict, namedtuple

import typing
from jax import numpy as jnp

from ..satquenchpop_model import (
    DEFAULT_SATQUENCHPOP_PARAMS,
)
from ..defaults_tpeak_line_sepms_satfrac import get_unbounded_diffstarpop_params

SFH_PDF_QUENCH_MU_PDICT = OrderedDict(
    [
        ("mean_ulgm_mseq_int", 11.890),
        ("mean_ulgm_mseq_slp", -0.112),
        ("mean_ulgy_mseq_int", 0.420),
        ("mean_ulgy_mseq_slp", 1.194),
        ("mean_ul_mseq_int", 0.059),
        ("mean_ul_mseq_slp", 1.259),
        ("mean_utau_mseq_int", -1.505),
        ("mean_utau_mseq_slp", -11.295),
        ("mean_ulgm_qseq_int", 11.998),
        ("mean_ulgm_qseq_slp", 0.289),
        ("mean_ulgy_qseq_int", 0.338),
        ("mean_ulgy_qseq_slp", 0.583),
        ("mean_ul_qseq_int", -0.479),
        ("mean_ul_qseq_slp", 2.143),
        ("mean_utau_qseq_int", 5.636),
        ("mean_utau_qseq_slp", -6.363),
        ("mean_uqt_int", 1.015),
        ("mean_uqt_slp", -0.102),
        ("mean_uqs_int", -0.061),
        ("mean_uqs_slp", 1.736),
        ("mean_udrop_int", -2.130),
        ("mean_udrop_slp", 0.957),
        ("mean_urej_int", -0.793),
        ("mean_urej_slp", -1.582),
    ]
)

SFH_PDF_QUENCH_COV_MS_BLOCK_PDICT = OrderedDict(
    [
        ("std_ulgm_mseq_int", 0.150),
        ("std_ulgm_mseq_slp", 0.154),
        ("std_ulgy_mseq_int", 0.186),
        ("std_ulgy_mseq_slp", 0.147),
        ("std_ul_mseq_int", 0.208),
        ("std_ul_mseq_slp", -0.533),
        ("std_utau_mseq_int", 2.917),
        ("std_utau_mseq_slp", 0.080),
        ("std_ulgm_qseq_int", 0.261),
        ("std_ulgm_qseq_slp", -0.554),
        ("std_ulgy_qseq_int", 0.143),
        ("std_ulgy_qseq_slp", 0.017),
        ("std_ul_qseq_int", 0.145),
        ("std_ul_qseq_slp", 0.053),
        ("std_utau_qseq_int", 2.140),
        ("std_utau_qseq_slp", 1.614),
    ]
)

SFH_PDF_QUENCH_COV_Q_BLOCK_PDICT = OrderedDict(
    [
        ("std_uqt_int", 0.110),
        ("std_uqt_slp", 0.084),
        ("std_uqs_int", 0.371),
        ("std_uqs_slp", 0.090),
        ("std_udrop_int", 0.483),
        ("std_udrop_slp", -0.214),
        ("std_urej_int", 0.956),
        ("std_urej_slp", -0.154),
    ]
)

SFH_PDF_FRAC_QUENCH_PDICT = OrderedDict(
    [
        ("frac_quench_cen_x0", 11.536),
        ("frac_quench_cen_k", 3.622),
        ("frac_quench_cen_ylo", 0.002),
        ("frac_quench_cen_yhi", 0.864),
        ("frac_quench_sat_x0", 11.599),
        ("frac_quench_sat_k", 1.832),
        ("frac_quench_sat_ylo", 0.157),
        ("frac_quench_sat_yhi", 0.934),
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
