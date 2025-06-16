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
        ("mean_ulgm_mseq_xtp", 11.982),
        ("mean_ulgm_mseq_ytp", 11.300),
        ("mean_ulgm_mseq_lo", 4.400),
        ("mean_ulgm_mseq_hi", 0.297),
        ("mean_ulgy_mseq_int", -0.900),
        ("mean_ulgy_mseq_slp", -0.412),
        ("mean_ul_mseq_int", 0.093),
        ("mean_ul_mseq_slp", 1.191),
        ("mean_utau_mseq_int", 4.691),
        ("mean_utau_mseq_slp", -4.223),
        ("mean_ulgm_qseq_xtp", 12.127),
        ("mean_ulgm_qseq_ytp", 11.344),
        ("mean_ulgm_qseq_lo", 1.447),
        ("mean_ulgm_qseq_hi", 0.227),
        ("mean_ulgy_qseq_int", -0.900),
        ("mean_ulgy_qseq_slp", -0.244),
        ("mean_ul_qseq_int", 0.198),
        ("mean_ul_qseq_slp", 0.108),
        ("mean_utau_qseq_int", 6.500),
        ("mean_utau_qseq_slp", 5.932),
        ("mean_uqt_int", 0.990),
        ("mean_uqt_slp", -0.073),
        ("mean_uqs_int", -0.120),
        ("mean_uqs_slp", 0.174),
        ("mean_udrop_int", -1.376),
        ("mean_udrop_slp", 0.178),
        ("mean_urej_int", -0.235),
        ("mean_urej_slp", -0.396),
    ]
)

SFH_PDF_QUENCH_COV_MS_BLOCK_PDICT = OrderedDict(
    [
        ("std_ulgm_mseq_int", 0.331),
        ("std_ulgm_mseq_slp", -0.078),
        ("std_ulgy_mseq_int", 0.367),
        ("std_ulgy_mseq_slp", -0.022),
        ("std_ul_mseq_int", 0.430),
        ("std_ul_mseq_slp", 0.302),
        ("std_utau_mseq_int", 4.300),
        ("std_utau_mseq_slp", -0.529),
        ("std_ulgm_qseq_int", 0.455),
        ("std_ulgm_qseq_slp", -0.900),
        ("std_ulgy_qseq_int", 0.494),
        ("std_ulgy_qseq_slp", -0.025),
        ("std_ul_qseq_int", 0.447),
        ("std_ul_qseq_slp", 0.094),
        ("std_utau_qseq_int", 3.173),
        ("std_utau_qseq_slp", 0.538),
    ]
)

SFH_PDF_QUENCH_COV_Q_BLOCK_PDICT = OrderedDict(
    [
        ("std_uqt_int", 0.115),
        ("std_uqt_slp", 0.069),
        ("std_uqs_int", 0.700),
        ("std_uqs_slp", 0.137),
        ("std_udrop_int", 0.639),
        ("std_udrop_slp", -0.117),
        ("std_urej_int", 0.931),
        ("std_urej_slp", -0.201),
    ]
)

SFH_PDF_FRAC_QUENCH_PDICT = OrderedDict(
    [
        ("frac_quench_cen_x0", 12.950),
        ("frac_quench_cen_k", 1.769),
        ("frac_quench_cen_ylo", 0.002),
        ("frac_quench_cen_yhi", 0.827),
        ("frac_quench_sat_x0", 12.877),
        ("frac_quench_sat_k", 0.780),
        ("frac_quench_sat_ylo", 0.002),
        ("frac_quench_sat_yhi", 0.998),
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


DIFFSTARFITS_GALACTICUS_IN_DIFFSTARPOP_PARAMS = DiffstarPopParams(
    SFH_PDF_QUENCH_PARAMS, DEFAULT_SATQUENCHPOP_PARAMS
)

_U_PNAMES = [
    "u_" + key for key in DIFFSTARFITS_GALACTICUS_IN_DIFFSTARPOP_PARAMS._fields
]
DiffstarPopUParams = namedtuple("DiffstarPopUParams", _U_PNAMES)

DIFFSTARFITS_GALACTICUS_IN_DIFFSTARPOP_U_PARAMS = get_unbounded_diffstarpop_params(
    DIFFSTARFITS_GALACTICUS_IN_DIFFSTARPOP_PARAMS
)
