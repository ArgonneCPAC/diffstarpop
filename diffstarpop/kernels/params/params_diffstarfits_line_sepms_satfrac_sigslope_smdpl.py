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
        ("mean_ulgm_mseq_xtp", 12.545),
        ("mean_ulgm_mseq_ytp", 12.210),
        ("mean_ulgm_mseq_lo", 0.601),
        ("mean_ulgm_mseq_hi", -0.482),
        ("mean_ulgy_mseq_int", -0.025),
        ("mean_ulgy_mseq_slp", -0.181),
        ("mean_ul_mseq_int", 0.097),
        ("mean_ul_mseq_slp", 0.241),
        ("mean_utau_mseq_int", -0.239),
        ("mean_utau_mseq_slp", -15.541),
        ("mean_ulgm_qseq_xtp", 12.308),
        ("mean_ulgm_qseq_ytp", 12.165),
        ("mean_ulgm_qseq_lo", 0.878),
        ("mean_ulgm_qseq_hi", 0.050),
        ("mean_ulgy_qseq_int", -0.079),
        ("mean_ulgy_qseq_slp", 0.045),
        ("mean_ul_qseq_int", 0.032),
        ("mean_ul_qseq_slp", 0.350),
        ("mean_utau_qseq_int", -0.735),
        ("mean_utau_qseq_slp", -13.700),
        ("mean_uqt_int", 0.897),
        ("mean_uqt_slp", -0.176),
        ("mean_uqs_int", -0.273),
        ("mean_uqs_slp", 0.612),
        ("mean_udrop_int", -1.915),
        ("mean_udrop_slp", -0.300),
        ("mean_urej_int", 0.040),
        ("mean_urej_slp", -1.100),
    ]
)

SFH_PDF_QUENCH_COV_MS_BLOCK_PDICT = OrderedDict(
    [
        ("std_ulgm_mseq_int", 0.262),
        ("std_ulgm_mseq_slp", 0.057),
        ("std_ulgy_mseq_int", 0.332),
        ("std_ulgy_mseq_slp", 0.033),
        ("std_ul_mseq_int", 0.330),
        ("std_ul_mseq_slp", 0.129),
        ("std_utau_mseq_int", 6.935),
        ("std_utau_mseq_slp", -0.295),
        ("std_ulgm_qseq_int", 0.290),
        ("std_ulgm_qseq_slp", 0.049),
        ("std_ulgy_qseq_int", 0.465),
        ("std_ulgy_qseq_slp", 0.068),
        ("std_ul_qseq_int", 0.359),
        ("std_ul_qseq_slp", 0.176),
        ("std_utau_qseq_int", 8.357),
        ("std_utau_qseq_slp", 2.700),
    ]
)

SFH_PDF_QUENCH_COV_Q_BLOCK_PDICT = OrderedDict(
    [
        ("std_uqt_int", 0.128),
        ("std_uqt_slp", 0.026),
        ("std_uqs_int", 0.595),
        ("std_uqs_slp", -0.049),
        ("std_udrop_int", 0.726),
        ("std_udrop_slp", -0.033),
        ("std_urej_int", 0.939),
        ("std_urej_slp", -0.040),
    ]
)

SFH_PDF_FRAC_QUENCH_PDICT = OrderedDict(
    [
        ("frac_quench_cen_x0", 12.118),
        ("frac_quench_cen_k", 3.512),
        ("frac_quench_cen_ylo", 0.002),
        ("frac_quench_cen_yhi", 0.985),
        ("frac_quench_sat_x0", 12.593),
        ("frac_quench_sat_k", 3.047),
        ("frac_quench_sat_ylo", 0.528),
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


DIFFSTARFITS_SMDPL_DIFFSTARPOP_PARAMS = DiffstarPopParams(
    SFH_PDF_QUENCH_PARAMS, DEFAULT_SATQUENCHPOP_PARAMS
)

_U_PNAMES = ["u_" + key for key in DIFFSTARFITS_SMDPL_DIFFSTARPOP_PARAMS._fields]
DiffstarPopUParams = namedtuple("DiffstarPopUParams", _U_PNAMES)

DIFFSTARFITS_SMDPL_DIFFSTARPOP_U_PARAMS = get_unbounded_diffstarpop_params(
    DIFFSTARFITS_SMDPL_DIFFSTARPOP_PARAMS
)
