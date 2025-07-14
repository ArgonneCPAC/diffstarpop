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
        ("mean_ulgm_mseq_xtp", 12.049),
        ("mean_ulgm_mseq_ytp", 11.798),
        ("mean_ulgm_mseq_lo", 1.247),
        ("mean_ulgm_mseq_hi", -0.089),
        ("mean_ulgy_mseq_int", 0.117),
        ("mean_ulgy_mseq_slp", 0.626),
        ("mean_ul_mseq_int", -0.691),
        ("mean_ul_mseq_slp", -0.042),
        ("mean_utau_mseq_int", 1.625),
        ("mean_utau_mseq_slp", -10.712),
        ("mean_ulgm_qseq_xtp", 11.877),
        ("mean_ulgm_qseq_ytp", 11.729),
        ("mean_ulgm_qseq_lo", 2.299),
        ("mean_ulgm_qseq_hi", 0.386),
        ("mean_ulgy_qseq_int", 0.153),
        ("mean_ulgy_qseq_slp", 0.111),
        ("mean_ul_qseq_int", -0.966),
        ("mean_ul_qseq_slp", 1.670),
        ("mean_utau_qseq_int", 4.446),
        ("mean_utau_qseq_slp", -9.239),
        ("mean_uqt_int", 0.974),
        ("mean_uqt_slp", 0.043),
        ("mean_uqs_int", 0.160),
        ("mean_uqs_slp", 2.336),
        ("mean_udrop_int", -1.921),
        ("mean_udrop_slp", 0.595),
        ("mean_urej_int", -2.087),
        ("mean_urej_slp", -0.556),
    ]
)

SFH_PDF_QUENCH_COV_MS_BLOCK_PDICT = OrderedDict(
    [
        ("std_ulgm_mseq_int", 0.249),
        ("std_ulgm_mseq_slp", -0.073),
        ("std_ulgy_mseq_int", 0.394),
        ("std_ulgy_mseq_slp", 0.256),
        ("std_ul_mseq_int", 0.076),
        ("std_ul_mseq_slp", 0.059),
        ("std_utau_mseq_int", 2.395),
        ("std_utau_mseq_slp", 2.357),
        ("std_ulgm_qseq_int", 0.142),
        ("std_ulgm_qseq_slp", -0.073),
        ("std_ulgy_qseq_int", 0.255),
        ("std_ulgy_qseq_slp", 0.009),
        ("std_ul_qseq_int", 0.237),
        ("std_ul_qseq_slp", -0.642),
        ("std_utau_qseq_int", 2.201),
        ("std_utau_qseq_slp", 2.730),
    ]
)

SFH_PDF_QUENCH_COV_Q_BLOCK_PDICT = OrderedDict(
    [
        ("std_uqt_int", 0.093),
        ("std_uqt_slp", 0.052),
        ("std_uqs_int", 0.564),
        ("std_uqs_slp", 0.066),
        ("std_udrop_int", 0.969),
        ("std_udrop_slp", -0.690),
        ("std_urej_int", 1.544),
        ("std_urej_slp", -0.201),
    ]
)

SFH_PDF_FRAC_QUENCH_PDICT = OrderedDict(
    [
        ("frac_quench_cen_x0_tpeak", 8.379),
        ("frac_quench_cen_k_tpeak", 7.612),
        ("frac_quench_cen_x0_ylotpeak", 13.803),
        ("frac_quench_cen_x0_yhitpeak", 12.037),
        ("frac_quench_cen_ylo_ylotpeak", 0.859),
        ("frac_quench_cen_ylo_yhitpeak", 0.012),
        ("frac_quench_cen_k", 1.165),
        ("frac_quench_cen_yhi", 0.971),
        ("frac_quench_sat_x0_tpeak", 6.149),
        ("frac_quench_sat_k_tpeak", 1.415),
        ("frac_quench_sat_x0_ylotpeak", 13.195),
        ("frac_quench_sat_x0_yhitpeak", 11.198),
        ("frac_quench_sat_ylo_ylotpeak", 0.435),
        ("frac_quench_sat_ylo_yhitpeak", 0.045),
        ("frac_quench_sat_k", 4.587),
        ("frac_quench_sat_yhi", 0.883),
    ]
)

DELTA_UQT_PDICT = OrderedDict(
    [
        ("delta_uqt_x0", 6.457),
        ("delta_uqt_k", 0.537),
        ("delta_uqt_ylo", -0.317),
        ("delta_uqt_yhi", 0.100),
        ("delta_uqt_slope", -0.070),
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
