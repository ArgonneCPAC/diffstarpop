from collections import OrderedDict, namedtuple

import typing
from jax import numpy as jnp

from ..satquenchpop_model import (
    DEFAULT_SATQUENCHPOP_PARAMS,
)
from ..defaults_tpeak_line_sepms_satfrac_sigslope_tpeak import get_unbounded_diffstarpop_params

SFH_PDF_QUENCH_MU_PDICT = OrderedDict([
    ('mean_ulgm_mseq_xtp', 12.654),
    ('mean_ulgm_mseq_ytp', 12.514),
    ('mean_ulgm_mseq_lo', 1.006),
    ('mean_ulgm_mseq_hi', -0.828),
    ('mean_ulgy_mseq_int', 0.011),
    ('mean_ulgy_mseq_slp', -0.220),
    ('mean_ul_mseq_int', 0.065),
    ('mean_ul_mseq_slp', 0.349),
    ('mean_utau_mseq_int', 5.620),
    ('mean_utau_mseq_slp', -0.049),
    ('mean_ulgm_qseq_xtp', 11.874),
    ('mean_ulgm_qseq_ytp', 11.814),
    ('mean_ulgm_qseq_lo', 3.470),
    ('mean_ulgm_qseq_hi', 0.071),
    ('mean_ulgy_qseq_int', 0.305),
    ('mean_ulgy_qseq_slp', 0.187),
    ('mean_ul_qseq_int', -0.363),
    ('mean_ul_qseq_slp', -0.392),
    ('mean_utau_qseq_int', 6.845),
    ('mean_utau_qseq_slp', 6.012),
    ('mean_uqt_int', 0.983),
    ('mean_uqt_slp', -0.142),
    ('mean_uqs_int', -0.153),
    ('mean_uqs_slp', 0.199),
    ('mean_udrop_int', -1.086),
    ('mean_udrop_slp', 0.102),
    ('mean_urej_int', 0.028),
    ('mean_urej_slp', -0.187),
])

SFH_PDF_QUENCH_COV_MS_BLOCK_PDICT = OrderedDict([
    ('std_ulgm_mseq_int', 0.233),
    ('std_ulgm_mseq_slp', -0.053),
    ('std_ulgy_mseq_int', 0.244),
    ('std_ulgy_mseq_slp', 0.055),
    ('std_ul_mseq_int', 0.250),
    ('std_ul_mseq_slp', 0.101),
    ('std_utau_mseq_int', 3.507),
    ('std_utau_mseq_slp', 0.870),
    ('std_ulgm_qseq_int', 0.300),
    ('std_ulgm_qseq_slp', 0.067),
    ('std_ulgy_qseq_int', 0.167),
    ('std_ulgy_qseq_slp', -0.054),
    ('std_ul_qseq_int', 0.126),
    ('std_ul_qseq_slp', -0.086),
    ('std_utau_qseq_int', 3.028),
    ('std_utau_qseq_slp', 1.182),
])

SFH_PDF_QUENCH_COV_Q_BLOCK_PDICT = OrderedDict([
    ('std_uqt_int', 0.119),
    ('std_uqt_slp', 0.045),
    ('std_uqs_int', 0.577),
    ('std_uqs_slp', 0.030),
    ('std_udrop_int', 0.616),
    ('std_udrop_slp', 0.136),
    ('std_urej_int', 1.023),
    ('std_urej_slp', 0.798),
])

SFH_PDF_FRAC_QUENCH_PDICT = OrderedDict([
    ('frac_quench_cen_x0_tpeak', 7.000),
    ('frac_quench_cen_k_tpeak', 2.000),
    ('frac_quench_cen_x0_ylotpeak', 11.470),
    ('frac_quench_cen_x0_yhitpeak', 12.915),
    ('frac_quench_cen_ylo_ylotpeak', 0.554),
    ('frac_quench_cen_ylo_yhitpeak', 0.131),
    ('frac_quench_cen_k', 3.848),
    ('frac_quench_cen_yhi', 0.971),
    ('frac_quench_sat_x0_tpeak', 7.000),
    ('frac_quench_sat_k_tpeak', 2.000),
    ('frac_quench_sat_x0_ylotpeak', 11.470),
    ('frac_quench_sat_x0_yhitpeak', 12.915),
    ('frac_quench_sat_ylo_ylotpeak', 0.554),
    ('frac_quench_sat_ylo_yhitpeak', 0.131),
    ('frac_quench_sat_k', 3.848),
    ('frac_quench_sat_yhi', 0.971),
])
DELTA_UQT_PDICT = OrderedDict([
    ('delta_uqt_x0', 11.117),
    ('delta_uqt_k', 0.011),
    ('delta_uqt_ylo', 0.545),
    ('delta_uqt_yhi', -0.588),
    ('delta_uqt_slope', 0.014),
])
SFH_PDF_QUENCH_PDICT = SFH_PDF_FRAC_QUENCH_PDICT.copy()
SFH_PDF_QUENCH_PDICT.update(SFH_PDF_QUENCH_MU_PDICT)
SFH_PDF_QUENCH_PDICT.update(SFH_PDF_QUENCH_COV_MS_BLOCK_PDICT)
SFH_PDF_QUENCH_PDICT.update(SFH_PDF_QUENCH_COV_Q_BLOCK_PDICT)
SFH_PDF_QUENCH_PDICT.update(DELTA_UQT_PDICT)

QseqParams = namedtuple('QseqParams', list(SFH_PDF_QUENCH_PDICT.keys()))
SFH_PDF_QUENCH_PARAMS = QseqParams(**SFH_PDF_QUENCH_PDICT)
_UPNAMES = ['u_' + key for key in QseqParams._fields]
QseqUParams = namedtuple('QseqUParams', _UPNAMES)


# Define a namedtuple container for the params of each component
class DiffstarPopParams(typing.NamedTuple):
    sfh_pdf_cens_params: jnp.array
    satquench_params: jnp.array

DIFFSTARFITS_GALACTICUS_IN_DIFFSTARPOP_PARAMS = DiffstarPopParams(
    SFH_PDF_QUENCH_PARAMS, DEFAULT_SATQUENCHPOP_PARAMS
)

_U_PNAMES = ['u_' + key for key in DIFFSTARFITS_GALACTICUS_IN_DIFFSTARPOP_PARAMS._fields]
DiffstarPopUParams = namedtuple('DiffstarPopUParams', _U_PNAMES)

DIFFSTARFITS_GALACTICUS_IN_DIFFSTARPOP_U_PARAMS = get_unbounded_diffstarpop_params(
    DIFFSTARFITS_GALACTICUS_IN_DIFFSTARPOP_PARAMS
)
