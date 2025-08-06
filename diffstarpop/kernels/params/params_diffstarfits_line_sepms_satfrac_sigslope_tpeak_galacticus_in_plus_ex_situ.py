from collections import OrderedDict, namedtuple

import typing
from jax import numpy as jnp

from ..satquenchpop_model import (
    DEFAULT_SATQUENCHPOP_PARAMS,
)
from ..defaults_tpeak_line_sepms_satfrac_sigslope_tpeak import get_unbounded_diffstarpop_params

SFH_PDF_QUENCH_MU_PDICT = OrderedDict([
    ('mean_ulgm_mseq_xtp', 12.471),
    ('mean_ulgm_mseq_ytp', 12.354),
    ('mean_ulgm_mseq_lo', 1.078),
    ('mean_ulgm_mseq_hi', -0.082),
    ('mean_ulgy_mseq_int', 0.256),
    ('mean_ulgy_mseq_slp', -0.004),
    ('mean_ul_mseq_int', -0.141),
    ('mean_ul_mseq_slp', 0.107),
    ('mean_utau_mseq_int', 4.862),
    ('mean_utau_mseq_slp', 0.091),
    ('mean_ulgm_qseq_xtp', 11.689),
    ('mean_ulgm_qseq_ytp', 11.436),
    ('mean_ulgm_qseq_lo', 4.400),
    ('mean_ulgm_qseq_hi', 0.400),
    ('mean_ulgy_qseq_int', 0.455),
    ('mean_ulgy_qseq_slp', 0.291),
    ('mean_ul_qseq_int', -0.502),
    ('mean_ul_qseq_slp', -0.460),
    ('mean_utau_qseq_int', 6.377),
    ('mean_utau_qseq_slp', 5.721),
    ('mean_uqt_int', 0.988),
    ('mean_uqt_slp', 0.041),
    ('mean_uqs_int', -0.368),
    ('mean_uqs_slp', 0.516),
    ('mean_udrop_int', -0.756),
    ('mean_udrop_slp', -0.053),
    ('mean_urej_int', 0.416),
    ('mean_urej_slp', -0.522),
])

SFH_PDF_QUENCH_COV_MS_BLOCK_PDICT = OrderedDict([
    ('std_ulgm_mseq_int', 0.238),
    ('std_ulgm_mseq_slp', -0.012),
    ('std_ulgy_mseq_int', 0.185),
    ('std_ulgy_mseq_slp', -0.010),
    ('std_ul_mseq_int', 0.300),
    ('std_ul_mseq_slp', 0.045),
    ('std_utau_mseq_int', 5.058),
    ('std_utau_mseq_slp', -1.332),
    ('std_ulgm_qseq_int', 0.253),
    ('std_ulgm_qseq_slp', -0.007),
    ('std_ulgy_qseq_int', 0.078),
    ('std_ulgy_qseq_slp', -0.086),
    ('std_ul_qseq_int', 0.115),
    ('std_ul_qseq_slp', -0.021),
    ('std_utau_qseq_int', 2.706),
    ('std_utau_qseq_slp', 1.002),
])

SFH_PDF_QUENCH_COV_Q_BLOCK_PDICT = OrderedDict([
    ('std_uqt_int', 0.112),
    ('std_uqt_slp', -0.016),
    ('std_uqs_int', 0.606),
    ('std_uqs_slp', -0.066),
    ('std_udrop_int', 0.794),
    ('std_udrop_slp', -0.161),
    ('std_urej_int', 1.038),
    ('std_urej_slp', 0.012),
])

SFH_PDF_FRAC_QUENCH_PDICT = OrderedDict([
    ('frac_quench_cen_x0_tpeak', 7.000),
    ('frac_quench_cen_k_tpeak', 2.000),
    ('frac_quench_cen_x0_ylotpeak', 12.906),
    ('frac_quench_cen_x0_yhitpeak', 12.964),
    ('frac_quench_cen_ylo_ylotpeak', 0.267),
    ('frac_quench_cen_ylo_yhitpeak', 0.105),
    ('frac_quench_cen_k', 3.848),
    ('frac_quench_cen_yhi', 0.971),
    ('frac_quench_sat_x0_tpeak', 7.000),
    ('frac_quench_sat_k_tpeak', 2.000),
    ('frac_quench_sat_x0_ylotpeak', 12.906),
    ('frac_quench_sat_x0_yhitpeak', 12.964),
    ('frac_quench_sat_ylo_ylotpeak', 0.267),
    ('frac_quench_sat_ylo_yhitpeak', 0.105),
    ('frac_quench_sat_k', 3.848),
    ('frac_quench_sat_yhi', 0.971),
])
DELTA_UQT_PDICT = OrderedDict([
    ('delta_uqt_x0', 1.180),
    ('delta_uqt_k', 4.651),
    ('delta_uqt_ylo', -0.370),
    ('delta_uqt_yhi', -0.021),
    ('delta_uqt_slope', -0.035),
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

DIFFSTARFITS_GALACTICUS_INPLUSEX_DIFFSTARPOP_PARAMS = DiffstarPopParams(
    SFH_PDF_QUENCH_PARAMS, DEFAULT_SATQUENCHPOP_PARAMS
)

_U_PNAMES = ['u_' + key for key in DIFFSTARFITS_GALACTICUS_INPLUSEX_DIFFSTARPOP_PARAMS._fields]
DiffstarPopUParams = namedtuple('DiffstarPopUParams', _U_PNAMES)

DIFFSTARFITS_GALACTICUS_INPLUSEX_DIFFSTARPOP_U_PARAMS = get_unbounded_diffstarpop_params(
    DIFFSTARFITS_GALACTICUS_INPLUSEX_DIFFSTARPOP_PARAMS
)
