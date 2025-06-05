""" """

# flake8: noqa

import typing
from collections import namedtuple, OrderedDict

from .params_diffstarfits_line_sepms_satfrac_smdpl import (
    DIFFSTARFITS_SMDPL_DIFFSTARPOP_PARAMS as DIFFSTARFITS_SMDPL_DIFFSTARPOP_PARAMS_line_sepms_satfrac,
    DIFFSTARFITS_SMDPL_DIFFSTARPOP_U_PARAMS as DIFFSTARFITS_SMDPL_DIFFSTARPOP_U_PARAMS_line_sepms_satfrac,
)

from .params_diffstarfits_line_sepms_satfrac_sigslope_smdpl import (
    DIFFSTARFITS_SMDPL_DIFFSTARPOP_PARAMS as DIFFSTARFITS_SMDPL_DIFFSTARPOP_PARAMS_line_sepms_satfrac_sigslope,
    DIFFSTARFITS_SMDPL_DIFFSTARPOP_U_PARAMS as DIFFSTARFITS_SMDPL_DIFFSTARPOP_U_PARAMS_line_sepms_satfrac_sigslope,
)

sim_name_list = [
    "smdpl",
    "smdpl_DR1",
    "tng",
    "galacticus_in_situ",
    "galacticus_in_plus_ex_situ",
]


DiffstarPop_Params_Diffstarfits_line_sepms_satfrac = OrderedDict(
    smdpl=DIFFSTARFITS_SMDPL_DIFFSTARPOP_PARAMS_line_sepms_satfrac
)
DiffstarPop_UParams_Diffstarfits_line_sepms_satfrac = OrderedDict(
    smdpl=DIFFSTARFITS_SMDPL_DIFFSTARPOP_U_PARAMS_line_sepms_satfrac
)

DiffstarPop_Params_Diffstarfits_line_sepms_satfrac_sigslope = OrderedDict(
    smdpl=DIFFSTARFITS_SMDPL_DIFFSTARPOP_PARAMS_line_sepms_satfrac_sigslope
)
DiffstarPop_UParams_Diffstarfits_line_sepms_satfrac_sigslope = OrderedDict(
    smdpl=DIFFSTARFITS_SMDPL_DIFFSTARPOP_U_PARAMS_line_sepms_satfrac_sigslope
)
