""" """

# flake8: noqa

import typing
from collections import namedtuple, OrderedDict

# SMDPL
from .params_diffstarfits_line_sepms_satfrac_sigslope_smdpl import (
    DIFFSTARFITS_SMDPL_DIFFSTARPOP_PARAMS as DIFFSTARFITS_SMDPL_DIFFSTARPOP_PARAMS_line_sepms_satfrac_sigslope,
    DIFFSTARFITS_SMDPL_DIFFSTARPOP_U_PARAMS as DIFFSTARFITS_SMDPL_DIFFSTARPOP_U_PARAMS_line_sepms_satfrac_sigslope,
)

# SMDPL DR1

from .params_diffstarfits_line_sepms_satfrac_sigslope_smdpl_DR1 import (
    DIFFSTARFITS_SMDPL_DR1_DIFFSTARPOP_PARAMS as DIFFSTARFITS_SMDPL_DR1_DIFFSTARPOP_PARAMS_line_sepms_satfrac_sigslope,
    DIFFSTARFITS_SMDPL_DR1_DIFFSTARPOP_U_PARAMS as DIFFSTARFITS_SMDPL_DR1_DIFFSTARPOP_U_PARAMS_line_sepms_satfrac_sigslope,
)

# TNG

from .params_diffstarfits_line_sepms_satfrac_sigslope_tng import (
    DIFFSTARFITS_TNG_DIFFSTARPOP_PARAMS as DIFFSTARFITS_TNG_DIFFSTARPOP_PARAMS_line_sepms_satfrac_sigslope,
    DIFFSTARFITS_TNG_DIFFSTARPOP_U_PARAMS as DIFFSTARFITS_TNG_DIFFSTARPOP_U_PARAMS_line_sepms_satfrac_sigslope,
)

# Glacticus in situ

from .params_diffstarfits_line_sepms_satfrac_sigslope_galacticus_in_situ import (
    DIFFSTARFITS_GALACTICUS_IN_DIFFSTARPOP_PARAMS as DIFFSTARFITS_GALACTICUS_IN_DIFFSTARPOP_PARAMS_line_sepms_satfrac_sigslope,
    DIFFSTARFITS_GALACTICUS_IN_DIFFSTARPOP_U_PARAMS as DIFFSTARFITS_GALACTICUS_IN_DIFFSTARPOP_U_PARAMS_line_sepms_satfrac_sigslope,
)

# Glacticus in plus ex situ
from .params_diffstarfits_line_sepms_satfrac_sigslope_galacticus_in_plus_ex_situ import (
    DIFFSTARFITS_GALACTICUS_INPLUSEX_DIFFSTARPOP_PARAMS as DIFFSTARFITS_GALACTICUS_INPLUSEX_DIFFSTARPOP_PARAMS_line_sepms_satfrac_sigslope,
    DIFFSTARFITS_GALACTICUS_INPLUSEX_DIFFSTARPOP_U_PARAMS as DIFFSTARFITS_GALACTICUS_INPLUSEX_DIFFSTARPOP_U_PARAMS_line_sepms_satfrac_sigslope,
)

sim_name_list = [
    "smdpl",
    "smdpl_dr1",
    "tng",
    "galacticus_in_situ",
    "galacticus_in_plus_ex_situ",
]

DiffstarPop_Params_Diffstarfits_line_sepms_satfrac_sigslope = OrderedDict(
    smdpl=DIFFSTARFITS_SMDPL_DIFFSTARPOP_PARAMS_line_sepms_satfrac_sigslope,
    smdpl_dr1=DIFFSTARFITS_SMDPL_DR1_DIFFSTARPOP_PARAMS_line_sepms_satfrac_sigslope,
    tng=DIFFSTARFITS_TNG_DIFFSTARPOP_PARAMS_line_sepms_satfrac_sigslope,
    galacticus_in_situ=DIFFSTARFITS_GALACTICUS_IN_DIFFSTARPOP_PARAMS_line_sepms_satfrac_sigslope,
    galacticus_in_plus_ex_situ=DIFFSTARFITS_GALACTICUS_INPLUSEX_DIFFSTARPOP_PARAMS_line_sepms_satfrac_sigslope,
)
DiffstarPop_UParams_Diffstarfits_line_sepms_satfrac_sigslope = OrderedDict(
    smdpl=DIFFSTARFITS_SMDPL_DIFFSTARPOP_U_PARAMS_line_sepms_satfrac_sigslope,
    smdpl_dr1=DIFFSTARFITS_SMDPL_DR1_DIFFSTARPOP_U_PARAMS_line_sepms_satfrac_sigslope,
    tng=DIFFSTARFITS_TNG_DIFFSTARPOP_U_PARAMS_line_sepms_satfrac_sigslope,
    galacticus_in_situ=DIFFSTARFITS_GALACTICUS_IN_DIFFSTARPOP_U_PARAMS_line_sepms_satfrac_sigslope,
    galacticus_in_plus_ex_situ=DIFFSTARFITS_GALACTICUS_INPLUSEX_DIFFSTARPOP_U_PARAMS_line_sepms_satfrac_sigslope,
)
