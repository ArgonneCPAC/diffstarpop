from collections import OrderedDict

# SMDPL
from .params_diffstarpopfits_line_sepms_satfrac_smdpl import (
    DIFFSTARPOP_FITS_SMDPL_DIFFSTARPOP_PARAMS as DIFFSTARPOP_FITS_SMDPL_DIFFSTARPOP_PARAMS_line_sepms_satfrac,
    DIFFSTARPOP_FITS_SMDPL_DIFFSTARPOP_U_PARAMS as DIFFSTARPOP_FITS_SMDPL_DIFFSTARPOP_U_PARAMS_line_sepms_satfrac,
)

# SMDPL DR1
from .params_diffstarpopfits_line_sepms_satfrac_smdpl_dr1 import (
    DIFFSTARPOP_FITS_SMDPL_DR1_DIFFSTARPOP_PARAMS as DIFFSTARPOP_FITS_SMDPL_DR1_DIFFSTARPOP_PARAMS_line_sepms_satfrac,
    DIFFSTARPOP_FITS_SMDPL_DR1_DIFFSTARPOP_U_PARAMS as DIFFSTARPOP_FITS_SMDPL_DR1_DIFFSTARPOP_U_PARAMS_line_sepms_satfrac,
)

# TNG
from .params_diffstarpopfits_line_sepms_satfrac_tng import (
    DIFFSTARPOP_FITS_TNG_DIFFSTARPOP_PARAMS as DIFFSTARPOP_FITS_TNG_DIFFSTARPOP_PARAMS_line_sepms_satfrac,
    DIFFSTARPOP_FITS_TNG_DIFFSTARPOP_U_PARAMS as DIFFSTARPOP_FITS_TNG_DIFFSTARPOP_U_PARAMS_line_sepms_satfrac,
)

# Glacticus in situ
from .params_diffstarpopfits_line_sepms_satfrac_galacticus_in_situ import (
    DIFFSTARPOP_FITS_GALACTICUS_IN_DIFFSTARPOP_PARAMS as DIFFSTARPOP_FITS_GALACTICUS_IN_DIFFSTARPOP_PARAMS_line_sepms_satfrac,
    DIFFSTARPOP_FITS_GALACTICUS_IN_DIFFSTARPOP_U_PARAMS as DIFFSTARPOP_FITS_GALACTICUS_IN_DIFFSTARPOP_U_PARAMS_line_sepms_satfrac,
)


# Glacticus in plus ex situ
from .params_diffstarpopfits_line_sepms_satfrac_galacticus_in_plus_ex_situ import (
    DIFFSTARPOP_FITS_GALACTICUS_INPLUSEX_DIFFSTARPOP_PARAMS as DIFFSTARPOP_FITS_GALACTICUS_INPLUSEX_DIFFSTARPOP_PARAMS_line_sepms_satfrac,
    DIFFSTARPOP_FITS_GALACTICUS_INPLUSEX_DIFFSTARPOP_U_PARAMS as DIFFSTARPOP_FITS_GALACTICUS_INPLUSEX_DIFFSTARPOP_U_PARAMS_line_sepms_satfrac,
)


sim_name_list = [
    "smdpl",
    "smdpl_dr1",
    "tng",
    "galacticus_in_situ",
    "galacticus_in_plus_ex_situ",
]


DiffstarPop_Params_Diffstarpop_fits_line_sepms_satfrac = OrderedDict(
    smdpl=DIFFSTARPOP_FITS_SMDPL_DIFFSTARPOP_PARAMS_line_sepms_satfrac,
    smdpl_dr1=DIFFSTARPOP_FITS_SMDPL_DR1_DIFFSTARPOP_PARAMS_line_sepms_satfrac,
    tng=DIFFSTARPOP_FITS_TNG_DIFFSTARPOP_PARAMS_line_sepms_satfrac,
    galacticus_in_situ=DIFFSTARPOP_FITS_GALACTICUS_IN_DIFFSTARPOP_PARAMS_line_sepms_satfrac,
    galacticus_in_plus_ex_situ=DIFFSTARPOP_FITS_GALACTICUS_INPLUSEX_DIFFSTARPOP_PARAMS_line_sepms_satfrac,
)
DiffstarPop_UParams_Diffstarpop_fits_line_sepms_satfrac = OrderedDict(
    smdpl=DIFFSTARPOP_FITS_SMDPL_DIFFSTARPOP_U_PARAMS_line_sepms_satfrac,
    smdpl_dr1=DIFFSTARPOP_FITS_SMDPL_DR1_DIFFSTARPOP_U_PARAMS_line_sepms_satfrac,
    tng=DIFFSTARPOP_FITS_TNG_DIFFSTARPOP_U_PARAMS_line_sepms_satfrac,
    galacticus_in_situ=DIFFSTARPOP_FITS_GALACTICUS_IN_DIFFSTARPOP_U_PARAMS_line_sepms_satfrac,
    galacticus_in_plus_ex_situ=DIFFSTARPOP_FITS_GALACTICUS_INPLUSEX_DIFFSTARPOP_U_PARAMS_line_sepms_satfrac,
)
