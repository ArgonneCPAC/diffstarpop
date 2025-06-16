""" """

# flake8: noqa

import typing
from collections import namedtuple, OrderedDict

# Import diffstarpop initial guess values from diffstar fits
from .params_diffstarfits_line_sepms_satfrac import (
    DiffstarPop_Params_Diffstarfits_line_sepms_satfrac,
    DiffstarPop_UParams_Diffstarfits_line_sepms_satfrac,
)
from .params_diffstarfits_line_sepms_satfrac_sigslope import (
    DiffstarPop_Params_Diffstarfits_line_sepms_satfrac_sigslope,
    DiffstarPop_UParams_Diffstarfits_line_sepms_satfrac_sigslope,
)

# Import diffstarpop best fits
from .params_diffstarpopfits_line_sepms_satfrac import (
    DiffstarPop_Params_Diffstarpop_fits_line_sepms_satfrac,
    DiffstarPop_UParams_Diffstarpop_fits_line_sepms_satfrac,
)
from .params_diffstarpopfits_line_sepms_satfrac_sigslope import (
    DiffstarPop_Params_Diffstarpop_fits_line_sepms_satfrac_sigslope,
    DiffstarPop_UParams_Diffstarpop_fits_line_sepms_satfrac_sigslope,
)
