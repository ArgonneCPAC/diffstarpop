"""
"""

from collections import namedtuple
from copy import deepcopy

from ..defaults import DEFAULT_DIFFSTARPOP_U_PARAMS
from ..param_utils import get_all_diffstarpop_u_params


def test_get_all_ms_massonly_params_from_varied():
    u_sfh_pdf_cens_pdict = deepcopy(
        DEFAULT_DIFFSTARPOP_U_PARAMS.u_sfh_pdf_cens_params._asdict()
    )
    for key, val in u_sfh_pdf_cens_pdict.items():
        if key.startswith("u_mean_"):
            u_sfh_pdf_cens_pdict[key] = val + 0.1

    Params = namedtuple("Params", u_sfh_pdf_cens_pdict.keys())
    u_sfh_pdf_cens_params = Params(**u_sfh_pdf_cens_pdict)

    varied_u_params = DEFAULT_DIFFSTARPOP_U_PARAMS._replace(
        u_sfh_pdf_cens_params=u_sfh_pdf_cens_params
    )
    all_u_params = get_all_diffstarpop_u_params(varied_u_params)

    u_sfh_pdf_cens_pdict_default = deepcopy(
        DEFAULT_DIFFSTARPOP_U_PARAMS.u_sfh_pdf_cens_params._asdict()
    )
    for key, val in u_sfh_pdf_cens_pdict_default.items():
        if key.startswith("u_mean_"):
            assert getattr(all_u_params.u_sfh_pdf_cens_params, key) == val + 0.1
        else:
            assert getattr(all_u_params.u_sfh_pdf_cens_params, key) == val
