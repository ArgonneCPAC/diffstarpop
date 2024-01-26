"""
"""
from collections import namedtuple
from copy import deepcopy

from ..defaults import DEFAULT_DIFFSTARPOP_U_PARAMS
from ..kernels.mainseq_massonly import DEFAULT_SFH_PDF_MAINSEQ_U_PARAMS
from ..param_utils import get_all_diffstarpop_u_params


def test_get_all_ms_massonly_params_from_varied():
    u_sfh_pdf_mainseq_pdict = deepcopy(DEFAULT_SFH_PDF_MAINSEQ_U_PARAMS._asdict())
    for key, val in u_sfh_pdf_mainseq_pdict.items():
        if key[:5] == "mean_":
            u_sfh_pdf_mainseq_pdict[key] = val + 0.1

    Params = namedtuple("Params", u_sfh_pdf_mainseq_pdict.keys())
    u_sfh_pdf_mainseq_params = Params(**u_sfh_pdf_mainseq_pdict)

    varied_u_params = DEFAULT_DIFFSTARPOP_U_PARAMS._replace(
        u_sfh_pdf_mainseq_params=u_sfh_pdf_mainseq_params
    )
    all_u_params = get_all_diffstarpop_u_params(varied_u_params)

    u_sfh_pdf_mainseq_pdict_default = deepcopy(
        DEFAULT_DIFFSTARPOP_U_PARAMS.u_sfh_pdf_mainseq_params._asdict()
    )
    for key, val in u_sfh_pdf_mainseq_pdict_default.items():
        if key[:5] == "mean_":
            assert getattr(all_u_params.u_sfh_pdf_mainseq_params, key) == val + 0.1
        else:
            assert getattr(all_u_params.u_sfh_pdf_mainseq_params, key) == val
