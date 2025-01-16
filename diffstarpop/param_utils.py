"""
"""

from jax import jit as jjit

from .defaults import DEFAULT_DIFFSTARPOP_U_PARAMS


@jjit
def get_all_params_from_varied(varied, defaults):
    """Replace entries of namedtuple defaults with subset found in varied

    Params
    ------
    varied : namedtuple
        Each entry of varied._fields should also appear in defaults._fields

    defaults : namedtuple
        Default values of all parameters in the model

    Returns
    -------
    params : namedtuple
        Same as defaults except for fields appearing in varied

    """
    return defaults._replace(**varied._asdict())


@jjit
def get_all_diffstarpop_u_params(varied_u_params):
    """Get the full collection of diffstarpop unbounded params from some subset

    Parameters
    ----------
    varied_u_params : namedtuple
        varied_u_params is a namedtuple with the same 5 entries as DiffstarPopUParams
            u_sfh_pdf_cens_params
            u_satquench_params
        Each entry is itself a namedtuple of parameters, which can be any subset
        of the same parameters appearing in that component of DiffstarPopUParams

    Returns
    -------
    u_params : namedtuple
        Instance of DiffstarPopUParams. Values will be taken from varied_u_params when
        present, and otherwise will be taken from DEFAULT_DIFFSTARPOP_U_PARAMS

    """
    u_sfh_pdf_cens_params = get_all_params_from_varied(
        varied_u_params.u_sfh_pdf_cens_params,
        DEFAULT_DIFFSTARPOP_U_PARAMS.u_sfh_pdf_cens_params,
    )
    u_satquench_params = get_all_params_from_varied(
        varied_u_params.u_satquench_params,
        DEFAULT_DIFFSTARPOP_U_PARAMS.u_satquench_params,
    )

    _diffstarpop_u_params = (
        u_sfh_pdf_cens_params,
        u_satquench_params,
    )
    return DEFAULT_DIFFSTARPOP_U_PARAMS._make(_diffstarpop_u_params)
