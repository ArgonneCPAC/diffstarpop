"""
"""

import typing
from collections import namedtuple

from jax import jit as jjit
from jax import numpy as jnp

from .mainseq_massonly import (
    DEFAULT_SFH_PDF_MAINSEQ_PARAMS,
    get_bounded_mainseq_massonly_params,
    get_unbounded_mainseq_massonly_params,
)
from .sfh_pdf_block_cov import (
    SFH_PDF_QUENCH_PARAMS,
    get_bounded_qseq_params,
    get_unbounded_qseq_params,
)


# Define a namedtuple container for the params of each component
class DiffstarPopParams(typing.NamedTuple):
    sfh_pdf_cens_params: jnp.array
    sfh_pdf_quench_params: jnp.array


DEFAULT_DIFFSTARPOP_PARAMS = DiffstarPopParams(
    DEFAULT_SFH_PDF_MAINSEQ_PARAMS,
    SFH_PDF_QUENCH_PARAMS,
)

_U_PNAMES = ["u_" + key for key in DEFAULT_DIFFSTARPOP_PARAMS._fields]
DiffstarPopUParams = namedtuple("DiffstarPopUParams", _U_PNAMES)


@jjit
def get_bounded_diffstarpop_params(diffstarpop_u_params):
    sfh_pdf_mainseq_params = get_bounded_mainseq_massonly_params(
        diffstarpop_u_params.u_sfh_pdf_cens_params
    )
    sfh_pdf_quench_params = get_bounded_qseq_params(
        diffstarpop_u_params.u_sfh_pdf_quench_params
    )
    return DiffstarPopParams(sfh_pdf_mainseq_params, sfh_pdf_quench_params)


@jjit
def get_unbounded_diffstarpop_params(diffstarpop_params):
    u_sfh_pdf_cens_params = get_unbounded_mainseq_massonly_params(
        diffstarpop_params.sfh_pdf_cens_params
    )
    u_sfh_pdf_quench_params = get_unbounded_qseq_params(
        diffstarpop_params.sfh_pdf_quench_params
    )
    return DiffstarPopUParams(u_sfh_pdf_cens_params, u_sfh_pdf_quench_params)


DEFAULT_DIFFSTARPOP_U_PARAMS = get_unbounded_diffstarpop_params(
    DEFAULT_DIFFSTARPOP_PARAMS
)
