"""
"""
import typing
from collections import namedtuple

from jax import jit as jjit
from jax import numpy as jnp

from .assembias_kernels import DEFAULT_AB_MAINSEQ_PARAMS, DEFAULT_AB_QSEQ_PARAMS
from .mainseq_massonly import DEFAULT_SFH_PDF_MAINSEQ_PARAMS
from .qseq_massonly import DEFAULT_SFH_PDF_QUENCH_PARAMS
from .satquenchpop_model import DEFAULT_SATQUENCHPOP_PARAMS


# Define a namedtuple container for the params of each component
class DiffstarPopParams(typing.NamedTuple):
    sfh_pdf_mainseq_params: jnp.array
    sfh_pdf_quench_params: jnp.array
    assembias_mainseq_params: jnp.array
    assembias_quench_params: jnp.array
    satquench_params: jnp.array


DEFAULT_DIFFSTARPOP_PARAMS = DiffstarPopParams(
    DEFAULT_SFH_PDF_MAINSEQ_PARAMS,
    DEFAULT_SFH_PDF_QUENCH_PARAMS,
    DEFAULT_AB_MAINSEQ_PARAMS,
    DEFAULT_AB_QSEQ_PARAMS,
    DEFAULT_SATQUENCHPOP_PARAMS,
)

_U_PNAMES = ["u_" + key for key in DEFAULT_DIFFSTARPOP_PARAMS._fields]
DiffstarPopUParams = namedtuple("DiffstarPopUParams", _U_PNAMES)

from .assembias_kernels import (
    get_bounded_ab_mainseq_params,
    get_bounded_ab_qseq_params,
    get_unbounded_ab_mainseq_params,
    get_unbounded_ab_qseq_params,
)
from .mainseq_massonly import (
    get_bounded_mainseq_massonly_params,
    get_unbounded_mainseq_massonly_params,
)
from .qseq_massonly import (
    get_bounded_qseq_massonly_params,
    get_unbounded_qseq_massonly_params,
)
from .satquenchpop_model import (
    get_bounded_satquenchpop_params,
    get_unbounded_satquenchpop_params,
)


@jjit
def get_bounded_diffstarpop_params(diffstarpop_u_params):
    sfh_pdf_mainseq_params = get_bounded_mainseq_massonly_params(
        diffstarpop_u_params.u_sfh_pdf_mainseq_params
    )
    sfh_pdf_quench_params = get_bounded_qseq_massonly_params(
        diffstarpop_u_params.u_sfh_pdf_quench_params
    )
    assembias_mainseq_params = get_bounded_ab_mainseq_params(
        diffstarpop_u_params.u_assembias_mainseq_params
    )
    assembias_quench_params = get_bounded_ab_qseq_params(
        diffstarpop_u_params.u_assembias_quench_params
    )

    satquench_params = get_bounded_satquenchpop_params(
        diffstarpop_u_params.u_satquench_params
    )
    return DiffstarPopParams(
        sfh_pdf_mainseq_params,
        sfh_pdf_quench_params,
        assembias_mainseq_params,
        assembias_quench_params,
        satquench_params,
    )


@jjit
def get_unbounded_diffstarpop_params(diffstarpop_params):
    u_sfh_pdf_mainseq_params = get_unbounded_mainseq_massonly_params(
        diffstarpop_params.sfh_pdf_mainseq_params
    )
    u_sfh_pdf_quench_params = get_unbounded_qseq_massonly_params(
        diffstarpop_params.sfh_pdf_quench_params
    )
    u_assembias_mainseq_params = get_unbounded_ab_mainseq_params(
        diffstarpop_params.assembias_mainseq_params
    )
    u_assembias_quench_params = get_unbounded_ab_qseq_params(
        diffstarpop_params.assembias_quench_params
    )

    u_satquench_params = get_unbounded_satquenchpop_params(
        diffstarpop_params.satquench_params
    )
    return DiffstarPopUParams(
        u_sfh_pdf_mainseq_params,
        u_sfh_pdf_quench_params,
        u_assembias_mainseq_params,
        u_assembias_quench_params,
        u_satquench_params,
    )


DEFAULT_DIFFSTARPOP_U_PARAMS = get_unbounded_diffstarpop_params(
    DEFAULT_DIFFSTARPOP_PARAMS
)
