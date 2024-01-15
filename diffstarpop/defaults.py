"""
"""
import typing
from collections import namedtuple

from jax import numpy as jnp

from .kernels.pdf_mainseq import DEFAULT_SFH_PDF_MAINSEQ_PDICT
from .kernels.pdf_model_assembly_bias_shifts import (
    DEFAULT_R_MAINSEQ_PDICT,
    DEFAULT_R_QUENCH_PDICT,
)
from .kernels.pdf_quenched import DEFAULT_SFH_PDF_QUENCH_PDICT

# Define a namedtuple for each component model parameter dictionary
DEFAULT_SFH_PDF_MAINSEQ_PARAMS = namedtuple(
    "Params", list(DEFAULT_SFH_PDF_MAINSEQ_PDICT.keys())
)(**DEFAULT_SFH_PDF_MAINSEQ_PDICT)

DEFAULT_SFH_PDF_QUENCH_PARAMS = namedtuple(
    "Params", list(DEFAULT_SFH_PDF_QUENCH_PDICT.keys())
)(**DEFAULT_SFH_PDF_QUENCH_PDICT)


DEFAULT_ASSEMBIAS_MAINSEQ_PARAMS = namedtuple(
    "Params", list(DEFAULT_R_MAINSEQ_PDICT.keys())
)(**DEFAULT_R_MAINSEQ_PDICT)

DEFAULT_ASSEMBIAS_QUENCH_PARAMS = namedtuple(
    "Params", list(DEFAULT_R_QUENCH_PDICT.keys())
)(**DEFAULT_R_QUENCH_PDICT)


# Define a namedtuple container for the params of each component
class DiffstarPopParams(typing.NamedTuple):
    sfh_pdf_mainseq_params: jnp.array
    sfh_pdf_quench_params: jnp.array
    assembias_mainseq_params: jnp.array
    assembias_quench_params: jnp.array


DEFAULT_DIFFSTARPOP_PARAMS = DiffstarPopParams(
    DEFAULT_SFH_PDF_MAINSEQ_PARAMS,
    DEFAULT_SFH_PDF_QUENCH_PARAMS,
    DEFAULT_ASSEMBIAS_MAINSEQ_PARAMS,
    DEFAULT_ASSEMBIAS_QUENCH_PARAMS,
)
