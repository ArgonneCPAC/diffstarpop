from collections import OrderedDict

from diffstar import DEFAULT_DIFFSTAR_U_PARAMS
from jax import jit as jjit
from jax import numpy as jnp

from .pdf_mainseq import DEFAULT_SFH_PDF_MAINSEQ_PDICT, _get_chol_params_mainseq
from .pdf_mainseq import _get_cov_scalar as _get_cov_scalar_ms
from .pdf_mainseq import _get_mean_smah_params_mainseq
from .pdf_model_assembly_bias_shifts import (
    DEFAULT_R_MAINSEQ_PDICT,
    DEFAULT_R_QUENCH_PDICT,
    _get_slopes_mainseq,
    _get_slopes_quench,
)
from .pdf_quenched import DEFAULT_SFH_PDF_QUENCH_PDICT, _get_chol_params_quench
from .pdf_quenched import _get_cov_scalar as _get_cov_scalar_q
from .pdf_quenched import _get_mean_smah_params_quench, frac_quench_vs_lgm0

DEFAULT_Q_U_PARAMS_UNQUENCHED = jnp.ones(4) * 5


@jjit
def main_sequence_mu_cov(mah_params, pdf_pdict_MS=DEFAULT_SFH_PDF_MAINSEQ_PDICT):
    lgm0 = mah_params[0]

    # main sequence
    means_mainseq_pdict = OrderedDict(
        [(key, val) for (key, val) in pdf_pdict_MS.items() if "mean_" in key]
    )
    cov_mainseq_pdict = OrderedDict(
        [(key, val) for (key, val) in pdf_pdict_MS.items() if "chol_" in key]
    )
    mu_ms = _get_mean_smah_params_mainseq(lgm0, **means_mainseq_pdict)
    chol_params_ms = _get_chol_params_mainseq(lgm0, **cov_mainseq_pdict)
    cov_ms = _get_cov_scalar_ms(*chol_params_ms)
