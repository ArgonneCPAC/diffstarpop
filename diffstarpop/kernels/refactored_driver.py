"""
"""

from jax import jit as jjit
from jax import numpy as jnp

from .mainseq_massonly import (
    MainseqMassOnlyParams,
    _get_cov_mainseq,
    _get_mean_u_params_mainseq,
)
from .qseq_massonly import QseqMassOnlyParams, _get_cov_qseq, _get_mean_u_params_qseq

DEFAULT_Q_U_PARAMS_UNQUENCHED = jnp.ones(4) * 5


@jjit
def main_sequence_mu_cov(ms_mass_params, mah_params):
    ms_mass_params = MainseqMassOnlyParams(*ms_mass_params)
    lgm0 = mah_params[0]
    mu_ms = _get_mean_u_params_mainseq(ms_mass_params, lgm0)
    cov_ms = _get_cov_mainseq(ms_mass_params, lgm0)
    return mu_ms, cov_ms


@jjit
def quenched_sequence_mu_cov(qs_mass_params, mah_params):
    qs_mass_params = QseqMassOnlyParams(*qs_mass_params)
    lgm0 = mah_params[0]
    mu_qs = _get_mean_u_params_qseq(qs_mass_params, lgm0)
    cov_qs = _get_cov_qseq(qs_mass_params, lgm0)
    return mu_qs, cov_qs
