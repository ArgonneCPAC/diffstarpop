from collections import OrderedDict

from diffstar import DEFAULT_DIFFSTAR_U_PARAMS
from jax import jit as jjit
from jax import numpy as jnp

from .mainseq_massonly import (
    MainseqMassOnlyParams,
    _get_cov_mainseq,
    _get_mean_u_params_mainseq,
)

DEFAULT_Q_U_PARAMS_UNQUENCHED = jnp.ones(4) * 5


@jjit
def main_sequence_mu_cov(ms_mass_params, mah_params):
    ms_mass_params = MainseqMassOnlyParams(*ms_mass_params)
    lgm0 = mah_params[0]
    mu_ms = _get_mean_u_params_mainseq(lgm0, ms_mass_params)
    cov_ms = _get_cov_mainseq(lgm0, ms_mass_params)
    return mu_ms, cov_ms
