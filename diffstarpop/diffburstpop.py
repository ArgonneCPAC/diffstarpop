"""
"""
from jax import jit as jjit
from jax import vmap
from dsps.stellar_ages import _get_age_weights_from_tables, _get_lg_age_bin_edges
from .diffburst import _burst_age_weights_pop


_A = (None, None, 0)
_get_age_weights_from_tables_pop = jjit(vmap(_get_age_weights_from_tables, in_axes=_A))


@jjit
def _get_bursty_age_weights(
    lgt_ages, lgt_table, logsm_tables, dburst_pop, lgfburst_pop
):
    lgt_birth_bin_edges = _get_lg_age_bin_edges(lgt_ages)
    age_weights_smooth = _get_age_weights_from_tables_pop(
        lgt_birth_bin_edges, lgt_table, logsm_tables
    )
    age_weights_burst = _burst_age_weights_pop(lgt_ages, dburst_pop)

    fburst_pop = 10 ** lgfburst_pop.reshape((-1, 1))
    age_weights = fburst_pop * age_weights_burst + (1 - fburst_pop) * age_weights_smooth

    return age_weights, age_weights_smooth, age_weights_burst
