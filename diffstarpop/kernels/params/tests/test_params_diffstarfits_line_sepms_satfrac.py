import numpy as np

from ....loss_kernels.namedtuple_utils_tpeak_sepms_satfrac import (
    tuple_to_array,
    register_tuple_new_diffstarpop_tpeak,
    array_to_tuple_new_diffstarpop_tpeak,
)

from ...defaults_tpeak_line_sepms_satfrac import (
    get_unbounded_diffstarpop_params,
    get_bounded_diffstarpop_params,
)

# SMDPL
from ..params_diffstarfits_line_sepms_satfrac_smdpl import (
    DIFFSTARFITS_SMDPL_DIFFSTARPOP_PARAMS as PARAMS_SMDPL,
    DIFFSTARFITS_SMDPL_DIFFSTARPOP_U_PARAMS as U_PARAMS_SMDPL,
)

# SMDPL DR1
from ..params_diffstarfits_line_sepms_satfrac_smdpl_DR1 import (
    DIFFSTARFITS_SMDPL_DR1_DIFFSTARPOP_PARAMS as PARAMS_SMDPL_DR1,
    DIFFSTARFITS_SMDPL_DR1_DIFFSTARPOP_U_PARAMS as U_PARAMS_SMDPL_DR1,
)


def test_smdpl():
    arr_params = tuple_to_array(PARAMS_SMDPL)
    arr_u_params = tuple_to_array(U_PARAMS_SMDPL)

    assert np.all(np.isfinite(arr_params))
    assert np.all(np.isfinite(arr_u_params))

    arr_u_params_bound = get_bounded_diffstarpop_params(U_PARAMS_SMDPL)
    arr_u_params_bound = tuple_to_array(arr_u_params_bound)
    assert np.allclose(arr_params, arr_u_params_bound, rtol=1e-4, atol=1e-4)

    arr_params_unbound = get_unbounded_diffstarpop_params(PARAMS_SMDPL)
    arr_params_unbound = tuple_to_array(arr_params_unbound)
    assert np.allclose(arr_u_params, arr_params_unbound, rtol=1e-4, atol=1e-4)


def test_smdpl_dr1():
    arr_params = tuple_to_array(PARAMS_SMDPL_DR1)
    arr_u_params = tuple_to_array(U_PARAMS_SMDPL_DR1)

    assert np.all(np.isfinite(arr_params))
    assert np.all(np.isfinite(arr_u_params))

    arr_u_params_bound = get_bounded_diffstarpop_params(U_PARAMS_SMDPL_DR1)
    arr_u_params_bound = tuple_to_array(arr_u_params_bound)
    assert np.allclose(arr_params, arr_u_params_bound, rtol=1e-4, atol=1e-4)

    arr_params_unbound = get_unbounded_diffstarpop_params(PARAMS_SMDPL_DR1)
    arr_params_unbound = tuple_to_array(arr_params_unbound)
    assert np.allclose(arr_u_params, arr_params_unbound, rtol=1e-4, atol=1e-4)
