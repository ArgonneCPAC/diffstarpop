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

from ..params_diffstarfits_line_sepms_satfrac_smdpl import (
    DIFFSTARFITS_SMDPL_DIFFSTARPOP_PARAMS,
    DIFFSTARFITS_SMDPL_DIFFSTARPOP_U_PARAMS,
)


def test_smdpl():
    arr_params = tuple_to_array(DIFFSTARFITS_SMDPL_DIFFSTARPOP_PARAMS)
    arr_u_params = tuple_to_array(DIFFSTARFITS_SMDPL_DIFFSTARPOP_U_PARAMS)

    assert np.all(np.isfinite(arr_params))
    assert np.all(np.isfinite(arr_u_params))

    arr_u_params_bound = get_bounded_diffstarpop_params(
        DIFFSTARFITS_SMDPL_DIFFSTARPOP_U_PARAMS
    )
    arr_u_params_bound = tuple_to_array(arr_u_params_bound)
    assert np.allclose(arr_params, arr_u_params_bound, rtol=1e-4)

    arr_params_unbound = get_unbounded_diffstarpop_params(
        DIFFSTARFITS_SMDPL_DIFFSTARPOP_PARAMS
    )
    arr_params_unbound = tuple_to_array(arr_params_unbound)
    assert np.allclose(arr_u_params, arr_params_unbound, rtol=1e-4)
