from collections import OrderedDict, namedtuple
import numpy as np
import jax
from jax import numpy as jnp

from ..kernels.defaults_tpeak import (
    DEFAULT_DIFFSTARPOP_U_PARAMS,
    DEFAULT_DIFFSTARPOP_PARAMS,
)


def register_tuple_new_diffstarpop(named_tuple_class):
    jax.tree_util.register_pytree_node(
        named_tuple_class,
        # tell JAX how to unpack the NamedTuple to an iterable
        lambda x: (tuple_to_jax_array(x), None),
        # tell JAX how to pack it back into the proper NamedTuple structure
        lambda _, x: array_to_tuple_new_diffstarpop(x, named_tuple_class)
    )


def register_tuple_new_diffstarpop_subset(named_tuple_class):
    jax.tree_util.register_pytree_node(
        named_tuple_class,
        # tell JAX how to unpack the NamedTuple to an iterable
        lambda x: (tuple_to_jax_array(x), None),
        # tell JAX how to pack it back into the proper NamedTuple structure
        lambda _, x: array_to_tuple_new_diffstarpop_subset(x, named_tuple_class)
    )

def flatten_tuples(t):
    for x in t:
        if isinstance(x, tuple):
            yield from flatten_tuples(x)
        else:
            yield x


def tuple_to_jax_array(t):
    res = tuple(flatten_tuples(t))
    return jnp.asarray(res)


def tuple_to_array(t):
    res = tuple(flatten_tuples(t))
    return np.asarray(res)

def array_to_tuple_new_diffstarpop(a, t):

    count = 0

    SFH_params = DEFAULT_DIFFSTARPOP_U_PARAMS.u_sfh_pdf_cens_params
    new_count = count + len(SFH_params)
    new_sfh_pdf_cens_params_u_params = SFH_params._make(a[count:new_count])

    SAT_params = DEFAULT_DIFFSTARPOP_U_PARAMS.u_satquench_params
    count = new_count
    new_count = count + len(SAT_params)
    new_satquenchpop_u_params = SAT_params._make(a[count:new_count])

    _up = (
        new_sfh_pdf_cens_params_u_params,
        new_satquenchpop_u_params
    )
    new_diffstarpop_u_params = DEFAULT_DIFFSTARPOP_U_PARAMS._make(_up)

    new_dict = OrderedDict(
        diffstarpop_u_params=new_diffstarpop_u_params
    )

    T = t(*list(new_dict.values()))

    return T

def select_sfh_quench_params():

    # Select only mean quench params
    cens_u_params = DEFAULT_DIFFSTARPOP_U_PARAMS.u_sfh_pdf_cens_params
    cens_keys = [key for key in cens_u_params._fields if (key[0:6] == 'u_mean') and (key[-5] == 'h') or (key[0:6] == 'u_frac')]
    varied_u_cens_p = namedtuple("QseqUParams", cens_keys)(
        *[getattr(cens_u_params, key) for key in cens_keys]
    )
    return varied_u_cens_p


def get_diffstarpop_u_p_init():
    """Retrieve u_p_init to define a subset of diffstarpop unbounded parameters to use
    as an initial guess for optimization

    Returns
    -------
    u_p_init : namedtuple
        The returned u_p_init has the same 5 entries as DEFAULT_DIFFSTARPOP_U_PARAMS
        Each entry contains some subset of the parameters of that component

    """

    varied_u_cens_p = select_sfh_quench_params()

    # Select all parameters from each remaining component
    _up = (varied_u_cens_p, *DEFAULT_DIFFSTARPOP_U_PARAMS[1:])
    u_p_init = DEFAULT_DIFFSTARPOP_U_PARAMS._make(_up)

    return u_p_init

def array_to_tuple_new_diffstarpop_subset(a, t):

    count = 0

    varied_diffstarpop_u_params = get_diffstarpop_u_p_init()

    SFH_params = varied_diffstarpop_u_params.u_sfh_pdf_cens_params
    new_count = count + len(SFH_params)
    new_sfh_pdf_cens_params_u_params = SFH_params._make(a[count:new_count])

    SAT_params = varied_diffstarpop_u_params.u_satquench_params
    count = new_count
    new_count = count + len(SAT_params)
    new_satquenchpop_u_params = SAT_params._make(a[count:new_count])

    _up = (
        new_sfh_pdf_cens_params_u_params,
        new_satquenchpop_u_params
    )
    new_diffstarpop_u_params = varied_diffstarpop_u_params._make(_up)

    new_dict = OrderedDict(
        diffstarpop_u_params=new_diffstarpop_u_params
    )

    T = t(*list(new_dict.values()))

    return T