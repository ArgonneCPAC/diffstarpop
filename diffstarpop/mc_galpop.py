import numpy as np
from jax import vmap
from jax import jit as jjit
from jax import random as jran
from jax import numpy as jnp
from collections import OrderedDict
from collections import namedtuple

from diffstar.stars import (
    _sfr_history_from_mah,
    _integrate_sfr,
    DEFAULT_SFR_PARAMS as DEFAULT_SFR_PARAMS_DICT,
    _get_unbounded_sfr_params,
)
from diffstar.constants import LGT0
from diffstar.quenching import (
    DEFAULT_Q_PARAMS as DEFAULT_Q_PARAMS_DICT,
    _get_unbounded_q_params,
)
from diffstar.utils import _jax_get_dt_array

from diffmah.individual_halo_assembly import calc_halo_history
from diffmah.monte_carlo_halo_population import mc_halo_population

from .pdf_quenched import get_smah_means_and_covs_quench
from .pdf_mainseq import get_smah_means_and_covs_mainseq


_MCGalPop = namedtuple(
    "MCGalPop",
    ["smh", "sfh", "log_mah", "msk_is_quenched"],
)


DEFAULT_UNBOUND_SFR_PARAMS = _get_unbounded_sfr_params(
    *tuple(DEFAULT_SFR_PARAMS_DICT.values())
)
DEFAULT_UNBOUND_SFR_PARAMS_DICT = OrderedDict(
    zip(DEFAULT_SFR_PARAMS_DICT.keys(), DEFAULT_UNBOUND_SFR_PARAMS)
)
DEFAULT_UNBOUND_Q_PARAMS = np.array(
    _get_unbounded_q_params(*tuple(DEFAULT_Q_PARAMS_DICT.values()))
)
UH = DEFAULT_UNBOUND_SFR_PARAMS_DICT["indx_hi"]

DEFAULT_UNBOUND_Q_PARAMS_MAIN_SEQ = DEFAULT_UNBOUND_Q_PARAMS.copy()
DEFAULT_UNBOUND_Q_PARAMS_MAIN_SEQ[0] = 1.9


def _calculate_sm(lgt, dt, dmhdt, log_mah, sfr_params, q_params):
    sfh = _sfr_history_from_mah(lgt, dt, dmhdt, log_mah, sfr_params, q_params)
    smh = _integrate_sfr(sfh, dt)
    return smh, sfh


calculate_sm = jjit(vmap(_calculate_sm, in_axes=(*[None] * 2, *[0] * 4)))


def mc_sfh_population(
    ran_key,
    cosmic_time,
    logmh=None,
    mah_params=None,
    pdf_parameters_MS={},
    pdf_parameters_Q={},
):
    """Generate Monte Carlo realization of the assembly of a population of halos.

    Parameters
    ----------
    ran_key : jax.random.PRNGKey(seed)
        jax random number key

    cosmic_time : ndarray of shape (n_t, )
        Array of cosmic times in units of Gyr

    logmh : ndarray of shape (n_halos, ), optional
        Base-10 log of present-day halo mass of the halo population
        If None, must pass mah_params

    mah_params : ndarray of shape (n_halos, 4), optional
        Diffmah parameters of the halo population
        If None, must pass logmh, in which case DiffmahPop will generate mah_params

    **kwargs : floats
        All parameters of the SFH PDF model are accepted as keyword arguments.
        Default values are set by rockstar_pdf_model.DEFAULT_SFH_PDF_PARAMS

    Returns
    -------
    smh : ndarray of shape (n_halos, n_t)
        Cumulative history of stellar mass formed in units of Msun

    sfh : ndarray of shape (n_halos, n_t)
        Star formation history in units of Msun

    log_mah : ndarray of shape (n_halos, n_t)
        Base-10 log of halo mass history in units of Msun

    msk_is_quenched : ndarray of shape (n_halos, )
        Boolean array indicating whether the galaxy experienced a quenching event

    """
    mah_key, q_key, ms_key, frac_q_key = jran.split(ran_key, 4)

    diffmah_args_msg = "Must input either mah_params or logmh"
    if mah_params is None:
        assert logmh is not None, diffmah_args_msg
        halopop = mc_halo_population(cosmic_time, 10**LGT0, logmh, ran_key=mah_key)
        dmhdt = halopop.dmhdt
        log_mah = halopop.log_mah
        msg = "Inconsistency in call to mc_halo_population function"
        assert np.allclose(log_mah[:, -1], logmh), msg
    elif mah_params is not None:
        assert logmh is None, diffmah_args_msg
        shape_msg = "mah_params.shape={0} must be (n_halos, 4)"
        assert mah_params.shape[1] == 4, shape_msg.format(mah_params.shape)
        mah_logmp = mah_params[:, 0]
        mah_lgtc = 10 ** mah_params[:, 1]
        mah_early = mah_params[:, 2]
        mah_late = mah_params[:, 3]
        dmhdt, log_mah = calc_halo_history(
            cosmic_time, 10**LGT0, mah_logmp, mah_lgtc, mah_early, mah_late
        )
        logmh = log_mah[:, -1]

    n_halos = log_mah.shape[0]

    _res = get_smah_means_and_covs_quench(logmh, **pdf_parameters_Q)
    frac_quench, means_quench, covs_quench = _res
    frac_quench = np.array(frac_quench)
    means_quench = np.array(means_quench)
    covs_quench = np.array(covs_quench)

    uran = jran.uniform(frac_q_key, shape=(n_halos,))
    msk_is_quenched = np.array(uran < frac_quench)

    n_halos_Q = msk_is_quenched.sum()
    n_halos_MS = n_halos - n_halos_Q

    sfr_params_quench = np.zeros((n_halos_Q, 5))
    sfr_params_mainseq = np.zeros((n_halos_MS, 5))
    q_params_mainseq = np.zeros((n_halos_MS, 4))

    sfr_params = np.zeros((n_halos, 5))
    q_params = np.zeros((n_halos, 4))

    if n_halos_Q > 0:
        mu = means_quench[msk_is_quenched]
        cov = covs_quench[msk_is_quenched]
        sfh_params_quench = jran.multivariate_normal(
            q_key, mean=mu, cov=cov, shape=(n_halos_Q,)
        )
        sfh_params_quench = np.array(sfh_params_quench)

        sfr_params_quench[:, :3] = sfh_params_quench[:, :3]
        sfr_params_quench[:, 3] = UH
        sfr_params_quench[:, 4] = sfh_params_quench[:, 3]

        sfr_params[msk_is_quenched] = sfr_params_quench
        q_params[msk_is_quenched] = sfh_params_quench[:, 4:8]

    if n_halos_MS > 0:
        _res = get_smah_means_and_covs_mainseq(
            logmh[~msk_is_quenched], **pdf_parameters_MS
        )
        means_mainseq, covs_mainseq = _res

        sfh_params_mainseq = jran.multivariate_normal(
            ms_key, mean=means_mainseq, cov=covs_mainseq, shape=(n_halos_MS,)
        )
        sfh_params_mainseq = np.array(sfh_params_mainseq)

        sfr_params_mainseq[:, :3] = sfh_params_mainseq[:, :3]
        sfr_params_mainseq[:, 3] = UH
        sfr_params_mainseq[:, 4] = sfh_params_mainseq[:, 3]
        q_params_mainseq[:, np.arange(4)] = DEFAULT_UNBOUND_Q_PARAMS_MAIN_SEQ

        sfr_params[~msk_is_quenched] = sfr_params_mainseq
        q_params[~msk_is_quenched] = q_params_mainseq

    lgt = jnp.log10(cosmic_time)
    dtarr = _jax_get_dt_array(cosmic_time)
    smh, sfh = calculate_sm(lgt, dtarr, dmhdt, log_mah, sfr_params, q_params)

    return _MCGalPop(*(smh, sfh, log_mah, msk_is_quenched))
