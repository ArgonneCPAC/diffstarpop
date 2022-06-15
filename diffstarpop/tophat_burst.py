"""
"""
from jax import numpy as jnp
from jax import jit as jjit


DT_BURST = 10_000


@jjit
def _tophat_burst_sfh(t_yr, tburst_yr, sfr_burst, dt_burst=DT_BURST):
    """Star formation history of a top-hat burst

    Parameters
    ----------
    t_yr : ndarray of shape (n_times, )
        Cosmic time in years

    tburst_yr : float
        Cosmic time the birst completes in years

    sfr_burst : float
        Constant value of SFR in Msun/yr during burst

    dt_burst : float, optional
        Duration of the burst
        Default is set by BURST_DT at top of module
        Should be much shorter than the shortest lifetime of the most massive star

    Returns
    -------
    sfh : ndarray of shape (n_times, )
        SFR in Msun/yr evaluated at the input times

    """
    t0 = tburst_yr - dt_burst
    early_msk = t_yr >= t0
    late_msk = t_yr <= tburst_yr
    sfh = jnp.where(early_msk & late_msk, sfr_burst, 0.0)
    return sfh


@jjit
def _tophat_burst_smh(t_yr, tburst_yr, sfr_burst, dt_burst=DT_BURST):
    """Stellar mass history of a top-hat burst

    Parameters
    ----------
    t_yr : ndarray of shape (n_times, )
        Cosmic time in years

    tburst_yr : float
        Cosmic time the birst completes in years

    sfr_burst : float
        Constant value of SFR in Msun/yr during burst

    dt_burst : float, optional
        Duration of the burst
        Default is set by BURST_DT at top of module
        Should be much shorter than the shortest lifetime of the most massive star

    Returns
    -------
    smh : ndarray of shape (n_times, )
        Stellar mass in Msun evaluated at the input times

    """
    t0 = tburst_yr - dt_burst
    early_msk = t_yr >= t0
    late_msk = t_yr <= tburst_yr
    dt = t_yr - t0
    smh = jnp.where(early_msk & late_msk, sfr_burst * dt, 0.0)
    smh = jnp.where(~late_msk, sfr_burst * dt_burst, smh)
    return smh


@jjit
def _get_burst_sfr(delta_logsfr, logsfr_smooth):
    """Get SFR of burst, defined in terms of delta_logsfr relative to smooth SFR

    Parameters
    ----------
    delta_logsfr : float
        Base-10 logarithmic difference between burst SFR and smooth SFR in Msun/yr

    logsfr_smooth : float
        Base-10 log of smooth SFR in Msun/yr at time of burst

    Returns
    -------
    sfr_burst : float
        Constant value of SFR in Msun/yr during burst
    """
    logsfr_burst = logsfr_smooth + delta_logsfr
    sfr_burst = 10 ** logsfr_burst
    return sfr_burst
