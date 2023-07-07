import numpy as np
from jax import vmap
from jax import jit as jjit
from jax import numpy as jnp
from jax import random as jran
from functools import partial

from diffsky.experimental.dspspop.photpop import get_obs_photometry_singlez
from diffsky.diffndhist import tw_ndhist_weighted, _tw_ndhist_weighted_kern

from .monte_carlo_diff_halo_population import draw_sfh_MIX
from .pdf_quenched import (
    DEFAULT_SFH_PDF_QUENCH_PARAMS,
)
from .pdf_mainseq import DEFAULT_SFH_PDF_MAINSEQ_PARAMS
from .pdf_model_assembly_bias_shifts import (
    DEFAULT_R_QUENCH_PARAMS,
    DEFAULT_R_MAINSEQ_PARAMS,
)

DEFAULT_lgfburst_u_params = np.array(
    [
        13.676748,
        24.993097,
        16.346615,
        -8.077414,
        -7.230221,
        7.977014,
        -10.025747,
        -31.93147,
        1.8577358,
    ]
)
DEFAULT_burstshape_u_params = np.array(
    [
        10.811945,
        -14.260326,
        -12.719431,
        13.915222,
        -22.360744,
        -11.798748,
        14.894418,
        11.894378,
        -6.2989817,
        14.317399,
        -25.193857,
        10.13922,
    ]
)
DEFAULT_lgav_dust_u_params = np.array(
    [
        13.403622,
        24.446869,
        18.417566,
        1.7414757,
        28.589493,
        18.164486,
        -34.442814,
        -37.12286,
        -11.0768175,
    ]
)
DEFAULT_delta_dust_u_params = np.array(
    [
        3.9146907,
        25.844297,
        5.9755583,
        -13.801724,
        -7.8439326,
        17.40668,
        -29.22427,
        -1.6012566,
        -12.697886,
    ]
)
DEFAULT_boris_dust_u_params = np.array(
    [
        7.4935946,
        -8.661586,
        7.034099,
        -1.7708112,
        0.87082285,
        -19.434017,
        14.279153,
        -8.100566,
        2.7693071,
        15.053559,
        32.051685,
        -40.424793,
        -16.915089,
        -21.71999,
        -38.98167,
        -13.244473,
        -2.2783546,
        29.663462,
        45.44704,
        -30.58583,
        -15.197127,
    ]
)

@jjit
def calc_hist_kern(arrA, arrB, ndsig, weight, ndbins_lo, ndbins_hi):
    eps = 1e-10

    nddata = jnp.array([arrA, arrB]).T
    counts = tw_ndhist_weighted(nddata, ndsig, weight, ndbins_lo, ndbins_hi)
    counts = jnp.clip(counts, eps, None)
    counts = counts / jnp.sum(counts)
    return counts


calc_hist = jjit(vmap(calc_hist_kern, in_axes=(0, 0, None, *[None] * 3)))


@jjit
def calculate_counts_mag_colors_bin(
    gal_mags,
    weight,
    ndsig_colmag,
    ndsig_colcol,
    bins_LO_colcol,
    bins_HI_colcol,
    bins_LO_colmag,
    bins_HI_colmag,
):

    imag = gal_mags[:, :, 3]  # i-band
    gal_col = -jnp.diff(gal_mags, axis=2)  # u-g, g-r, r-i, i-z

    # nzarr, nhist = imag.shape
    # weights_magcut = _tw_cuml_lax_kern_vmap(imag, 25.0, 0.2)
    # weights_magcut = weights_magcut.reshape(nzarr, nhist)
    # weights_magcut = 1.0 - weights_magcut # so it's 1 for brigther (smaller) magnitudes.

    counts_i_ug = calc_hist(
        imag, gal_col[:, :, 0], ndsig_colmag, weight, bins_LO_colmag, bins_HI_colmag
    )
    counts_i_gr = calc_hist(
        imag, gal_col[:, :, 1], ndsig_colmag, weight, bins_LO_colmag, bins_HI_colmag
    )
    counts_i_ri = calc_hist(
        imag, gal_col[:, :, 2], ndsig_colmag, weight, bins_LO_colmag, bins_HI_colmag
    )
    counts_i_iz = calc_hist(
        imag, gal_col[:, :, 3], ndsig_colmag, weight, bins_LO_colmag, bins_HI_colmag
    )

    counts_ug_gr = calc_hist(
        gal_col[:, :, 0],
        gal_col[:, :, 1],
        ndsig_colcol,
        weight,
        bins_LO_colcol,
        bins_HI_colcol,
    )
    counts_gr_ri = calc_hist(
        gal_col[:, :, 1],
        gal_col[:, :, 2],
        ndsig_colcol,
        weight,
        bins_LO_colcol,
        bins_HI_colcol,
    )
    counts_ri_iz = calc_hist(
        gal_col[:, :, 2],
        gal_col[:, :, 3],
        ndsig_colcol,
        weight,
        bins_LO_colcol,
        bins_HI_colcol,
    )

    out = (
        counts_i_ug,
        counts_i_gr,
        counts_i_ri,
        counts_i_iz,
        counts_ug_gr,
        counts_gr_ri,
        counts_ri_iz,
    )

    return out


@jjit
def get_colors_kern(
    gal_t_table, 
    gal_sfr_table, 
    z_obs, 
    ssp_obs_photflux_table, 
    ran_key,
    dsps_data, 
    lgfburst_u_params,
    burstshape_u_params,
    lgav_dust_u_params,
    delta_dust_u_params,
    boris_dust_u_params,
):
    (
        filter_waves,
        filter_trans,
        ssp_lgmet,
        ssp_lg_age_gyr,
        cosmo_params,
    ) = dsps_data

    res = get_obs_photometry_singlez(
        ran_key,
        filter_waves,
        filter_trans,
        ssp_obs_photflux_table,
        ssp_lgmet,
        ssp_lg_age_gyr,
        gal_t_table,
        gal_sfr_table,
        lgfburst_u_params,
        burstshape_u_params,
        lgav_dust_u_params,
        delta_dust_u_params,
        boris_dust_u_params,
        cosmo_params,
        z_obs,
    )

    (
        weights,
        lgmet_weights,
        smooth_age_weights,
        bursty_age_weights,
        frac_trans,
        gal_obsflux_nodust,
        gal_obsflux,
    ) = res

    # Magnitudes
    galpop_obs_mags = -2.5 * jnp.log10(gal_obsflux)

    return galpop_obs_mags


_A = (None, None, 0, 0, 0, *[None]*6)
get_colors = jjit(vmap(get_colors_kern, in_axes=_A))


@partial(jjit, static_argnames=["n_histories"])
def sumstats_lightcone_colors_single_m0(
    t_table,
    logmh,
    mah_params,
    p50,
    n_histories,
    ran_key,
    index_select,
    index_high,
    fstar_tdelay,
    ndsig_colcol,
    ndsig_colmag,
    bins_LO_colcol,
    bins_HI_colcol,
    bins_LO_colmag,
    bins_HI_colmag,
    z_arr, 
    dVdz,
    dsps_data,
    ssp_obs_photflux_table_arr,
    pdf_parameters_Q=DEFAULT_SFH_PDF_QUENCH_PARAMS,
    pdf_parameters_MS=DEFAULT_SFH_PDF_MAINSEQ_PARAMS,
    R_model_params_Q=DEFAULT_R_QUENCH_PARAMS,
    R_model_params_MS=DEFAULT_R_MAINSEQ_PARAMS,
    lgfburst_u_params=DEFAULT_lgfburst_u_params,
    burstshape_u_params=DEFAULT_burstshape_u_params,
    lgav_dust_u_params=DEFAULT_lgav_dust_u_params,
    delta_dust_u_params=DEFAULT_delta_dust_u_params,
    boris_dust_u_params=DEFAULT_boris_dust_u_params,
):

    sfh_key, subkey = jran.split(ran_key, 2)
    ssp_keys = jran.split(subkey, len(z_arr))
    
    
    mstar, sfr, fstar, p50_sampled, weight = draw_sfh_MIX(
        t_table,
        logmh,
        mah_params,
        p50,
        n_histories,
        sfh_key,
        index_select,
        index_high,
        fstar_tdelay,
        pdf_parameters_Q,
        pdf_parameters_MS,
        R_model_params_Q,
        R_model_params_MS,
    )
        
    gal_mags = get_colors(
        t_table, 
        sfr, 
        z_arr, 
        ssp_obs_photflux_table_arr, 
        ssp_keys,
        dsps_data, 
        lgfburst_u_params,
        burstshape_u_params,
        lgav_dust_u_params,
        delta_dust_u_params,
        boris_dust_u_params,
    )

    _res = calculate_counts_mag_colors_bin(
        gal_mags,
        weight,
        ndsig_colmag,
        ndsig_colcol,
        bins_LO_colcol,
        bins_HI_colcol,
        bins_LO_colmag,
        bins_HI_colmag,
    )

    (
        counts_i_ug,
        counts_i_gr,
        counts_i_ri,
        counts_i_iz,
        counts_ug_gr,
        counts_gr_ri,
        counts_ri_iz,
    ) = _res

    counts_i_ug = jnp.einsum("zb,z->b", counts_i_ug, dVdz)
    counts_i_gr = jnp.einsum("zb,z->b", counts_i_gr, dVdz)
    counts_i_ri = jnp.einsum("zb,z->b", counts_i_ri, dVdz)
    counts_i_iz = jnp.einsum("zb,z->b", counts_i_iz, dVdz)
    counts_ug_gr = jnp.einsum("zb,z->b", counts_ug_gr, dVdz)
    counts_gr_ri = jnp.einsum("zb,z->b", counts_gr_ri, dVdz)
    counts_ri_iz = jnp.einsum("zb,z->b", counts_ri_iz, dVdz)

    counts = (
        counts_i_ug,
        counts_i_gr,
        counts_i_ri,
        counts_i_iz,
        counts_ug_gr,
        counts_gr_ri,
        counts_ri_iz,
    )
    return counts


_A = (None, 0, 0, 0, None, 0, *[None]*22)
sumstats_lightcone_colors_single_m0_vmap = jjit(vmap(sumstats_lightcone_colors_single_m0, in_axes=_A), static_argnames=["n_histories"])


@partial(jjit, static_argnames=["n_histories"])
def sumstats_lightcone_colors(
    t_table,
    logmh_arr,
    mah_params_arr,
    p50_arr,
    pm0,
    n_histories,
    ran_key,
    index_select,
    index_high,
    fstar_tdelay,
    ndsig_colcol,
    ndsig_colmag,
    bins_LO_colcol,
    bins_HI_colcol,
    bins_LO_colmag,
    bins_HI_colmag,
    z_arr, 
    dVdz,
    dsps_data,
    ssp_obs_photflux_table_arr,
    pdf_parameters_Q=DEFAULT_SFH_PDF_QUENCH_PARAMS,
    pdf_parameters_MS=DEFAULT_SFH_PDF_MAINSEQ_PARAMS,
    R_model_params_Q=DEFAULT_R_QUENCH_PARAMS,
    R_model_params_MS=DEFAULT_R_MAINSEQ_PARAMS,
    lgfburst_u_params=DEFAULT_lgfburst_u_params,
    burstshape_u_params=DEFAULT_burstshape_u_params,
    lgav_dust_u_params=DEFAULT_lgav_dust_u_params,
    delta_dust_u_params=DEFAULT_delta_dust_u_params,
    boris_dust_u_params=DEFAULT_boris_dust_u_params,
):

    sfh_keys = jran.split(ran_key, len(pm0))
    
    _res = sumstats_lightcone_colors_single_m0_vmap(
        t_table,
        logmh_arr,
        mah_params_arr,
        p50_arr,
        n_histories,
        sfh_keys,
        index_select,
        index_high,
        fstar_tdelay,
        ndsig_colcol,
        ndsig_colmag,
        bins_LO_colcol,
        bins_HI_colcol,
        bins_LO_colmag,
        bins_HI_colmag,
        z_arr, 
        dVdz,
        dsps_data,
        ssp_obs_photflux_table_arr,
        pdf_parameters_Q,
        pdf_parameters_MS,
        R_model_params_Q,
        R_model_params_MS,
        lgfburst_u_params,
        burstshape_u_params,
        lgav_dust_u_params,
        delta_dust_u_params,
        boris_dust_u_params,
    )
    (
        counts_i_ug,
        counts_i_gr,
        counts_i_ri,
        counts_i_iz,
        counts_ug_gr,
        counts_gr_ri,
        counts_ri_iz,
    ) = _res

    counts_i_ug = jnp.einsum("mb,m->b", counts_i_ug, pm0)
    counts_i_gr = jnp.einsum("mb,m->b", counts_i_gr, pm0)
    counts_i_ri = jnp.einsum("mb,m->b", counts_i_ri, pm0)
    counts_i_iz = jnp.einsum("mb,m->b", counts_i_iz, pm0)
    counts_ug_gr = jnp.einsum("mb,m->b", counts_ug_gr, pm0)
    counts_gr_ri = jnp.einsum("mb,m->b", counts_gr_ri, pm0)
    counts_ri_iz = jnp.einsum("mb,m->b", counts_ri_iz, pm0)

    counts = (
        counts_i_ug,
        counts_i_gr,
        counts_i_ri,
        counts_i_iz,
        counts_ug_gr,
        counts_gr_ri,
        counts_ri_iz,
    )
    return counts
