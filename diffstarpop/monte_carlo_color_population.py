import numpy as np
from jax import vmap
from jax import jit as jjit
from jax import numpy as jnp
from jax import random as jran
from functools import partial

from diffsky.experimental.dspspop.photpop import get_obs_photometry_singlez
from diffsky.diffndhist import tw_ndhist_weighted

from dsps.cosmology import DEFAULT_COSMOLOGY
from dsps.data_loaders import load_ssp_templates
from dsps.data_loaders.load_filter_data import load_transmission_curve
from dsps.photometry.utils import interpolate_filter_trans_curves
from dsps.photometry.photpop import precompute_ssp_obsmags_on_z_table

from .monte_carlo_diff_halo_population import (
    _tw_cuml_lax_kern_vmap,
    compute_sumstats_MIX_p50,
    draw_sfh_MIX,
)
from .pdf_quenched import DEFAULT_SFH_PDF_QUENCH_PARAMS
from .pdf_mainseq import DEFAULT_SFH_PDF_MAINSEQ_PARAMS
from .pdf_model_assembly_bias_shifts import (
    DEFAULT_R_MAINSEQ_PARAMS,
    DEFAULT_R_QUENCH_PARAMS,
)


DSPS_data_path = "/Users/alarcon/Documents/DSPS_data/"
DSPS_filter_path = "/Users/alarcon/Documents/DSPS_data/filters/"


def precompute_DSPS_data(z_arr, filter_list):
    cosmo_params = DEFAULT_COSMOLOGY

    # Load ssp data
    ssp_data = load_ssp_templates(drn=DSPS_data_path)
    ssp_lgmet = ssp_data[0]
    ssp_lg_age_gyr = ssp_data[1]
    ssp_waves = ssp_data[2]
    ssp_spectra = ssp_data[3]

    filter_data = [
        load_transmission_curve(drn=DSPS_filter_path, bn_pat=filt)
        for filt in filter_list
    ]
    wave_filters = [x[0] for x in filter_data]
    trans_filters = [x[1] for x in filter_data]
    filter_waves, filter_trans = interpolate_filter_trans_curves(
        wave_filters, trans_filters
    )

    lgfburst_u_params = np.array(
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
    burstshape_u_params = np.array(
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
    lgav_dust_u_params = np.array(
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
    delta_dust_u_params = np.array(
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
    boris_dust_u_params = np.array(
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

    ssp_obs_photflux_table_arr = []
    for z_obs in z_arr:
        # Precompute ssp mags for zobs
        args = [
            ssp_waves,
            ssp_spectra,
            filter_waves,
            filter_trans,
            np.array((z_obs,)),
            *cosmo_params,
        ]
        ssp_obs_photmags = precompute_ssp_obsmags_on_z_table(*args)[0]
        ssp_obs_photflux_table = 10 ** (-0.4 * ssp_obs_photmags)
        ssp_obs_photflux_table_arr.append(ssp_obs_photflux_table)

    ssp_obs_photflux_table_arr = np.array(ssp_obs_photflux_table_arr)

    dsps_data = (
        filter_waves,
        filter_trans,
        ssp_lgmet,
        ssp_lg_age_gyr,
        lgfburst_u_params,
        burstshape_u_params,
        lgav_dust_u_params,
        delta_dust_u_params,
        boris_dust_u_params,
        cosmo_params,
    )

    return ssp_obs_photflux_table_arr, dsps_data


@jjit
def get_colors_kern(
    gal_t_table, gal_sfr_table, z_obs, ssp_obs_photflux_table, dsps_data, ran_key
):
    (
        filter_waves,
        filter_trans,
        ssp_lgmet,
        ssp_lg_age_gyr,
        lgfburst_u_params,
        burstshape_u_params,
        lgav_dust_u_params,
        delta_dust_u_params,
        boris_dust_u_params,
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


_A = (None, None, 0, 0, None, None)
get_colors = jjit(vmap(get_colors_kern, in_axes=_A))


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


@partial(jjit, static_argnames=["n_histories"])
def sumstats_sfh_colmags(
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
    dsps_data,
    ssp_obs_photflux_table_arr,
    pdf_parameters_Q=DEFAULT_SFH_PDF_QUENCH_PARAMS,
    pdf_parameters_MS=DEFAULT_SFH_PDF_MAINSEQ_PARAMS,
    R_model_params_Q=DEFAULT_R_QUENCH_PARAMS,
    R_model_params_MS=DEFAULT_R_MAINSEQ_PARAMS,
):

    sfh_key, ssp_key = jran.split(ran_key, 2)

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

    ssfr = sfr / mstar
    # weights_quench_bin = jnp.where(ssfr > 1e-11, 1.0, 0.0)
    nhist, nt = ssfr.shape
    ssfr = jnp.where(ssfr > 0.0, jnp.log10(ssfr), -50.0)
    weights_quench_bin = _tw_cuml_lax_kern_vmap(ssfr.reshape(nhist * nt), -11.0, 0.2)
    weights_quench_bin = weights_quench_bin.reshape(nhist, nt)

    mstar = jnp.where(mstar > 0.0, jnp.log10(mstar), 0.0)
    log_sfr = jnp.where(sfr > 0.0, jnp.log10(sfr), 0.0)

    _res = compute_sumstats_MIX_p50(
        mstar, log_sfr, p50_sampled, weights_quench_bin, weight
    )
    (
        mean_sm,
        variance_sm,
        mean_sfr_MS,
        mean_sfr_Q,
        variance_sfr_MS,
        variance_sfr_Q,
        quench_frac,
        mean_sm_early,
        mean_sm_late,
        variance_sm_early,
        variance_sm_late,
        quench_frac_early,
        quench_frac_late,
    ) = _res

    gal_mags = get_colors(
        t_table, sfr, z_arr, ssp_obs_photflux_table_arr, dsps_data, ssp_key
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

    out = (
        mean_sm,
        variance_sm,
        mean_sfr_MS,
        mean_sfr_Q,
        variance_sfr_MS,
        variance_sfr_Q,
        quench_frac,
        mean_sm_early,
        mean_sm_late,
        variance_sm_early,
        variance_sm_late,
        quench_frac_early,
        quench_frac_late,
        counts_i_ug,
        counts_i_gr,
        counts_i_ri,
        counts_i_iz,
        counts_ug_gr,
        counts_gr_ri,
        counts_ri_iz,
    )
    return out


_A = (None, 0, 0, 0, *[None] * 18)
sumstats_sfh_colmags_vmap = jjit(
    vmap(sumstats_sfh_colmags, in_axes=_A), static_argnames=["n_histories"]
)
