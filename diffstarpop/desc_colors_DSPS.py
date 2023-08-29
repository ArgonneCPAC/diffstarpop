import h5py
import os
import numpy as np
from jax import vmap, grad
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
from .utils import _tw_cuml_lax_kern

from .lightcone_colors import (
    DEFAULT_lgfburst_u_params,
    DEFAULT_burstshape_u_params,
    DEFAULT_lgav_dust_u_params,
    DEFAULT_delta_dust_u_params,
    DEFAULT_boris_dust_u_params,
)

from diffsky.experimental.photometry_interpolation import interpolate_ssp_photmag_table
from dsps.photometry.photpop import precompute_ssp_obsmags_on_z_table
from dsps.cosmology import DEFAULT_COSMOLOGY
from dsps.data_loaders import load_ssp_templates
from dsps.data_loaders.load_filter_data import load_transmission_curve
from dsps.photometry.utils import interpolate_filter_trans_curves

_tw_cuml_lax_kern_vmap = jjit(vmap(_tw_cuml_lax_kern, in_axes=(0, None, None)))


N_BURST_F = len(DEFAULT_lgfburst_u_params)
N_BURST_SHAPE = len(DEFAULT_burstshape_u_params)
N_DUST_LGAV = len(DEFAULT_lgav_dust_u_params)
N_DUST_DELTA = len(DEFAULT_delta_dust_u_params)
N_DUST_BORIS = len(DEFAULT_boris_dust_u_params)


@jjit
def mse(pred, targ):
    return jnp.mean((pred - targ) ** 2)


@jjit
def get_colors_single_redshift(
    z_obs,
    ssp_waves,
    ssp_spectra,
    filter_waves,
    filter_trans,
    cosmo_params,
):
    args = [
        ssp_waves,
        ssp_spectra,
        filter_waves,
        filter_trans,
        jnp.atleast_1d(z_obs),
        *cosmo_params,
    ]
    ssp_obs_photmags = precompute_ssp_obsmags_on_z_table(*args)[0]
    ssp_obs_photflux_table = 10 ** (-0.4 * ssp_obs_photmags)
    return ssp_obs_photflux_table


_get_colors_single_redshift_vmap_kern = jjit(
    vmap(get_colors_single_redshift, in_axes=(0, *[None] * 5))
)


@jjit
def get_colors_array(
    z_arr,
    ssp_waves,
    ssp_spectra,
    filter_waves,
    filter_trans,
    cosmo_params,
):
    ssp_obs_photflux_table = _get_colors_single_redshift_vmap_kern(
        z_arr,
        ssp_waves,
        ssp_spectra,
        filter_waves,
        filter_trans,
        cosmo_params,
    )
    ssp_obs_photflux_table = jnp.einsum("zmab->mabz", ssp_obs_photflux_table)
    return ssp_obs_photflux_table


interpolate_ssp_photmag_table_vmap = jjit(
    vmap(interpolate_ssp_photmag_table, in_axes=(None, None, 0))
)


@jjit
def interpolate_ssp_obs_photflux_table_single_gal(z_gal, z_table, ssp_photmag_table):
    nmet, nage, nbands, nz = ssp_photmag_table.shape
    ssp_photmag_table_reshape = ssp_photmag_table.reshape((nmet * nage * nbands, nz))
    interp_table = interpolate_ssp_photmag_table_vmap(
        jnp.atleast_1d(z_gal), z_table, ssp_photmag_table_reshape
    )
    interp_table = interp_table[:, 0]
    interp_table = interp_table.reshape(nmet, nage, nbands)
    return interp_table


interpolate_ssp_obs_photflux_table = jjit(
    vmap(interpolate_ssp_obs_photflux_table_single_gal, in_axes=(0, None, None))
)


def interpolate_ssp_obs_photflux_table_batch(z_gal, z_table, ssp_photmag_table):
    nmet, nage, nbands, nz = ssp_photmag_table.shape
    ng = len(z_gal)
    indices = np.array_split(np.arange(ng), max(int(ng) / (int(1e4)), 1))
    interp_table_out = np.zeros((ng, nmet, nage, nbands))
    for inds in indices:
        interp_table_out[inds] = interpolate_ssp_obs_photflux_table(
            z_gal[inds], z_table, ssp_photmag_table
        )
    return interp_table_out


@jjit
def get_colors_single_object(
    gal_t_table,
    gal_sfr_table,
    gal_z_obs,
    gal_ssp_obs_photflux_table,
    ran_key,
    filter_waves,
    filter_trans,
    ssp_lgmet,
    ssp_lg_age_gyr,
    cosmo_params,
    lgfburst_u_params,
    burstshape_u_params,
    lgav_dust_u_params,
    delta_dust_u_params,
    boris_dust_u_params,
):
    gal_sfr_table = jnp.atleast_2d(gal_sfr_table)

    res = get_obs_photometry_singlez(
        ran_key,
        filter_waves,
        filter_trans,
        gal_ssp_obs_photflux_table,
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
        gal_z_obs,
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

    gal_obsflux = gal_obsflux[0]

    gal_obs_mags = -2.5 * jnp.log10(gal_obsflux)

    return gal_obs_mags


_A = (None, 0, 0, 0, 0, *[None] * 10)
get_colors_pop = jjit(vmap(get_colors_single_object, in_axes=_A))


@jjit
def calc_hist_1d(nddata, ndsig, weight, ndbins_lo, ndbins_hi):
    eps = 1e-50
    nd = len(nddata)
    nb = len(ndbins_lo)

    counts = tw_ndhist_weighted(
        nddata.reshape((nd, 1)),
        ndsig.reshape((nd, 1)),
        weight,
        ndbins_lo.reshape((nb, 1)),
        ndbins_hi.reshape((nb, 1)),
    )
    counts = jnp.clip(counts, eps, None)
    return counts


# vmap (bins,) -> (colors, bins)
calc_hist_1d_vmap = jjit(
    vmap(calc_hist_1d, in_axes=(0, *[None] * 4)),
)


@jjit
def return_weights_magbin(mag, mag_cut_bright, mag_cut_faint):
    weights_magcut_bright = _tw_cuml_lax_kern_vmap(mag, mag_cut_bright, 0.1)
    weights_magcut_faint = 1.0 - _tw_cuml_lax_kern_vmap(mag, mag_cut_faint, 0.1)
    weights_magcut = weights_magcut_bright * weights_magcut_faint
    return weights_magcut


@jjit
def calculate_dNdz_DEEP2(mag_r, mag_i, z_obs, bins_dNdz):
    """
    Function to calculate DEEP2 dNdz for CFHT r and CFHT i magnitude bins
    from Table 4 in Coil et. al. (2004).

    The following bins are implemented:
        - 18 < i < 20
        - 18 < i < 21
        - 18 < i < 22
        - 18 < i < 23
        - 18 < r < 20
        - 18 < r < 21
        - 18 < r < 22
        - 18 < r < 23
    """
    weights_imag_18_20 = return_weights_magbin(mag_i, 18.0, 20.0)
    weights_imag_18_21 = return_weights_magbin(mag_i, 18.0, 21.0)
    weights_imag_18_22 = return_weights_magbin(mag_i, 18.0, 22.0)
    weights_imag_18_23 = return_weights_magbin(mag_i, 18.0, 23.0)

    dNdz_imag_18_20 = jnp.histogram(
        z_obs, bins_dNdz, weights=weights_imag_18_20, density=1
    )
    dNdz_imag_18_21 = jnp.histogram(
        z_obs, bins_dNdz, weights=weights_imag_18_21, density=1
    )
    dNdz_imag_18_22 = jnp.histogram(
        z_obs, bins_dNdz, weights=weights_imag_18_22, density=1
    )
    dNdz_imag_18_23 = jnp.histogram(
        z_obs, bins_dNdz, weights=weights_imag_18_23, density=1
    )

    weights_rmag_18_20 = return_weights_magbin(mag_r, 18.0, 20.0)
    weights_rmag_18_21 = return_weights_magbin(mag_r, 18.0, 21.0)
    weights_rmag_18_22 = return_weights_magbin(mag_r, 18.0, 22.0)
    weights_rmag_18_23 = return_weights_magbin(mag_r, 18.0, 23.0)

    dNdz_rmag_18_20 = jnp.histogram(
        z_obs, bins_dNdz, weights=weights_rmag_18_20, density=1
    )
    dNdz_rmag_18_21 = jnp.histogram(
        z_obs, bins_dNdz, weights=weights_rmag_18_21, density=1
    )
    dNdz_rmag_18_22 = jnp.histogram(
        z_obs, bins_dNdz, weights=weights_rmag_18_22, density=1
    )
    dNdz_rmag_18_23 = jnp.histogram(
        z_obs, bins_dNdz, weights=weights_rmag_18_23, density=1
    )

    output = (
        dNdz_rmag_18_20,
        dNdz_rmag_18_21,
        dNdz_rmag_18_22,
        dNdz_rmag_18_23,
        dNdz_imag_18_20,
        dNdz_imag_18_21,
        dNdz_imag_18_22,
        dNdz_imag_18_23,
    )
    return output


@jjit
def calculate_1d_COSMOS_colors_counts_singlez_bin(
    gal_mags,
    ndsig_mag,
    ndsig_color,
    bins_LO_mag,
    bins_HI_mag,
    bins_LO_color,
    bins_HI_color,
):
    # ng = len(gal_mags)
    imag = gal_mags[:, 3]  # i-band
    gal_col = -jnp.diff(gal_mags, axis=1).T  # u-g, g-r, r-i, i-z, z-Y, Y-J, J-H, H-K

    weights_magcut = return_weights_magbin(imag, 18.0, 23.0)

    ng = jnp.sum(weights_magcut)

    # weight_final = jnp.einsum("zh,h->zh", weights_magcut, weight)

    counts_i = (
        calc_hist_1d(
            imag,
            ndsig_mag,
            weights_magcut,
            bins_LO_mag,
            bins_HI_mag,
        )
        / ng
    )

    counts_colors = (
        calc_hist_1d_vmap(
            gal_col,
            ndsig_color,
            weights_magcut,
            bins_LO_color,
            bins_HI_color,
        )
        / ng
    )

    return counts_i, counts_colors


@jjit
def calculate_1d_COSMOS_colors_counts(
    gal_mags_z01_03,
    gal_mags_z03_05,
    gal_mags_z05_07,
    gal_mags_z07_09,
    gal_mags_z09_11,
    gal_mags_z11_13,
    gal_mags_z13_15,
    ndsig_mag,
    ndsig_color,
    bins_LO_mag,
    bins_HI_mag,
    bins_LO_color,
    bins_HI_color,
):
    """
    Function to calculate COSMOS i-mag and color distributions conditioned on redshift bins.

    Function assumes u,g,r,i,z,Y,J,H,K magnitudes are supplied.

    The following bins are implemented:
        * 0.1 < redshift < 0.3
        * 0.3 < redshift < 0.5
        * 0.5 < redshift < 0.7
        * 0.7 < redshift < 0.9
        * 0.9 < redshift < 1.1
        * 1.1 < redshift < 1.3
        * 1.3 < redshift < 1.5
    """
    args = (
        ndsig_mag,
        ndsig_color,
        bins_LO_mag,
        bins_HI_mag,
        bins_LO_color,
        bins_HI_color,
    )

    (
        counts_i_z01_03,
        counts_colors_z01_03,
    ) = calculate_1d_COSMOS_colors_counts_singlez_bin(gal_mags_z01_03, *args)

    (
        counts_i_z03_05,
        counts_colors_z03_05,
    ) = calculate_1d_COSMOS_colors_counts_singlez_bin(gal_mags_z03_05, *args)

    (
        counts_i_z05_07,
        counts_colors_z05_07,
    ) = calculate_1d_COSMOS_colors_counts_singlez_bin(gal_mags_z05_07, *args)

    (
        counts_i_z07_09,
        counts_colors_z07_09,
    ) = calculate_1d_COSMOS_colors_counts_singlez_bin(gal_mags_z07_09, *args)

    (
        counts_i_z09_11,
        counts_colors_z09_11,
    ) = calculate_1d_COSMOS_colors_counts_singlez_bin(gal_mags_z09_11, *args)

    (
        counts_i_z11_13,
        counts_colors_z11_13,
    ) = calculate_1d_COSMOS_colors_counts_singlez_bin(gal_mags_z11_13, *args)

    (
        counts_i_z13_15,
        counts_colors_z13_15,
    ) = calculate_1d_COSMOS_colors_counts_singlez_bin(gal_mags_z13_15, *args)

    output = (
        counts_i_z01_03,
        counts_i_z03_05,
        counts_i_z05_07,
        counts_i_z07_09,
        counts_i_z09_11,
        counts_i_z11_13,
        counts_i_z13_15,
        counts_colors_z01_03,
        counts_colors_z03_05,
        counts_colors_z05_07,
        counts_colors_z07_09,
        counts_colors_z09_11,
        counts_colors_z11_13,
        counts_colors_z13_15,
    )
    return output


@jjit
def calculate_1d_SDSS_colors_counts(
    gal_mags,
    z_obs,
    ndsig_color,
    bins_LO_color,
    bins_HI_color,
):
    """
    Function to calculate COSMOS color distributions for m_r < 17.7 and 0.05 < redshift < 0.1.

    Function assumes u,g,r,i,z magnitudes are supplied.

    """
    mask = (z_obs > 0.05) & (z_obs < 0.1)

    gal_mags_masked = gal_mags[mask]

    ng = len(gal_mags_masked)
    rmag = gal_mags_masked[:, 2]  # r-band
    gal_col = -jnp.diff(gal_mags_masked, axis=1).T  # u-g, g-r, r-i, i-z

    # r < 17.7
    weights_magcut = weights_magcut_faint = 1.0 - _tw_cuml_lax_kern_vmap(
        rmag, 17.7, 0.1
    )

    counts_colors = (
        calc_hist_1d_vmap(
            gal_col,
            ndsig_color,
            weights_magcut,
            bins_LO_color,
            bins_HI_color,
        )
        / ng
    )

    return counts_colors


@jjit
def cumulative_imag_kern(mag, mag_cut_faint):
    weights_magcut_faint = 1.0 - _tw_cuml_lax_kern_vmap(mag, mag_cut_faint, 0.1)
    return jnp.sum(weights_magcut_faint)


cumulative_imag = jjit(vmap(cumulative_imag_kern, in_axes=(None, 0)))


@jjit
def calculate_1d_HSC_cumulative_imag(mag_i, mag_i_bins, area_norm):
    """
    Function to calculate HSC cumulative i-band number density (gal/sq.deg).
    """
    counts_imag = cumulative_imag(mag_i, mag_i_bins)

    return counts_imag / area_norm


@jjit
def loss_DEEP2(params, loss_data, ran_key):
    (
        t_table,
        gal_sfr_arr,
        gal_z_arr,
        gal_ssp_obs_photflux_table,
        filter_waves,
        filter_trans,
        ssp_lgmet,
        ssp_lg_age_gyr,
        cosmo_params,
        bins_dNdz,
        target_data,
    ) = loss_data

    (
        dNdz_rmag_18_20_target,
        dNdz_rmag_18_21_target,
        dNdz_rmag_18_22_target,
        dNdz_rmag_18_23_target,
        dNdz_imag_18_20_target,
        dNdz_imag_18_21_target,
        dNdz_imag_18_22_target,
        dNdz_imag_18_23_target,
    ) = target_data

    ran_key_arr = jran.split(ran_key, len(gal_z_arr))

    _npar = 0
    lgfburst_u_params = params[_npar : _npar + N_BURST_F]
    _npar += N_BURST_F
    burstshape_u_params = params[_npar : _npar + N_BURST_SHAPE]
    _npar += N_BURST_SHAPE
    lgav_dust_u_params = params[_npar : _npar + N_DUST_LGAV]
    _npar += N_DUST_LGAV
    delta_dust_u_params = params[_npar : _npar + N_DUST_DELTA]
    _npar += N_DUST_DELTA
    boris_dust_u_params = params[_npar : _npar + N_DUST_BORIS]
    _npar += N_DUST_BORIS

    mag_r_CFHT, mag_i_CFHT = get_colors_pop(
        t_table,
        gal_sfr_arr,
        gal_z_arr,
        gal_ssp_obs_photflux_table,
        ran_key_arr,
        filter_waves,
        filter_trans,
        ssp_lgmet,
        ssp_lg_age_gyr,
        cosmo_params,
        lgfburst_u_params,
        burstshape_u_params,
        lgav_dust_u_params,
        delta_dust_u_params,
        boris_dust_u_params,
    )

    pred_data = calculate_dNdz_DEEP2(mag_r_CFHT, mag_i_CFHT, gal_z_arr, bins_dNdz)

    (
        dNdz_rmag_18_20,
        dNdz_rmag_18_21,
        dNdz_rmag_18_22,
        dNdz_rmag_18_23,
        dNdz_imag_18_20,
        dNdz_imag_18_21,
        dNdz_imag_18_22,
        dNdz_imag_18_23,
    ) = pred_data

    loss = mse(dNdz_rmag_18_20, dNdz_rmag_18_20_target)
    loss += mse(dNdz_rmag_18_21, dNdz_rmag_18_21_target)
    loss += mse(dNdz_rmag_18_22, dNdz_rmag_18_22_target)
    loss += mse(dNdz_rmag_18_23, dNdz_rmag_18_23_target)
    loss += mse(dNdz_imag_18_20, dNdz_imag_18_20_target)
    loss += mse(dNdz_imag_18_21, dNdz_imag_18_21_target)
    loss += mse(dNdz_imag_18_22, dNdz_imag_18_22_target)
    loss += mse(dNdz_imag_18_23, dNdz_imag_18_23_target)

    return loss


@jjit
def loss_COSMOS(params, loss_data, ran_key):
    (
        t_table,
        gal_sfr_arr,
        gal_z_arr,
        gal_ssp_obs_photflux_table,
        filter_waves,
        filter_trans,
        ssp_lgmet,
        ssp_lg_age_gyr,
        cosmo_params,
        ndsig_mag,
        ndsig_color,
        bins_LO_mag,
        bins_HI_mag,
        bins_LO_color,
        bins_HI_color,
        target_data_COSMOS,
    ) = loss_data

    (
        counts_i_z01_03_target,
        counts_i_z03_05_target,
        counts_i_z05_07_target,
        counts_i_z07_09_target,
        counts_i_z09_11_target,
        counts_i_z11_13_target,
        counts_i_z13_15_target,
        counts_colors_z01_03_target,
        counts_colors_z03_05_target,
        counts_colors_z05_07_target,
        counts_colors_z07_09_target,
        counts_colors_z09_11_target,
        counts_colors_z11_13_target,
        counts_colors_z13_15_target,
    ) = target_data_COSMOS

    (
        gal_sfr_arr_z01_03,
        gal_sfr_arr_z03_05,
        gal_sfr_arr_z05_07,
        gal_sfr_arr_z07_09,
        gal_sfr_arr_z09_11,
        gal_sfr_arr_z11_13,
        gal_sfr_arr_z13_15,
    ) = gal_sfr_arr
    (
        gal_z_arr_z01_03,
        gal_z_arr_z03_05,
        gal_z_arr_z05_07,
        gal_z_arr_z07_09,
        gal_z_arr_z09_11,
        gal_z_arr_z11_13,
        gal_z_arr_z13_15,
    ) = gal_z_arr
    (
        gal_ssp_obs_photflux_table_z01_03,
        gal_ssp_obs_photflux_table_z03_05,
        gal_ssp_obs_photflux_table_z05_07,
        gal_ssp_obs_photflux_table_z07_09,
        gal_ssp_obs_photflux_table_z09_11,
        gal_ssp_obs_photflux_table_z11_13,
        gal_ssp_obs_photflux_table_z13_15,
    ) = gal_ssp_obs_photflux_table

    ran_key_arr_z01_03 = jran.split(ran_key, len(gal_z_arr_z01_03))
    ran_key_arr_z03_05 = jran.split(ran_key, len(gal_z_arr_z03_05))
    ran_key_arr_z05_07 = jran.split(ran_key, len(gal_z_arr_z05_07))
    ran_key_arr_z07_09 = jran.split(ran_key, len(gal_z_arr_z07_09))
    ran_key_arr_z09_11 = jran.split(ran_key, len(gal_z_arr_z09_11))
    ran_key_arr_z11_13 = jran.split(ran_key, len(gal_z_arr_z11_13))
    ran_key_arr_z13_15 = jran.split(ran_key, len(gal_z_arr_z13_15))

    _npar = 0
    lgfburst_u_params = params[_npar : _npar + N_BURST_F]
    _npar += N_BURST_F
    burstshape_u_params = params[_npar : _npar + N_BURST_SHAPE]
    _npar += N_BURST_SHAPE
    lgav_dust_u_params = params[_npar : _npar + N_DUST_LGAV]
    _npar += N_DUST_LGAV
    delta_dust_u_params = params[_npar : _npar + N_DUST_DELTA]
    _npar += N_DUST_DELTA
    boris_dust_u_params = params[_npar : _npar + N_DUST_BORIS]
    _npar += N_DUST_BORIS

    def get_colors_pop_COSMOS(
        gal_sfr_arr_COS, gal_z_arr_COS, gal_ssp_obs_photflux_table_COS, ran_key_arr_COS
    ):
        gal_mags = get_colors_pop(
            t_table,
            gal_sfr_arr_COS,
            gal_z_arr_COS,
            gal_ssp_obs_photflux_table_COS,
            ran_key_arr_COS,
            filter_waves,
            filter_trans,
            ssp_lgmet,
            ssp_lg_age_gyr,
            cosmo_params,
            lgfburst_u_params,
            burstshape_u_params,
            lgav_dust_u_params,
            delta_dust_u_params,
            boris_dust_u_params,
        )
        return gal_mags

    gal_mags_z01_03 = get_colors_pop_COSMOS(
        gal_sfr_arr_z01_03,
        gal_z_arr_z01_03,
        gal_ssp_obs_photflux_table_z01_03,
        ran_key_arr_z01_03,
    )
    gal_mags_z03_05 = get_colors_pop_COSMOS(
        gal_sfr_arr_z03_05,
        gal_z_arr_z03_05,
        gal_ssp_obs_photflux_table_z03_05,
        ran_key_arr_z03_05,
    )
    gal_mags_z05_07 = get_colors_pop_COSMOS(
        gal_sfr_arr_z05_07,
        gal_z_arr_z05_07,
        gal_ssp_obs_photflux_table_z05_07,
        ran_key_arr_z05_07,
    )
    gal_mags_z07_09 = get_colors_pop_COSMOS(
        gal_sfr_arr_z07_09,
        gal_z_arr_z07_09,
        gal_ssp_obs_photflux_table_z07_09,
        ran_key_arr_z07_09,
    )
    gal_mags_z09_11 = get_colors_pop_COSMOS(
        gal_sfr_arr_z09_11,
        gal_z_arr_z09_11,
        gal_ssp_obs_photflux_table_z09_11,
        ran_key_arr_z09_11,
    )
    gal_mags_z11_13 = get_colors_pop_COSMOS(
        gal_sfr_arr_z11_13,
        gal_z_arr_z11_13,
        gal_ssp_obs_photflux_table_z11_13,
        ran_key_arr_z11_13,
    )
    gal_mags_z13_15 = get_colors_pop_COSMOS(
        gal_sfr_arr_z13_15,
        gal_z_arr_z13_15,
        gal_ssp_obs_photflux_table_z13_15,
        ran_key_arr_z13_15,
    )

    pred_data = calculate_1d_COSMOS_colors_counts(
        gal_mags_z01_03,
        gal_mags_z03_05,
        gal_mags_z05_07,
        gal_mags_z07_09,
        gal_mags_z09_11,
        gal_mags_z11_13,
        gal_mags_z13_15,
        gal_z_arr,
        ndsig_mag,
        ndsig_color,
        bins_LO_mag,
        bins_HI_mag,
        bins_LO_color,
        bins_HI_color,
    )

    (
        counts_i_z01_03,
        counts_i_z03_05,
        counts_i_z05_07,
        counts_i_z07_09,
        counts_i_z09_11,
        counts_i_z11_13,
        counts_i_z13_15,
        counts_colors_z01_03,
        counts_colors_z03_05,
        counts_colors_z05_07,
        counts_colors_z07_09,
        counts_colors_z09_11,
        counts_colors_z11_13,
        counts_colors_z13_15,
    ) = pred_data

    loss = mse(counts_i_z01_03, counts_i_z01_03_target)
    loss += mse(counts_i_z03_05, counts_i_z03_05_target)
    loss += mse(counts_i_z05_07, counts_i_z05_07_target)
    loss += mse(counts_i_z07_09, counts_i_z07_09_target)
    loss += mse(counts_i_z09_11, counts_i_z09_11_target)
    loss += mse(counts_i_z11_13, counts_i_z11_13_target)
    loss += mse(counts_i_z13_15, counts_i_z13_15_target)

    loss += mse(counts_colors_z01_03, counts_colors_z01_03_target)
    loss += mse(counts_colors_z03_05, counts_colors_z03_05_target)
    loss += mse(counts_colors_z05_07, counts_colors_z05_07_target)
    loss += mse(counts_colors_z07_09, counts_colors_z07_09_target)
    loss += mse(counts_colors_z09_11, counts_colors_z09_11_target)
    loss += mse(counts_colors_z11_13, counts_colors_z11_13_target)
    loss += mse(counts_colors_z13_15, counts_colors_z13_15_target)

    return loss


@jjit
def loss_HSC(params, loss_data, ran_key):
    (
        t_table,
        gal_sfr_arr,
        gal_z_arr,
        gal_ssp_obs_photflux_table,
        filter_waves,
        filter_trans,
        ssp_lgmet,
        ssp_lg_age_gyr,
        cosmo_params,
        mag_i_bins,
        area_norm,
        target_data_HSC,
    ) = loss_data

    ran_key_arr = jran.split(ran_key, len(gal_z_arr))

    _npar = 0
    lgfburst_u_params = params[_npar : _npar + N_BURST_F]
    _npar += N_BURST_F
    burstshape_u_params = params[_npar : _npar + N_BURST_SHAPE]
    _npar += N_BURST_SHAPE
    lgav_dust_u_params = params[_npar : _npar + N_DUST_LGAV]
    _npar += N_DUST_LGAV
    delta_dust_u_params = params[_npar : _npar + N_DUST_DELTA]
    _npar += N_DUST_DELTA
    boris_dust_u_params = params[_npar : _npar + N_DUST_BORIS]
    _npar += N_DUST_BORIS

    mag_i = get_colors_pop(
        t_table,
        gal_sfr_arr,
        gal_z_arr,
        gal_ssp_obs_photflux_table,
        ran_key_arr,
        filter_waves,
        filter_trans,
        ssp_lgmet,
        ssp_lg_age_gyr,
        cosmo_params,
        lgfburst_u_params,
        burstshape_u_params,
        lgav_dust_u_params,
        delta_dust_u_params,
        boris_dust_u_params,
    )[:, 0]
    mag_i_cdf = calculate_1d_HSC_cumulative_imag(mag_i, mag_i_bins, area_norm)

    loss = mse(mag_i_cdf, target_data_HSC)

    return loss


@jjit
def loss_SDSS(params, loss_data, ran_key):
    (
        t_table,
        gal_sfr_arr,
        gal_z_arr,
        gal_ssp_obs_photflux_table,
        filter_waves,
        filter_trans,
        ssp_lgmet,
        ssp_lg_age_gyr,
        cosmo_params,
        ndsig_color,
        bins_LO_color,
        bins_HI_color,
        target_data,
    ) = loss_data

    ran_key_arr = jran.split(ran_key, len(gal_z_arr))

    _npar = 0
    lgfburst_u_params = params[_npar : _npar + N_BURST_F]
    _npar += N_BURST_F
    burstshape_u_params = params[_npar : _npar + N_BURST_SHAPE]
    _npar += N_BURST_SHAPE
    lgav_dust_u_params = params[_npar : _npar + N_DUST_LGAV]
    _npar += N_DUST_LGAV
    delta_dust_u_params = params[_npar : _npar + N_DUST_DELTA]
    _npar += N_DUST_DELTA
    boris_dust_u_params = params[_npar : _npar + N_DUST_BORIS]
    _npar += N_DUST_BORIS

    gal_mags = get_colors_pop(
        t_table,
        gal_sfr_arr,
        gal_z_arr,
        gal_ssp_obs_photflux_table,
        ran_key_arr,
        filter_waves,
        filter_trans,
        ssp_lgmet,
        ssp_lg_age_gyr,
        cosmo_params,
        lgfburst_u_params,
        burstshape_u_params,
        lgav_dust_u_params,
        delta_dust_u_params,
        boris_dust_u_params,
    )

    pred_data = calculate_1d_SDSS_colors_counts(
        gal_mags,
        gal_z_arr,
        ndsig_color,
        bins_LO_color,
        bins_HI_color,
    )

    loss = mse(pred_data, target_data)

    return loss


@jjit
def loss_HSC(params, loss_data, ran_key):
    (
        t_table,
        gal_sfr_arr,
        gal_z_arr,
        gal_ssp_obs_photflux_table,
        dsps_data,
        mag_i_bins,
        area_norm,
        target_data_HSC,
    ) = loss_data

    ran_key_arr = jran.split(ran_key, len(gal_z_arr))

    _npar = 0
    lgfburst_u_params = params[_npar : _npar + N_BURST_F]
    _npar += N_BURST_F
    burstshape_u_params = params[_npar : _npar + N_BURST_SHAPE]
    _npar += N_BURST_SHAPE
    lgav_dust_u_params = params[_npar : _npar + N_DUST_LGAV]
    _npar += N_DUST_LGAV
    delta_dust_u_params = params[_npar : _npar + N_DUST_DELTA]
    _npar += N_DUST_DELTA
    boris_dust_u_params = params[_npar : _npar + N_DUST_BORIS]
    _npar += N_DUST_BORIS

    gal_mags = get_colors_pop(
        t_table,
        gal_sfr_arr,
        gal_z_arr,
        gal_ssp_obs_photflux_table,
        ran_key_arr,
        dsps_data,
        lgfburst_u_params,
        burstshape_u_params,
        lgav_dust_u_params,
        delta_dust_u_params,
        boris_dust_u_params,
    )

    mag_i = gal_mags[:, 3]
    mag_i_cdf = calculate_1d_HSC_cumulative_imag(mag_i, mag_i_bins, area_norm)

    loss = mse(mag_i_cdf, target_data_HSC)

    return loss


loss_COSMOS_deriv = jjit(grad(loss_COSMOS, argnums=(0)))
loss_DEEP2_deriv = jjit(grad(loss_DEEP2, argnums=(0)))
loss_SDSS_deriv = jjit(grad(loss_SDSS, argnums=(0)))


def loss_COSMOS_deriv_np(params, data, n_histories):
    return np.array(loss_COSMOS_deriv(params, data, n_histories)).astype(float)


def loss_DEEP2_deriv_np(params, data, n_histories):
    return np.array(loss_DEEP2_deriv(params, data, n_histories)).astype(float)


def loss_SDSS_deriv_np(params, data, n_histories):
    return np.array(loss_SDSS_deriv(params, data, n_histories)).astype(float)


DSPS_data_path = "/lcrc/project/halotools/alarcon/data/DSPS_data/"
DSPS_COSMOS_filter_path = DSPS_data_path + "filters/cosmos/"
target_data_path = "/lcrc/project/halotools/alarcon/data/DESC_mocks_data/"


def get_loss_data_COSMOS(
    t_table,
    gal_sfr_arr,
    gal_z_arr,
):
    print("Loading COSMOS filters")
    filter_list_COSMOS = [
        "COSMOS_CFHT_ustar.h5",
        "COSMOS_HSC_g.h5",
        "COSMOS_HSC_r.h5",
        "COSMOS_HSC_i.h5",
        "COSMOS_HSC_z.h5",
        "COSMOS_UVISTA_Y.h5",
        "COSMOS_UVISTA_J.h5",
        "COSMOS_UVISTA_H.h5",
        "COSMOS_UVISTA_Ks.h5",
    ]

    def get_filter_data(filter_list, filter_path):
        filter_data = [
            load_transmission_curve(drn=filter_path, bn_pat=filt)
            for filt in filter_list
        ]
        wave_filters = [x[0] for x in filter_data]
        trans_filters = [x[1] for x in filter_data]
        filter_waves, filter_trans = interpolate_filter_trans_curves(
            wave_filters, trans_filters
        )
        return filter_waves, filter_trans

    filter_waves_COSMOS, filter_trans_COSMOS = get_filter_data(
        filter_list_COSMOS, DSPS_COSMOS_filter_path
    )
    # filter_waves_DEEP2, filter_trans_DEEP2 = get_filter_data(filter_list_DEEP2, DSPS_DEEP2_filter_path)
    # filter_waves_SDSS, filter_trans_SDSS = get_filter_data(filter_list_SDSS, DSPS_SDSS_filter_path)

    print("Calculating SSPs in COSMOS filters in z_table")
    z_table = np.linspace(gal_z_arr.min() - 1e-3, gal_z_arr.max() + 1e-3, 100)

    ssp_data = load_ssp_templates(drn=DSPS_data_path)
    ssp_lgmet = ssp_data[0]
    ssp_lg_age_gyr = ssp_data[1]
    ssp_waves = ssp_data[2]
    ssp_spectra = ssp_data[3]

    ssp_obs_photflux_table_COSMOS = get_colors_array(
        z_table,
        ssp_waves,
        ssp_spectra,
        filter_waves_COSMOS,
        filter_trans_COSMOS,
        DEFAULT_COSMOLOGY,
    )
    # gal_ssp_table_COSMOS = interpolate_ssp_obs_photflux_table(
    #     gal_z_arr, z_table, ssp_obs_photflux_table_COSMOS
    # )
    print("Interpolating SSPs to galaxy redshifts")
    gal_ssp_table_COSMOS = interpolate_ssp_obs_photflux_table_batch(
        gal_z_arr, z_table, ssp_obs_photflux_table_COSMOS
    )

    fn = "COSMOS_target_data_20bins.h5"
    with h5py.File(os.path.join(target_data_path, fn), "r") as f:
        bins_mag_COSMOS = f["bins_mag"][...]
        bins_color_COSMOS = f["bins_color"][...]
        diff_i_counts = f["diff_i_counts"][...]
        diff_color_counts = f["diff_color_counts"][...]

    target_data_COSMOS = (*diff_i_counts, *diff_color_counts)
    bins_LO_mag_COSMOS = bins_mag_COSMOS[:-1]
    bins_HI_mag_COSMOS = bins_mag_COSMOS[1:]
    ndsig_mag_COSMOS = np.ones_like(gal_z_arr) * np.diff(bins_mag_COSMOS)[0]

    bins_LO_color_COSMOS = bins_color_COSMOS[:-1]
    bins_HI_color_COSMOS = bins_color_COSMOS[1:]
    ndsig_color_COSMOS = np.ones_like(gal_z_arr) * np.diff(bins_color_COSMOS)[0]

    loss_data_COSMOS = (
        t_table,
        gal_sfr_arr,
        gal_z_arr,
        gal_ssp_table_COSMOS,
        filter_waves_COSMOS,
        filter_trans_COSMOS,
        ssp_lgmet,
        ssp_lg_age_gyr,
        DEFAULT_COSMOLOGY,
        ndsig_mag_COSMOS,
        ndsig_color_COSMOS,
        bins_LO_mag_COSMOS,
        bins_HI_mag_COSMOS,
        bins_LO_color_COSMOS,
        bins_HI_color_COSMOS,
        target_data_COSMOS,
    )

    for x in loss_data_COSMOS[:-1]:
        assert np.all(np.isfinite(x))
    for x in target_data_COSMOS:
        assert np.all(np.isfinite(x))

    return loss_data_COSMOS


def get_loss_data_HSC(
    t_table,
    gal_sfr_arr,
    gal_z_arr,
    area_norm_HSC,
):
    print("Loading HSC filters")
    filter_list_HSC = [
        "COSMOS_HSC_i.h5",
    ]

    def get_filter_data(filter_list, filter_path):
        filter_data = [
            load_transmission_curve(drn=filter_path, bn_pat=filt)
            for filt in filter_list
        ]
        wave_filters = [x[0] for x in filter_data]
        trans_filters = [x[1] for x in filter_data]
        filter_waves, filter_trans = interpolate_filter_trans_curves(
            wave_filters, trans_filters
        )
        return filter_waves, filter_trans

    filter_waves_HSC, filter_trans_HSC = get_filter_data(
        filter_list_HSC, DSPS_COSMOS_filter_path
    )
    print("Calculating SSPs in HSC filters in z_table")
    z_table = np.linspace(gal_z_arr.min() - 1e-3, gal_z_arr.max() + 1e-3, 100)

    ssp_data = load_ssp_templates(drn=DSPS_data_path)
    ssp_lgmet = ssp_data[0]
    ssp_lg_age_gyr = ssp_data[1]
    ssp_waves = ssp_data[2]
    ssp_spectra = ssp_data[3]

    ssp_obs_photflux_table_HSC = get_colors_array(
        z_table,
        ssp_waves,
        ssp_spectra,
        filter_waves_HSC,
        filter_trans_HSC,
        DEFAULT_COSMOLOGY,
    )
    print("Interpolating SSPs to galaxy redshifts")
    gal_ssp_table_HSC = interpolate_ssp_obs_photflux_table_batch(
        gal_z_arr, z_table, ssp_obs_photflux_table_HSC
    )

    fn = "HSC_target_data_20_24_limits.h5"
    with h5py.File(os.path.join(target_data_path, fn), "r") as f:
        mag_i_bins_HSC = f["mag_i_bins"][...]
        target_data_HSC = f["hsc_cumcounts_imag_tri"][...]

    loss_data_HSC = (
        t_table,
        gal_sfr_arr,
        gal_z_arr,
        gal_ssp_table_HSC,
        filter_waves_HSC,
        filter_trans_HSC,
        ssp_lgmet,
        ssp_lg_age_gyr,
        DEFAULT_COSMOLOGY,
        mag_i_bins_HSC,
        area_norm_HSC,
        target_data_HSC,
    )

    for x in loss_data_HSC:
        assert np.all(np.isfinite(x))

    return loss_data_HSC


filter_list_DEEP2 = [
    "CFHT_r_megaprime_sagem.h5",
    "CFHT_i_megaprime_sagem.h5",
]

filter_list_SDSS = [
    "SDSS_u.h5",
    "SDSS_g.h5",
    "SDSS_r.h5",
    "SDSS_i.h5",
    "SDSS_z.h5",
]
