import re
import os
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import warnings
import h5py

from astropy.cosmology import Planck15, z_at_value

mred = "#d62728"
morange = "#ff7f0e"
mgreen = "#2ca02c"
mblue = "#1f77b4"
mpurple = "#9467bd"
plt.rc("font", family="serif")
plt.rc("font", size=22)
plt.rc("text", usetex=False)
plt.rc("text.latex", preamble=r"\usepackage{amsmath}")  # necessary to use \dfrac

import smdpl_smhm_utils
from smdpl_smhm_utils import load_diffstar_sfh_tables

from diffstar.data_loaders.load_smah_data import (
    FB_SMDPL,
    T0_SMDPL,
    load_smdpl_diffmah_fits,
    load_SMDPL_DR1_data,
    load_SMDPL_nomerging_data,
)

from jax import vmap, jit as jjit, numpy as jnp
from diffstar.defaults import TODAY


def _jnp_interp_vmap(x_new, x, y):
    return jnp.interp(x_new, x, y)


jnp_interp_vmap = jjit(vmap(_jnp_interp_vmap, in_axes=(None, None, 0)))


def calculate_smdpl_nomerging():
    diffmah_drn = smdpl_smhm_utils.LCRC_NOMERGING_DIFFMAH_DRN
    diffstar_drn = smdpl_smhm_utils.LCRC_NOMERGING_DIFFSTAR_DRN
    binaries_drn = smdpl_smhm_utils.LCRC_NOMERGING_BINARIES_DRN
    diffstar_bnpat = smdpl_smhm_utils.LCRC_NOMERGING_diffstar_bnpat
    sim_name = "DR1_nomerging"

    regex_str = re.escape(diffstar_bnpat).replace(r"\{\}", r"(\d{1,3})")
    pattern = re.compile(f"^{regex_str}$")
    matching_files = [f for f in os.listdir(diffstar_drn) if pattern.match(f)]
    subvols = [x.split("_")[1] for x in matching_files]
    subvols = np.sort(np.array(subvols).astype(int))
    n_subvol_smdpl = len(subvols)

    log_smahs_fits = []
    log_sfrh_fits = []
    log_smahs_data = []
    log_sfrh_data = []
    logmp0_data = []

    for subvol in range(10):

        out = smdpl_smhm_utils.load_diffstar_sfh_tables(
            subvol,
            sim_name,
            n_subvol_smdpl,
            diffmah_drn,
            diffstar_drn,
            diffstar_bnpat,
        )
        (
            t_table,
            log_mah_table,
            log_smh_table,
            log_ssfrh_table,
            mah_params,
            ms_params,
            q_params,
            has_fit,
        ) = out

        log_sfh_table = log_ssfrh_table + log_smh_table

        out = load_SMDPL_nomerging_data([subvol], binaries_drn)
        (halo_ids, log_smahs, sfrh, SMDPL_t, log_mahs, logmp0) = out
        log_sfrh = np.where(sfrh > 0.0, np.log10(sfrh), 0.0)

        _log_smahs_data = log_smahs[has_fit]
        _log_sfrh_data = log_sfrh[has_fit]

        _log_smahs_fits = jnp_interp_vmap(SMDPL_t, t_table, log_smh_table)
        _log_sfrh_fits = jnp_interp_vmap(SMDPL_t, t_table, log_sfh_table)

        log_smahs_fits.append(_log_smahs_fits)
        log_sfrh_fits.append(_log_sfrh_fits)
        log_smahs_data.append(_log_smahs_data)
        log_sfrh_data.append(_log_sfrh_data)
        logmp0_data.append(logmp0[has_fit])

    log_smahs_fits = np.concatenate(log_smahs_fits)
    log_sfrh_fits = np.concatenate(log_sfrh_fits)
    log_smahs_data = np.concatenate(log_smahs_data)
    log_sfrh_data = np.concatenate(log_sfrh_data)
    logmp0_data = np.concatenate(logmp0_data)

    out = (
        SMDPL_t,
        log_smahs_fits,
        log_sfrh_fits,
        log_smahs_data,
        log_sfrh_data,
        logmp0_data,
    )

    return out


def make_diffstar_fits_plot(
    outdir,
    sim_name,
    tarr,
    log_smahs_fits,
    log_sfrh_fits,
    log_smahs_data,
    log_sfrh_data,
    logmp0_data,
):
    smahs_fits = np.where(log_smahs_fits == 0.0, 0.0, 10**log_smahs_fits)
    sfrh_fits = np.where(log_sfrh_fits == 0.0, 0.0, 10**log_sfrh_fits)
    smahs_data = np.where(log_smahs_data == 0.0, 0.0, 10**log_smahs_data)
    sfrh_data = np.where(log_sfrh_data == 0.0, 0.0, 10**log_sfrh_data)

    fig, ax = plt.subplots(
        5,
        1,
        figsize=(6, 17),
        gridspec_kw={"height_ratios": [1.6, 1, 1.6, 1, 1], "hspace": 0},
        sharex=True,
    )

    colors = [
        "#0077BB",
        "#33BBEE",
        "#009988",
        "#EE7733",
        "#CC3311",
        "#EE3377",
        "#882255",
        "#AA4499",
    ]
    # colors = ["#0077BB", "#33BBEE", "#009988", "#EE7733", "#CC3311", "#EE3377", '#AA4499', '#882255']
    # mpeak_bins = np.arange(11.0,14.1,0.50)
    mpeak_bins = np.arange(11.25, 14.0, 0.50)

    ssfrh = sfrh_data / smahs_data
    ssfrh_fit = sfrh_fits / smahs_fits
    ssfrh = np.clip(ssfrh, 1e-12, np.inf)
    ssfrh_fit = np.clip(ssfrh_fit, 1e-12, np.inf)
    sfrh = np.where(smahs_data > 0.0, ssfrh * smahs_data, sfrh_data)
    sfrh_fits = ssfrh_fit * smahs_fits

    for i in range(len(mpeak_bins) - 1):
        masksel = (logmp0_data > mpeak_bins[i]) & (logmp0_data < mpeak_bins[i + 1])

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            mstar_data_mean = np.mean(smahs_data[masksel], axis=0)
            mstar_fit_mean = np.mean(smahs_fits[masksel], axis=0)

            ax[0].plot(tarr, mstar_data_mean, color=colors[i])
            ax[0].plot(tarr, mstar_fit_mean, color=colors[i], ls="--")

            smh = smahs_data[masksel]
            _mask = np.log10(smh)[:, [-1]] - np.log10(smh) < 3.5
            _mask &= np.log10(smh) > 7.0

            diff_smh = np.log10(smahs_fits[masksel]) - np.log10(smh)
            diff_smh = np.where(_mask, diff_smh, np.nan)
            diff_smh = np.where(smh == 0, np.nan, diff_smh)
            diff_smh_avg = np.nanmean(diff_smh, axis=0)

            diff_smh_avg = np.log10(mstar_fit_mean) - np.log10(mstar_data_mean)
            ax[1].plot(tarr, diff_smh_avg, color=colors[i])

            sfr_data_mean = np.mean(sfrh[masksel], axis=0)
            sfr_fit_mean = np.mean(sfrh_fits[masksel], axis=0)

            ax[2].plot(tarr, sfr_data_mean, color=colors[i])
            ax[2].plot(tarr, sfr_fit_mean, color=colors[i], ls="--")

            diff_smh_avg = np.log10(sfr_fit_mean) - np.log10(sfr_data_mean)
            ax[3].plot(tarr, diff_smh_avg, color=colors[i])

            diff_ssfh_avg = 1e8 * (sfr_fit_mean - sfr_data_mean) / mstar_data_mean

            ax[4].plot(tarr, diff_ssfh_avg, color=colors[i])

    fontsize = 18
    ax[0].set_yscale("log")
    ax[0].set_ylim(1e7, 5e11)
    ax[0].set_ylabel(r"$\langle M_\star | M_0 \rangle [M_{\odot}]$", fontsize=fontsize)

    ax[1].set_ylim(-0.5, 0.30)
    ax[1].set_yticks(np.arange(-0.4, 0.25, 0.2))
    ax[1].set_ylabel(
        r"$\log \left( \langle M^{\rm fit}_\star\rangle / \langle M^{\rm data}_\star \rangle \right) $"
    )

    ax[1].axhline(0.0, color="k", ls=":")

    ax[2].set_yscale("log")
    ax[2].set_ylabel(
        r"$\langle \dot{M}_\star | M_0 \rangle [M_{\odot}/{\rm yr}]$", fontsize=fontsize
    )
    ax[2].set_ylim(1e-2, 1e3)

    ax[3].set_ylim(-0.25, 0.25)
    ax[3].set_yticks(np.arange(-0.2, 0.21, 0.1))
    ax[3].set_ylabel(
        r"$\log ( \langle \dot{M}^{\rm fit}_\star\rangle / \langle \dot{M}^{\rm data}_\star \rangle ) $"
    )

    ax[3].axhline(0.0, color="k", ls=":")
    ax[3].set_xticks(np.arange(1.0, 14.0, 2.0))

    ax[4].axhline(0.0, color="k", ls=":")
    ax[4].set_ylim(-0.12, 0.12)
    ax[4].set_yticks(np.arange(-0.1, 0.11, 0.05))

    ax[4].set_ylabel(
        r"$\dfrac{\langle \dot{M}^{\rm fit}_\star\rangle - \langle \dot{M}^{\rm data}_\star \rangle}{\langle M^{\rm data}_\star \rangle / (100\,{\rm Myr})} $"
    )
    ax[4].set_xlabel("Cosmic time [Gyr]", fontsize=fontsize)
    ax[4].set_xlim(1.0, TODAY)
    ax[4].set_xticks(np.arange(1.0, 14.0, 2.0))

    mpeak_bins = np.arange(11.5, 14, 0.50)

    legend_elements = [
        Line2D([0], [0], color="k", ls="-", label="UniverseMachine"),
        Line2D([0], [0], color="k", ls="--", label="diffstar"),
    ]

    legend1 = ax[0].legend(handles=legend_elements, loc=4, ncol=1, fontsize=18)

    legend_elements = []
    for i in range(len(mpeak_bins)):
        legend_elements.append(
            Line2D(
                [0],
                [0],
                color=colors[i],
                # label=r'$[%.1f, %.1f]$'%(mpeak_bins[i],mpeak_bins[i+1]))
                # label=r'$M_0=10^{%.1f}\,M_\odot$'%(mpeak_bins[i]))
                label=r"$%.1f$" % (mpeak_bins[i]),
            )
        )

    ax[2].legend(
        handles=legend_elements,
        loc=1,
        ncol=2,
        fontsize=16,
        title_fontsize=16,
        title="$\log (M_0\,[M_\odot])$",
    )

    xlim = ax[0].set_xlim()
    ax2 = ax[0].twiny()
    ax2.set_xlim(xlim[0], xlim[1])
    ax2.plot([], [])

    ticks_z = np.array([0, 0.1, 0.3, 0.5, 1, 2, 3, 5])
    ticks_t = np.array(Planck15.age(ticks_z))
    ax[0].set_xticks(np.arange(1.0, 14.0, 2.0))

    ax2.set_xticks(ticks_t)
    ax2.set_xticklabels([r"$%.1f$" % x if x < 1 else r"$%d$" % x for x in ticks_z])
    ax2.set_xlabel(r"Redshift")
    ax2.xaxis.set_label_coords(0.5, 1.12)

    outname = f"average_histories_w_residuals_{sim_name}"
    out_path = os.path.join(outdir, outname)
    fig.savefig(out_path + ".png", bbox_inches="tight", dpi=300)
    fig.savefig(out_path + ".pdf", bbox_inches="tight")
    plt.clf()
    plt.close()


def save_data(outdrn, outname, data):
    fnout = os.path.join(outdrn, outname)

    (
        tarr,
        log_smahs_fits,
        log_sfrh_fits,
        log_smahs_data,
        log_sfrh_data,
        logmp0_data,
    ) = data

    with h5py.File(fnout, "w") as hdfout:
        hdfout["tarr"] = tarr
        hdfout["log_smahs_fits"] = log_smahs_fits
        hdfout["log_sfrh_fits"] = log_sfrh_fits
        hdfout["log_smahs_data"] = log_smahs_data
        hdfout["log_sfrh_data"] = log_sfrh_data
        hdfout["logmp0_data"] = logmp0_data


out_smdpl_nomerging = calculate_smdpl_nomerging()
outdir = "/lcrc/project/halotools/alarcon/results/diffstar_quality_fits/"
outname = "diffstar_quality_smdpl.h5"


# outdir = "/lcrc/project/halotools/alarcon/results/smdpl_pdf_target_data/"
# sim_name = "SMDPL_UM_Nomerging"
# make_diffstar_fits_plot(outdir, sim_name, *out_smdpl_nomerging)
