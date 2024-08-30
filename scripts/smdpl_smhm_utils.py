"""
"""

import os

import h5py
import numpy as np
from diffmah.diffmah_kernels import DEFAULT_MAH_PARAMS, mah_halopop
from diffsky.diffndhist import tw_ndhist_weighted
from diffstar.defaults import DEFAULT_DIFFSTAR_PARAMS, LGT0, T_TABLE_MIN
from diffstar.sfh_model_tpeak import calc_sfh_galpop

LCRC_DIFFSTAR_DRN = (
    "/lcrc/project/halotools/SMDPL/dr1_no_merging_upidh/diffstar_tpeak_fits"
)
LCRC_DIFFMAH_DRN = (
    "/lcrc/project/halotools/SMDPL/dr1_no_merging_upidh/diffmah_tpeak_fits"
)
TASSO_DIFFSTAR_DRN = "/Users/aphearin/work/DATA/diffstar_data/SMDPL"
N_SUBVOL_SMDPL = 576
LOGMH_BINS = np.linspace(11, 14.75, 40)


def _load_flat_hdf5(fn):
    data = dict()
    with h5py.File(fn, "r") as hdf:
        for key in hdf.keys():
            data[key] = hdf[key][...]
    return data


def load_diffstar_subvolume(
    subvol,
    n_subvol_tot=N_SUBVOL_SMDPL,
    diffmah_drn=TASSO_DIFFSTAR_DRN,
    diffstar_drn=TASSO_DIFFSTAR_DRN,
):
    nchar_subvol = len(str(n_subvol_tot))
    diffstar_bnpat = "subvol_{}_diffstar_fits.h5"
    subvol_str = f"{subvol:0{nchar_subvol}d}"
    diffstar_bn = diffstar_bnpat.format(subvol_str)
    diffstar_fn = os.path.join(diffstar_drn, diffstar_bn)
    diffstar_data = _load_flat_hdf5(diffstar_fn)

    diffmah_bn = diffstar_bn.replace("diffstar", "diffmah")
    diffmah_fn = os.path.join(diffmah_drn, diffmah_bn)
    diffmah_data = _load_flat_hdf5(diffmah_fn)

    return diffmah_data, diffstar_data


def load_diffstar_sfh_tables(
    subvol,
    n_subvol_tot=N_SUBVOL_SMDPL,
    diffmah_drn=TASSO_DIFFSTAR_DRN,
    diffstar_drn=TASSO_DIFFSTAR_DRN,
    lgt0=LGT0,
    n_times=200,
):
    diffmah_data, diffstar_data = load_diffstar_subvolume(
        subvol,
        n_subvol_tot=n_subvol_tot,
        diffmah_drn=diffmah_drn,
        diffstar_drn=diffstar_drn,
    )
    mah_params = DEFAULT_MAH_PARAMS._make(
        [diffmah_data[key] for key in DEFAULT_MAH_PARAMS._fields]
    )
    t_peak = diffmah_data["t_peak"]

    ms_params = DEFAULT_DIFFSTAR_PARAMS.ms_params._make(
        [diffstar_data[key] for key in DEFAULT_DIFFSTAR_PARAMS.ms_params._fields]
    )
    q_params = DEFAULT_DIFFSTAR_PARAMS.q_params._make(
        [diffstar_data[key] for key in DEFAULT_DIFFSTAR_PARAMS.q_params._fields]
    )
    sfh_params = DEFAULT_DIFFSTAR_PARAMS._make((ms_params, q_params))

    t_0 = 10**lgt0
    t_table = np.linspace(T_TABLE_MIN, t_0, n_times)

    __, log_mah_table = mah_halopop(mah_params, t_table, t_peak, LGT0)

    sfh_table, smh_table = calc_sfh_galpop(
        sfh_params, mah_params, t_peak, t_table, lgt0=LGT0, return_smh=True
    )
    log_sfh_table = np.log10(sfh_table)
    log_smh_table = np.log10(smh_table)
    log_ssfrh_table = log_sfh_table - log_smh_table

    return t_table, log_mah_table, log_smh_table, log_ssfrh_table


def compute_weighted_histograms_z0(
    subvol,
    n_subvol_tot=N_SUBVOL_SMDPL,
    diffmah_drn=TASSO_DIFFSTAR_DRN,
    diffstar_drn=TASSO_DIFFSTAR_DRN,
    lgt0=LGT0,
    logmh_bins=LOGMH_BINS,
):
    _res = load_diffstar_sfh_tables(
        subvol,
        n_subvol_tot=n_subvol_tot,
        diffmah_drn=diffmah_drn,
        diffstar_drn=diffstar_drn,
        lgt0=lgt0,
    )
    t_table, log_mah_table, log_smh_table, log_ssfrh_table = _res

    n_halos = log_smh_table.shape[0]

    nddata = log_mah_table[:, -1].reshape((-1, 1))

    sigma = np.mean(np.diff(logmh_bins)) + np.zeros(n_halos)
    ndsig = sigma.reshape((-1, 1))

    ydata = log_smh_table[:, -1].reshape((-1, 1))
    _ones = np.ones_like(ydata)

    ndbins_lo = logmh_bins[:-1].reshape((-1, 1))
    ndbins_hi = logmh_bins[1:].reshape((-1, 1))

    whist = tw_ndhist_weighted(nddata, ndsig, ydata, ndbins_lo, ndbins_hi)
    wcounts = tw_ndhist_weighted(nddata, ndsig, _ones, ndbins_lo, ndbins_hi)

    return wcounts, whist
