"""
"""

import os

import h5py
import numpy as np
from diffmah.diffmah_kernels import DEFAULT_MAH_PARAMS, mah_halopop
from diffsky.diffndhist import tw_ndhist_weighted
from diffstar.defaults import DEFAULT_DIFFSTAR_PARAMS, LGT0, T_TABLE_MIN
from diffstar.sfh_model_tpeak import calc_sfh_galpop
from scipy.stats import binned_statistic
from astropy.cosmology import Planck13

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


def compute_diff_histograms_atz(
    logmh_bins, 
    log_mah_table, 
    log_smh_table
):

    n_halos = log_smh_table.shape[0]

    nddata = log_mah_table.reshape((-1, 1))

    sigma = np.mean(np.diff(logmh_bins)) + np.zeros(n_halos)
    ndsig = sigma.reshape((-1, 1))

    ydata = log_smh_table.reshape((-1, 1))
    _ones = np.ones_like(ydata)

    ndbins_lo = logmh_bins[:-1].reshape((-1, 1))
    ndbins_hi = logmh_bins[1:].reshape((-1, 1))

    whist = tw_ndhist_weighted(nddata, ndsig, ydata, ndbins_lo, ndbins_hi)
    wcounts = tw_ndhist_weighted(nddata, ndsig, _ones, ndbins_lo, ndbins_hi)

    return wcounts, whist

def compute_histograms_atz(
    logmh_bins, 
    log_mah_table, 
    log_smh_table
):
    count = binned_statistic(log_mah_table, values=log_smh_table, bins=logmh_bins, statistic='count')[0]
    whist = binned_statistic(log_mah_table, values=log_smh_table, bins=logmh_bins, statistic='sum')[0]
    return count, whist
    

def get_redshift_from_age(age):
    z_table = np.linspace(0, 10, 2000)[::-1]
    t_table = Planck13.age(z_table).value
    redshift_from_age = np.interp(age, t_table, z_table)
    return redshift_from_age

def return_target_redshfit_index(t_table, redshift_targets):
    z_table = get_redshift_from_age(t_table)
    return np.digitize(redshift_targets, z_table)



def compute_weighted_histograms(
    subvol,
    redshift_targets,
    n_subvol_tot=N_SUBVOL_SMDPL,
    diffmah_drn=LCRC_DIFFMAH_DRN,
    diffstar_drn=LCRC_DIFFSTAR_DRN,
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

    tids = return_target_redshfit_index(t_table, redshift_targets)

    nz, nm = len(redshift_targets), len(logmh_bins)-1

    wcounts_zid = np.zeros((nz, nm))
    whist_zid = np.zeros((nz, nm))
    counts_zid = np.zeros((nz, nm))
    hist_zid = np.zeros((nz, nm))

    for i, tid in enumerate(tids):
        _res = compute_diff_histograms_atz(logmh_bins, log_mah_table[:,tid], log_smh_table[:,tid])
        wcounts_zid[i] = _res[0]
        whist_zid[i] = _res[1]
        
        _res = compute_histograms_atz(logmh_bins, log_mah_table[:,tid], log_smh_table[:,tid])
        counts_zid[i] = _res[0]
        hist_zid[i] = _res[1]
    
    return wcounts_zid, whist_zid, counts_zid, hist_zid, t_table[tids]




    