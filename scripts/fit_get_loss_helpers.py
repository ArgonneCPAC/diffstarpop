"""
"""

import h5py
import numpy as np
from diffmah.diffmah_kernels import DiffmahParams, mah_halopop
from diffstar.defaults import LGT0
from jax import random as jran


def get_loss_data_smhm(indir, nhalos):
    # Load SMHM data ---------------------------------------------
    print("Loading SMHM data...")

    with h5py.File(indir + "smdpl_smhm.h5", "r") as hdf:
        redshift_targets = hdf["redshift_targets"][:]
        # smhm_diff = hdf["smhm_diff"][:]
        smhm = hdf["smhm"][:]
        logmh_bins = hdf["logmh_bins"][:]
        age_targets = hdf["age_targets"][:]
        """
            hdfout["counts_diff"] = wcounts
            hdfout["hist_diff"] = whist
            hdfout["counts"] = counts
            hdfout["hist"] = hist
            hdfout["smhm_diff"] = whist / wcounts
            hdfout["smhm"] = hist / counts
            hdfout["logmh_bins"] = smhm_utils.LOGMH_BINS
            hdfout["subvol_used"] = subvol_used
        """

    logmh_binsc = 0.5 * (logmh_bins[1:] + logmh_bins[:-1])

    with h5py.File(indir + "smdpl_smhm_samples_haloes.h5", "r") as hdf:
        logmh_id = hdf["logmh_id"][:]
        # logmh_val = hdf["logmh_id"][:]
        mah_params_samp = hdf["mah_params_samp"][:]
        # ms_params_samp = hdf["ms_params_samp"][:]
        # q_params_samp = hdf["q_params_samp"][:]
        t_peak_samp = hdf["t_peak_samp"][:]
        tobs_id = hdf["tobs_id"][:]
        # tobs_val = hdf["tobs_val"][:]
        # redshift_val = hdf["redshift_val"][:]

    mah_params_samp = np.concatenate((mah_params_samp, t_peak_samp[None, :]), axis=0)

    # Create loss_data ---------------------------------------------
    print("Creating loss data...")

    ran_key = jran.PRNGKey(np.random.randint(2**32))

    lgmu_infall = -1.0
    logmhost_infall = 13.0
    gyr_since_infall = 2.0

    mah_params_data = []
    lomg0_data = []
    lgmu_infall_data = []
    logmhost_infall_data = []
    gyr_since_infall_data = []
    t_obs_targets = []
    smhm_targets = []

    tarr_logm0 = np.logspace(-1, LGT0, 50)

    for i in range(len(age_targets)):
        t_target = age_targets[i]

        for j in range(len(logmh_binsc)):
            sel = (tobs_id == i) & (logmh_id == j)

            if sel.sum() < nhalos:
                continue
            arange_sel = np.arange(len(tobs_id))[sel]
            arange_sel = np.random.choice(arange_sel, nhalos, replace=False)
            mah_params_data.append(mah_params_samp[:, arange_sel])
            lgmu_infall_data.append(np.ones(len(arange_sel)) * lgmu_infall)
            logmhost_infall_data.append(np.ones(len(arange_sel)) * logmhost_infall)
            gyr_since_infall_data.append(np.ones(len(arange_sel)) * gyr_since_infall)
            t_obs_targets.append(t_target)
            smhm_targets.append(smhm[i, j])
            mah_pars_ntuple = DiffmahParams(*mah_params_samp[:, arange_sel])
            dmhdt_fit, log_mah_fit = mah_halopop(mah_pars_ntuple, tarr_logm0, LGT0)
            lomg0_data.append(log_mah_fit[:, -1])

    mah_params_data = np.array(mah_params_data)
    lomg0_data = np.array(lomg0_data)
    lgmu_infall_data = np.array(lgmu_infall_data)
    logmhost_infall_data = np.array(logmhost_infall_data)
    gyr_since_infall_data = np.array(gyr_since_infall_data)
    t_obs_targets = np.array(t_obs_targets)
    smhm_targets = np.array(smhm_targets)

    ran_key_data = jran.split(ran_key, len(smhm_targets))
    loss_data = (
        mah_params_data,
        lomg0_data,
        lgmu_infall_data,
        logmhost_infall_data,
        gyr_since_infall_data,
        ran_key_data,
        t_obs_targets,
        smhm_targets,
    )

    plot_data = (
        age_targets,
        logmh_binsc,
        tobs_id,
        logmh_id,
        tarr_logm0,
        lgmu_infall,
        logmhost_infall,
        gyr_since_infall,
        ran_key,
        redshift_targets,
        smhm,
        mah_params_samp,
    )

    return loss_data, plot_data


def get_loss_data_pdfs_mstar(indir, nhalos):
    # Load SMHM data ---------------------------------------------
    print("Loading SMHM data...")

    tpeak_path = "/Users/alarcon/Documents/diffmah_data/tpeak/random_data_241008/"
    with h5py.File(tpeak_path + "smdpl_smhm.h5", "r") as hdf:
        counts_cen = hdf["counts_cen"][:]
        counts_sat = hdf["counts_sat"][:]

    tpeak_path = "/Users/alarcon/Documents/diffmah_data/tpeak/"

    with h5py.File(tpeak_path + "smdpl_smhm.h5", "r") as hdf:
        redshift_targets = hdf["redshift_targets"][:]
        smhm_diff = hdf["smhm_diff"][:]
        smhm = hdf["smhm"][:]
        logmh_bins = hdf["logmh_bins"][:]
        age_targets = hdf["age_targets"][:]
        hist = hdf["hist"][:]
        counts = hdf["counts"][:]
        # counts_cen = hdf["counts_cen"][:]
        # counts_sat = hdf["counts_sat"][:]

    logmh_binsc = 0.5 * (logmh_bins[1:] + logmh_bins[:-1])

    tpeak_path = "/Users/alarcon/Documents/diffmah_data/tpeak/random_data_241007/"
    with h5py.File(tpeak_path + "smdpl_smhm_samples_haloes.h5", "r") as hdf:
        logmh_id = hdf["logmh_id"][:]
        logmh_val = hdf["logmh_id"][:]
        mah_params_samp = hdf["mah_params_samp"][:]
        ms_params_samp = hdf["ms_params_samp"][:]
        q_params_samp = hdf["q_params_samp"][:]
        t_peak_samp = hdf["t_peak_samp"][:]
        upid_samp = hdf["upid_samp"][:]
        tobs_id = hdf["tobs_id"][:]
        tobs_val = hdf["tobs_val"][:]
        redshift_val = hdf["redshift_val"][:]

    tpeak_path = "/Users/alarcon/Documents/diffmah_data/tpeak/random_data_241007/"
    with h5py.File(tpeak_path + "smdpl_mstar_ssfr.h5", "r") as hdf:
        mstar_wcounts = hdf["mstar_wcounts"][:]
        mstar_counts = hdf["mstar_counts"][:]
        mstar_ssfr_wcounts_cent = hdf["mstar_ssfr_wcounts_cent"][:]
        mstar_ssfr_wcounts_sat = hdf["mstar_ssfr_wcounts_sat"][:]
        logssfr_bins_pdf = hdf["logssfr_bins_pdf"][:]
        logmstar_bins_pdf = hdf["logmstar_bins_pdf"][:]
        """
        hdfout["mstar_wcounts"] = mstar_wcounts
        hdfout["mstar_counts"] = mstar_counts
        hdfout["mstar_ssfr_wcounts_cent"] = mstar_ssfr_wcounts_cent
        hdfout["mstar_ssfr_wcounts_sat"] = mstar_ssfr_wcounts_sat
        hdfout["logmh_bins"] = smhm_utils.LOGMH_BINS
        hdfout["logmstar_bins_pdf"] = smhm_utils.LOGMSTAR_BINS_PDF
        hdfout["logssfr_bins_pdf"] = smhm_utils.LOGSSFR_BINS_PDF
        hdfout["redshift_targets"] = redshift_targets
        hdfout["age_targets"] = age_targets
        """

    logssfr_binsc_pdf = 0.5 * (logssfr_bins_pdf[1:] + logssfr_bins_pdf[:-1])
    logmstar_binsc_pdf = 0.5 * (logmstar_bins_pdf[1:] + logmstar_bins_pdf[:-1])

    mah_params_samp = np.concatenate((mah_params_samp, t_peak_samp[None, :]), axis=0)

    # Create loss_data ---------------------------------------------
    print("Creating loss data...")

    ran_key = jran.PRNGKey(np.random.randint(2**32))

    lgmu_infall = -1.0
    logmhost_infall = 13.0
    gyr_since_infall = 2.0

    mah_params_data = []
    lomg0_data = []
    lgmu_infall_data = []
    logmhost_infall_data = []
    gyr_since_infall_data = []
    t_obs_targets = []
    mstar_counts_target = []

    tarr_logm0 = np.logspace(-1, LGT0, 50)

    for i in range(len(age_targets)):
        t_target = age_targets[i]

        for j in range(len(logmh_binsc)):
            sel = (tobs_id == i) & (logmh_id == j)

            if sel.sum() < 50:
                continue
            replace = True if sel.sum() < nhalos else False
            arange_sel = np.arange(len(tobs_id))[sel]
            arange_sel = np.random.choice(arange_sel, nhalos, replace=replace)
            mah_params_data.append(mah_params_samp[:, arange_sel])
            lgmu_infall_data.append(np.ones(len(arange_sel)) * lgmu_infall)
            logmhost_infall_data.append(np.ones(len(arange_sel)) * logmhost_infall)
            gyr_since_infall_data.append(np.ones(len(arange_sel)) * gyr_since_infall)
            t_obs_targets.append(t_target)
            mstar_counts_target.append(mstar_wcounts[i, j] / mstar_wcounts[i, j].sum())
            mah_pars_ntuple = DiffmahParams(*mah_params_samp[:, arange_sel])
            dmhdt_fit, log_mah_fit = mah_halopop(mah_pars_ntuple, tarr_logm0, LGT0)
            lomg0_data.append(log_mah_fit[:, -1])
        # break

    mah_params_data = np.array(mah_params_data)
    lomg0_data = np.array(lomg0_data)
    lgmu_infall_data = np.array(lgmu_infall_data)
    logmhost_infall_data = np.array(logmhost_infall_data)
    gyr_since_infall_data = np.array(gyr_since_infall_data)
    t_obs_targets = np.array(t_obs_targets)
    mstar_counts_target = np.array(mstar_counts_target)

    ran_key_data = jran.split(ran_key, len(mstar_counts_target))
    loss_data_mstar = (
        mah_params_data,
        lomg0_data,
        lgmu_infall_data,
        logmhost_infall_data,
        gyr_since_infall_data,
        ran_key_data,
        t_obs_targets,
        logmstar_bins_pdf,
        mstar_counts_target,
    )

    # Create loss_data for plot ---------------------------------------------
    print("Creating loss data for plot...")
    nhalos_plot = 10000
    ran_key = jran.PRNGKey(np.random.randint(2**32))

    lgmu_infall = -1.0
    logmhost_infall = 13.0
    gyr_since_infall = 2.0

    mah_params_data = []
    lomg0_data = []
    lgmu_infall_data = []
    logmhost_infall_data = []
    gyr_since_infall_data = []
    t_obs_targets = []
    mstar_counts_target = []

    tarr_logm0 = np.logspace(-1, LGT0, 50)

    for i in range(len(age_targets)):
        t_target = age_targets[i]

        for j in range(len(logmh_binsc)):
            sel = (tobs_id == i) & (logmh_id == j)

            if sel.sum() < 50:
                continue
            replace = True if sel.sum() < nhalos_plot else False
            arange_sel = np.arange(len(tobs_id))[sel]
            arange_sel = np.random.choice(arange_sel, nhalos_plot, replace=replace)
            mah_params_data.append(mah_params_samp[:, arange_sel])
            lgmu_infall_data.append(np.ones(len(arange_sel)) * lgmu_infall)
            logmhost_infall_data.append(np.ones(len(arange_sel)) * logmhost_infall)
            gyr_since_infall_data.append(np.ones(len(arange_sel)) * gyr_since_infall)
            t_obs_targets.append(t_target)
            mstar_counts_target.append(mstar_wcounts[i, j] / mstar_wcounts[i, j].sum())
            mah_pars_ntuple = DiffmahParams(*mah_params_samp[:, arange_sel])
            dmhdt_fit, log_mah_fit = mah_halopop(mah_pars_ntuple, tarr_logm0, LGT0)
            lomg0_data.append(log_mah_fit[:, -1])
        # break

    mah_params_data = np.array(mah_params_data)
    lomg0_data = np.array(lomg0_data)
    lgmu_infall_data = np.array(lgmu_infall_data)
    logmhost_infall_data = np.array(logmhost_infall_data)
    gyr_since_infall_data = np.array(gyr_since_infall_data)
    t_obs_targets = np.array(t_obs_targets)
    mstar_counts_target = np.array(mstar_counts_target)

    ran_key_data = jran.split(ran_key, len(mstar_counts_target))
    loss_data_mstar_pred = (
        mah_params_data,
        lomg0_data,
        lgmu_infall_data,
        logmhost_infall_data,
        gyr_since_infall_data,
        ran_key_data,
        t_obs_targets,
        logmstar_bins_pdf,
        mstar_counts_target,
    )
    plot_data = (
        logmstar_bins_pdf,
        mstar_wcounts,
        age_targets,
        redshift_targets,
        tobs_id,
        logmh_id,
        logmh_binsc,
        loss_data_mstar_pred,
    )

    return loss_data_mstar, plot_data
