"""This module tabulates <logMstar | logMhalo, z=0> for SMDPL
"""

import argparse
import os
from time import time
import subprocess

import h5py
import numpy as np
import gc

import smdpl_smhm_utils as smhm_utils

from mpi4py import MPI


if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    rank, nranks = comm.Get_rank(), comm.Get_size()

    parser = argparse.ArgumentParser()
    parser.add_argument("-istart", help="start of subvolume loop", type=int, default=0)
    parser.add_argument(
        "-iend",
        help="end of subvolume loop",
        type=int,
        default=smhm_utils.N_SUBVOL_SMDPL,
    )
    parser.add_argument(
        "-n_subvol_max",
        help="Last subvolume",
        type=int,
        default=smhm_utils.N_SUBVOL_SMDPL,
    )
    parser.add_argument(
        "-diffmah_drn", help="input drn", type=str, default=smhm_utils.LCRC_DIFFMAH_DRN
    )
    parser.add_argument(
        "-diffstar_drn",
        help="input drn",
        type=str,
        default=smhm_utils.LCRC_DIFFSTAR_DRN,
    )

    parser.add_argument("-outdrn", help="output directory", type=str, default="")
    args = parser.parse_args()
    istart = args.istart
    iend = args.iend
    n_subvol_max = args.n_subvol_max
    outdrn = args.outdrn
    diffmah_drn = args.diffmah_drn
    diffstar_drn = args.diffstar_drn

    # redshift_targets = np.concatenate((np.arange(0,1,0.1), np.arange(1, 2.1, 0.5)))
    redshift_targets = smhm_utils.Z_BINS
    nz, nm = len(redshift_targets), smhm_utils.LOGMH_BINS.size - 1
    nmstar = len(smhm_utils.LOGMSTAR_BINS_PDF) - 1
    nssfr = len(smhm_utils.LOGSSFR_BINS_PDF) - 1

    subvol_used = np.zeros(n_subvol_max).astype(int)

    subvols = np.arange(istart, iend)
    subvols_arr = np.array_split(subvols, nranks)[rank]

    haloes_data = []
    print("Beginning loop over subvolumes...\n")

    for i in subvols_arr:
        gc.collect()
        try:
            start = time()
            _res = smhm_utils.create_target_data(
                i, redshift_targets, diffmah_drn=diffmah_drn, diffstar_drn=diffstar_drn
            )
            (
                wcounts_i,
                whist_i,
                counts_i,
                hist_i,
                age_targets,
                haloes,
                counts_cen_i,
                counts_sat_i,
            ) = _res

            _res = smhm_utils.create_pdf_target_data(
                i, redshift_targets, diffmah_drn=diffmah_drn, diffstar_drn=diffstar_drn
            )

            fnout = os.path.join(outdrn, "_tmp_subvol_%d_smdpl_smhm.h5" % i)
            with h5py.File(fnout, "w") as hdfout:
                hdfout["wcounts_i"] = wcounts_i
                hdfout["whist_i"] = whist_i
                hdfout["counts_i"] = counts_i
                hdfout["hist_i"] = hist_i
                hdfout["age_targets"] = age_targets
                hdfout["haloes"] = haloes
                hdfout["counts_cen_i"] = counts_cen_i
                hdfout["counts_sat_i"] = counts_sat_i
                hdfout["mstar_wcounts_i"] = _res[0]
                hdfout["mstar_counts_i"] = _res[1]
                hdfout["mstar_ssfr_wcounts_cent_i"] = _res[2]
                hdfout["mstar_ssfr_wcounts_sat_i"] = _res[3]

            end = time()
            runtime = end - start
            print(
                f"...computed sumstat counts for subvolume {i}",
                "Time: %.2f seconds." % (end - start),
            )
        except FileNotFoundError:
            print(f"...NO sumstat counts for subvolume {i}")
            pass

    comm.Barrier()

    if rank == 0:
        print("Collecting all data in rank 0.")
        start = time()

        wcounts = np.zeros((nz, nm))
        whist = np.zeros_like(wcounts)
        counts = np.zeros_like(wcounts)
        hist = np.zeros_like(wcounts)
        counts_cen = np.zeros_like(wcounts)
        counts_sat = np.zeros_like(wcounts)

        mstar_wcounts = np.zeros((nz, nm, nmstar))
        mstar_counts = np.zeros((nz, nm, nmstar))

        mstar_ssfr_wcounts_cent = np.zeros((nz, nm, nmstar, nssfr))
        mstar_ssfr_wcounts_sat = np.zeros((nz, nm, nmstar, nssfr))

        for i in subvols:
            fnout = os.path.join(outdrn, "_tmp_subvol_%d_smdpl_smhm.h5" % i)
            with h5py.File(fnout, "r") as hdfout:
                wcounts = wcounts + hdfout["wcounts_i"][:]
                whist = whist + hdfout["whist_i"][:]
                counts = counts + hdfout["counts_i"][:]
                hist = hist + hdfout["hist_i"][:]
                counts_cen = counts_cen + hdfout["counts_cen_i"][:]
                counts_sat = counts_sat + hdfout["counts_sat_i"][:]
                subvol_used[i] = 1
                haloes_data.append(hdfout["haloes"][:])
                mstar_wcounts += hdfout["mstar_wcounts_i"][:]
                mstar_counts += hdfout["mstar_counts_i"][:]
                mstar_ssfr_wcounts_cent += hdfout["mstar_ssfr_wcounts_cent_i"][:]
                mstar_ssfr_wcounts_sat += hdfout["mstar_ssfr_wcounts_sat_i"][:]

        sampled_haloes = smhm_utils.concatenate_samples_haloes(haloes_data)
        end = time()
        runtime = end - start
        print("Ended collecting data. Time: %.2f seconds." % (end - start))

        print("Saving final target data...")

        fnout = os.path.join(outdrn, "smdpl_smhm.h5")
        with h5py.File(fnout, "w") as hdfout:
            hdfout["counts_diff"] = wcounts
            hdfout["hist_diff"] = whist
            hdfout["counts"] = counts
            hdfout["hist"] = hist
            hdfout["counts_cen"] = counts_cen
            hdfout["counts_sat"] = counts_sat
            hdfout["smhm_diff"] = whist / wcounts
            hdfout["smhm"] = hist / counts
            hdfout["logmh_bins"] = smhm_utils.LOGMH_BINS
            hdfout["subvol_used"] = subvol_used
            hdfout["redshift_targets"] = redshift_targets
            hdfout["age_targets"] = age_targets

        (
            logmh_id,
            logmh_val,
            mah_params_samp,
            ms_params_samp,
            q_params_samp,
            upid_samp,
            tobs_id,
            tobs_val,
            redshift_val,
        ) = sampled_haloes

        fnout = os.path.join(outdrn, "smdpl_smhm_samples_haloes.h5")
        with h5py.File(fnout, "w") as hdfout:
            hdfout["logmh_id"] = logmh_id
            hdfout["logmh_val"] = logmh_val
            hdfout["mah_params_samp"] = mah_params_samp
            hdfout["ms_params_samp"] = ms_params_samp
            hdfout["q_params_samp"] = q_params_samp
            hdfout["upid_samp"] = upid_samp
            hdfout["tobs_id"] = tobs_id
            hdfout["tobs_val"] = tobs_val
            hdfout["redshift_val"] = redshift_val

        fnout = os.path.join(outdrn, "smdpl_mstar_ssfr.h5")
        with h5py.File(fnout, "w") as hdfout:
            hdfout["mstar_wcounts"] = mstar_wcounts
            hdfout["mstar_counts"] = mstar_counts
            hdfout["mstar_ssfr_wcounts_cent"] = mstar_ssfr_wcounts_cent
            hdfout["mstar_ssfr_wcounts_sat"] = mstar_ssfr_wcounts_sat
            hdfout["logmh_bins"] = smhm_utils.LOGMH_BINS
            hdfout["logmstar_bins_pdf"] = smhm_utils.LOGMSTAR_BINS_PDF
            hdfout["logssfr_bins_pdf"] = smhm_utils.LOGSSFR_BINS_PDF
            hdfout["redshift_targets"] = redshift_targets
            hdfout["age_targets"] = age_targets

        n_used = subvol_used.sum()

        # clean up all temporary data
        bnpat = os.path.join(outdrn, "_tmp_subvol_*")
        command = "rm " + bnpat
        subprocess.os.system(command)
