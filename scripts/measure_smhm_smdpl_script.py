"""This module tabulates <logMstar | logMhalo, z=0> for SMDPL
"""

import argparse
import os
from time import time

import h5py
import numpy as np

import smdpl_smhm_utils as smhm_utils

if __name__ == "__main__":

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

    redshift_targets = np.concatenate((np.arange(0,1,0.1), np.arange(1, 2.1, 0.5)))

    nz, nm = len(redshift_targets), smhm_utils.LOGMH_BINS.size - 1

    wcounts = np.zeros(nz, nm)
    whist = np.zeros_like(wcounts)
    counts = np.zeros_like(wcounts)
    hist = np.zeros_like(wcounts)

    subvol_used = np.zeros(n_subvol_max).astype(int)
    haloes_data = []
    print("Beginning loop over subvolumes...\n")
    start = time()
    for i in range(istart, iend):
        try:
            _res = smhm_utils.create_target_data(
                i, 
                redshift_targets, 
                diffmah_drn=diffmah_drn, 
                diffstar_drn=diffstar_drn
            )
            wcounts_i, whist_i, counts_i, hist_i, age_targets, haloes = _res

            wcounts = wcounts + wcounts_i
            whist = whist + whist_i
            counts = counts + counts_i
            hist = hist + hist_i
            subvol_used[i] = 1
            haloes_data.append(haloes)
            print(f"...computed sumstat counts for subvolume {i}")
        except FileNotFoundError:
            print(f"...NO sumstat counts for subvolume {i}")
            pass

    sampled_haloes = smhm_utils.concatenate_samples_haloes(haloes_data)
    end = time()
    runtime = end - start

    fnout = os.path.join(outdrn, "smdpl_smhm.h5")
    with h5py.File(fnout, "w") as hdfout:
        hdfout["counts_diff"] = wcounts
        hdfout["hist_diff"] = whist
        hdfout["counts"] = counts
        hdfout["hist"] = hist
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
        hdfout["tobs_id"] = tobs_id
        hdfout["tobs_val"] = tobs_val
        hdfout["redshift_val"] = redshift_val

        
    n_used = subvol_used.sum()
    print(f"Total runtime to loop over {n_used} subvolumes = {runtime:.1f} seconds")
