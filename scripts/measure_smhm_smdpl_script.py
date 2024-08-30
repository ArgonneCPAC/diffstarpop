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

    wcounts = np.zeros(smhm_utils.LOGMH_BINS.size - 1)
    whist = np.zeros_like(wcounts)

    subvol_used = np.zeros(n_subvol_max).astype(int)

    print("Beginning loop over subvolumes...\n")
    start = time()
    for i in range(istart, iend):
        try:
            wcounts_i, whist_i = smhm_utils.compute_weighted_histograms_z0(
                i, diffmah_drn=diffmah_drn, diffstar_drn=diffstar_drn
            )
            wcounts = wcounts + wcounts_i
            whist = whist + whist_i
            subvol_used[i] = 1
            print(f"...computed sumstat counts for subvolume {i}")
        except FileNotFoundError:
            print(f"...NO sumstat counts for subvolume {i}")
            pass
    end = time()
    runtime = end - start

    fnout = os.path.join(outdrn, "smdpl_smhm_z0.h5")
    with h5py.File(fnout, "w") as hdfout:
        hdfout["wcounts"] = wcounts
        hdfout["whist"] = whist
        hdfout["smhm"] = whist / wcounts
        hdfout["logmh_bins"] = smhm_utils.LOGMH_BINS
        hdfout["subvol_used"] = subvol_used

    n_used = subvol_used.sum()
    print(f"Total runtime to loop over {n_used} subvolumes = {runtime:.1f} seconds")
