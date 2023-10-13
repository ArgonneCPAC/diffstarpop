import numpy as np
import pandas as pd
from umachine_pyio.load_mock import load_mock_from_binaries
from halotools.utils import crossmatch
from collections import OrderedDict
import h5py
import time
import sys 
import os 

root_dir = "/lcrc/project/galsampler/SMDPL/dr1_no_merging_upidh/"
lighcone_path = root_dir+"lightcones/"
smdpl_binaries_path = root_dir+"sfh_binary_catalogs/a_1.000000/"
diffmah_path = root_dir+"sfh_binary_catalogs/diffmah_fits/subvol_%d/diffmah_fits.h5"
diffstar_path = root_dir+"sfh_binary_catalogs/diffstar_fits/subvol_%d/diffstar_fits.h5"
save_dir = "/lcrc/project/halotools/alarcon/data/"
sfr_binaries_path = "/lcrc/project/galsampler/SMDPL/dr1_no_merging_upidh/sfr_binaries/"

save_dir = "/lcrc/project/halotools/alarcon/data/"

lightcone = np.load(lighcone_path + "survey_z0.02-2.00_x60.00_y60.00_0.npy")

lightcone_path = "/lcrc/project/galsampler/SMDPL/DR1/lightcones/"

for light_id in range(10):
    fn = f"sfh_crossmatch/survey_z0.02-2.00_x60.00_y60.00_{light_id}.sfh_crossmatched.h5"
    with h5py.File(os.path.join(lightcone_path, fn), "r") as f:
        halo_ids = f["halo_id"][...]
        mps = 10**f["logmp"][...]

    # df = pd.DataFrame(lightcone)
    df = pd.DataFrame({'id':halo_ids, 'mp':mps})

    df_indx = pd.DataFrame()
    df_diff_fits = pd.DataFrame()

    count = 0

    # for subvol in range(0, 576):
    for subvol in range(0, 55):
    # for subvol in test.binary_subvol.unique():
    # subvol = 205
        t0 = time.time()
        galprops = ["halo_id", "halo_id_history", "mpeak_history_main_prog"]
        halos = load_mock_from_binaries(
            np.array([subvol]), root_dirname=smdpl_binaries_path, galprops=galprops
        )
        halo_id = np.array(halos["halo_id"])
        halo_id_history = np.array(halos["halo_id_history"])
        mpeak_history_main_prog = np.array(halos["mpeak_history_main_prog"])
        ng, nt = halo_id_history.shape
        halo_id_history_flat = halo_id_history.flatten()
        _mask = halo_id_history_flat == 0
        halo_id_history_flat[_mask] = -np.arange(2, _mask.sum() + 2, 1).astype(int)
        idx_x, idx_y = crossmatch(df.id.values, halo_id_history_flat)

        idx_halo_id, idx_snapshot = np.unravel_index(idx_y, shape=(ng, nt))
        count += len(idx_x)
        # assert False
        if len(idx_halo_id) == 0:
            t1 = time.time()
            print(
                "subvol: %d," % subvol,
                "Time:  %ds" % (t1 - t0),
            )
            continue

        # Check this is correct
        assert np.allclose(
            df.id.values[idx_x],
            np.array([halo_id_history[x, y] for (x, y) in zip(idx_halo_id, idx_snapshot)]),
        )
        assert np.allclose(
            df.mp.values[idx_x],
            np.array([mpeak_history_main_prog[x, y] for (x, y) in zip(idx_halo_id, idx_snapshot)]),
            rtol=1e-03 # the lightcone mp values seem to be rounded to 4 signinficant figures
        )

        # assert False
        df_sub = pd.DataFrame(
            {
                "id": df.id.values[idx_x],
                "halo_id": halo_id_history[idx_halo_id, -1],
                "binary_index": idx_halo_id,
                "binary_subvol": subvol,
            }
        )
        df_indx = pd.concat([df_indx, df_sub], ignore_index=True)
        # """
        # print(count)
        # print(len(df_indx))
        diffmah_params = OrderedDict()
        fn = diffmah_path % subvol
        with h5py.File(fn, "r") as hdf:
        # print(hdf.keys())
        # print(len(hdf["halo_id"]))
            for key in hdf.keys():
                diffmah_params[key] = hdf[key][...]
        """
        diffstar_params = OrderedDict()
        fn = diffstar_path % subvol
        with h5py.File(fn, "r") as hdf:
        # print(hdf.keys())
            for key in hdf.keys():
                diffstar_params[key] = hdf[key][...]

        df_sub = df_sub.merge(pd.DataFrame(diffmah_params), on="halo_id", how="left")
        df_sub = df_sub.merge(pd.DataFrame(diffstar_params), on="halo_id", how="left")

        df_diff_fits = pd.concat([df_diff_fits, df_sub], ignore_index=True)
        # df = df.merge(df_sub, on="id", how="left")
        """
        t1 = time.time()
        assert False
        print(
            
            "subvol: %d, %d" % (subvol, light_id),
            "Found: %.2f %%," % (100 * (len(np.unique(df_indx.halo_id)) / len(df))),
            # "Found: %.2f %%," % (100 * (len(np.unique(df_diff_fits.halo_id)) / len(df))),
            "Time:  %ds" % (t1 - t0),
        )

    fn = save_dir+f"survey_z0.02-2.00_x60.00_y60.00_{light_id}.haloid_indices_v1.h5"
    with h5py.File(fn, "w") as hdf:
        hdf["halo_id_at_snapshot"] = df_indx.id.values
        hdf["halo_id_at_z0"] = df_indx.halo_id.values
        hdf["binary_snapshot_index"] = df_indx.binary_index.values
        hdf["binary_subvol_index"] = df_indx.binary_subvol.values

assert False
# df = df.merge(df_sub, on="id", how="left")

np.save(save_dir+"survey_z0.02-2.00_x60.00_y60.00_10lightcones_matched_haloid_indices.npy", df_indx.to_records(index=False))
np.save(save_dir+"survey_z0.02-2.00_x60.00_y60.00_10lightcones_matched_allinfo.npy", df_diff_fits.to_records(index=False))

# np.save(save_dir+"survey_z0.02-2.00_x60.00_y60.00_0_matched_haloid_indices.npy", df_indx.to_records(index=False))
# np.save(save_dir+"survey_z0.02-2.00_x60.00_y60.00_0_matched_allinfo.npy", df_diff_fits.to_records(index=False))

# nz_matched = np.histogram(df_indx[['id']].merge(df[['z(cosmo)','id']], on="id", how="left")['z(cosmo)'], np.arange(0,2,0.1))[0]
# nz_all = np.histogram(df['z(cosmo)'], np.arange(0,2,0.1))[0]
# nz_matched / nz_all

