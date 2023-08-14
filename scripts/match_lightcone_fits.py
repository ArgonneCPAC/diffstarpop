import numpy as np
import pandas as pd
from umachine_pyio.load_mock import load_mock_from_binaries
from halotools.utils import crossmatch
from collections import OrderedDict
import h5py
import time

lighcone_path = "/lcrc/project/galsampler/SMDPL/dr1_no_merging_upidh/lightcones/"
smdpl_binaries_path = "/lcrc/project/galsampler/SMDPL/dr1_no_merging_upidh/sfh_binary_catalogs/a_1.000000/"
diffmah_path = "/lcrc/project/galsampler/SMDPL/dr1_no_merging_upidh/sfh_binary_catalogs/diffmah_fits/subvol_%d/diffmah_fits.h5"
diffstar_path = "/lcrc/project/galsampler/SMDPL/dr1_no_merging_upidh/sfh_binary_catalogs/diffstar_fits/subvol_%d/diffstar_fits.h5"

lightcone = np.load(lighcone_path + "survey_z0.02-2.00_x60.00_y60.00_0.npy")

df = pd.DataFrame(lightcone)
df_indx = pd.DataFrame()

for subvol in range(0, 576):
    subvol = 205
    t0 = time.time()
    galprops = ["halo_id_history"]
    halos = load_mock_from_binaries(
        np.array([subvol]), root_dirname=smdpl_binaries_path, galprops=galprops
    )
    halo_id_history = np.array(halos["halo_id_history"])
    ng, nt = halo_id_history.shape
    halo_id_history_flat = halo_id_history.flatten()
    _mask = halo_id_history_flat == 0
    halo_id_history_flat[_mask] = -np.arange(2, _mask.sum() + 2, 1).astype(int)
    idx_x, idx_y = crossmatch(df.id.values, halo_id_history_flat)

    idx_halo_id, idx_snapshot = np.unravel_index(idx_y, shape=(ng, nt))

    if len(idx_halo_id) == 0:
        t1 = time.time()
        print(
            "subvol: %d," % subvol,
            "Found: %.2f %%,"
            % (100 * (np.isfinite(df.halo_id.values).sum() / len(df))),
            "Time:  %ds" % (t1 - t0),
        )
        continue

    # Check this is correct
    assert np.allclose(
        df.id.values[idx_x],
        np.array([halo_id_history[x, y] for (x, y) in zip(idx_halo_id, idx_snapshot)]),
    )

    df_sub = pd.DataFrame(
        {
            "id": df.id.values[idx_x],
            "halo_id": halo_id_history[idx_halo_id, -1],
            "binary_index": idx_halo_id,
            "binary_subvol": subvol,
        }
    )

    df_indx = df_indx.append(df_sub)

    diffmah_params = OrderedDict()
    fn = diffmah_path % subvol
    with h5py.File(fn, "r") as hdf:
        print(hdf.keys())
        print(len(hdf["halo_id"]))
        for key in hdf.keys():
            diffmah_params[key] = hdf[key][...]

    diffstar_params = OrderedDict()
    fn = diffstar_path % subvol
    with h5py.File(fn, "r") as hdf:
        print(hdf.keys())
        for key in hdf.keys():
            diffstar_params[key] = hdf[key][...]

    df_sub = df_sub.merge(pd.DataFrame(diffmah_params), on="halo_id", how="left")
    df_sub = df_sub.merge(pd.DataFrame(diffstar_params), on="halo_id", how="left")

    df = df.merge(df_sub, on="id", how="left")
    t1 = time.time()
    print(
        "subvol: %d," % subvol,
        "Found: %.2f %%," % (100 * (np.isfinite(df.halo_id.values).sum() / len(df))),
        "Time:  %ds" % (t1 - t0),
    )
