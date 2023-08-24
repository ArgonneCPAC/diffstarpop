import os
import h5py
import numpy as np
from diffstarpop.desc_colors_DSPS import calc_hist_1d

# fn = "/Users/alarcon/Downloads/hsc_i_n.dat"
fn = "/lcrc/project/halotools/alarcon/data/DESC_mocks_data/hsc_i_n.dat"
data = np.loadtxt(fn, skiprows=1)

# m, n(i < m) cumulative number counts (per unit area)
hsc_m, hsc_cumcounts_imag = data[:, 0], data[:, -1]

# n(i=m) galaxy number counts (per unit area)
hsc_ncounts_imag = np.diff(hsc_cumcounts_imag)

# Calculate target/differentiable n(i=m) where a galaxy's
# magnitude is a triweight pdf with a given widht.
hsc_mbins_centers = hsc_m[:-1] + np.diff(hsc_m) / 2.0
hsc_mbins_LO = hsc_m[:-1]
hsc_mbins_HI = hsc_m[1:]

# Right now width is 0.1 in magnitude
ndsig = np.ones_like(hsc_ncounts_imag) * 0.1

# Calculate the weighted contribution of real n(i=m) to target n(i=m)
hsc_ncounts_imag_tri = calc_hist_1d(
    hsc_mbins_centers, ndsig, hsc_ncounts_imag, hsc_mbins_LO, hsc_mbins_HI
)

# Target n(i < m) cumulative number counts
hsc_cumcounts_imag_tri = np.cumsum(hsc_ncounts_imag_tri)

# Comments:
# Since real n(i=m) increases as a power law and the
# triweight kernel is symmetric in m, the target n(i < m) is larger
# than the real n(i < m), except in the edges of the array.

# Mask target n(i < m) where there might be resoultion issues in the simulations.
mask = (hsc_m[1:] >= 20.0) & (hsc_m[1:] <= 24.0)

mag_i_bins = hsc_m[1:][mask]
hsc_cumcounts_imag_tri = hsc_cumcounts_imag_tri[mask]

outpath = "/lcrc/project/halotools/alarcon/data/DESC_mocks_data/"
out_name = "HSC_target_data_20_24_limits.h5"
with h5py.File(os.path.join(outpath, out_name), "w") as f:
    f["mag_i_bins"] = mag_i_bins
    f["hsc_cumcounts_imag_tri"] = hsc_cumcounts_imag_tri
