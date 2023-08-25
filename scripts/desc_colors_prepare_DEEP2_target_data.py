import os
import h5py
import numpy as np
from scipy.integrate import quad

# fn = "/Users/alarcon/Downloads/Coil_et_al_2004_Table4_i.txt"
fn = "/lcrc/project/halotools/alarcon/data/DESC_mocks_data/Coil_et_al_2004_Table4_i.txt"
table4_i = np.loadtxt(fn, skiprows=3)

# fn = "/Users/alarcon/Downloads/Coil_et_al_2004_Table4_r.txt"
fn = "/lcrc/project/halotools/alarcon/data/DESC_mocks_data/Coil_et_al_2004_Table4_r.txt"
table4_r = np.loadtxt(fn, skiprows=3)


def dndz_optionA(z, z0):
    """Option a in Coil 2004 Table 4."""

    def _fun(z, z0):
        return (z**2) * np.exp(-z / z0)

    norm = quad(_fun, 0.0, np.inf, args=(z0))[0]

    return (1.0 / norm) * _fun(z, z0)


def dndz_optionB(z, z0):
    """Option b in Coil 2004 Table 4."""

    def _fun(z, z0):
        return (z**2) * (np.exp(-z / z0) ** 1.2)

    norm = quad(_fun, 0.0, np.inf, args=(z0))[0]

    return (1.0 / norm) * _fun(z, z0)


def calculate_dNdZ_target_data(func, bins, z0_list):
    target_data = np.zeros((len(z0_list), len(bins) - 1))
    for i, z0 in enumerate(z0_list):
        target_data[i] = np.array(
            [
                quad(func, bins[j], bins[j + 1], args=(z0))[0]
                for j in range(len(bins) - 1)
            ]
        )
    return target_data


bin_edges = np.linspace(0.0, 1.5, 16)
target_dNdz_rmag = calculate_dNdZ_target_data(dndz_optionA, bin_edges, table4_r[:4, 2])
target_dNdz_imag = calculate_dNdZ_target_data(dndz_optionA, bin_edges, table4_i[:4, 2])

outpath = "/lcrc/project/halotools/alarcon/data/DESC_mocks_data/"
out_name = "DEEP2_target_data.h5"
with h5py.File(os.path.join(outpath, out_name), "w") as f:
    f["bin_edges"] = bin_edges
    f["target_dNdz_rmag"] = target_dNdz_rmag
    f["target_dNdz_imag"] = target_dNdz_imag
