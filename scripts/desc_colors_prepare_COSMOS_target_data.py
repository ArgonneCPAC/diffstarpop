import numpy as np
from jax import numpy as jnp
from astropy.io import fits
from scipy.interpolate import interp1d
import pandas as pd
from diffstarpop.desc_colors_DSPS import calculate_1d_COSMOS_colors_counts_singlez_bin


def flux2mag(flux):
    """COSMOS2020 flux to magnitude conversion."""
    return -2.5 * np.log10(flux * 1e-6) + 8.9


def flux2lupmag(flux, k=1e8):
    """COSMOS2020 flux to arcsinh magnitude conversion. Identical up to mag~29."""
    val = flux * 1e-6
    val = (1 / jnp.log(10)) * jnp.arcsinh(val / 2 * k) - jnp.log10(k)
    return -2.5 * val + 8.9


# Load the 2020 FARMER data
# fn = "/Users/alarcon/Documents/data/COSMOS2020/COSMOS2020_FARMER_R1_v2.2_p3.fits"
fn = "/lcrc/project/halotools/alarcon/data/DESC_mocks_data/COSMOS2020_FARMER_R1_v2.2_p3.fits"
data = fits.open(fn)[1].data

# filter_path = "/Users/alarcon/Documents/DSPS_data/filters/cosmos/"
filter_path = "/lcrc/project/halotools/alarcon/data/DSPS_data/filters/cosmos/"

filter_filenames = [
    # "CFHT_MegaCam.u.dat",
    "u_cfht.res",
    "g_HSC.txt",
    "r_HSC.txt",
    "i_HSC.txt",
    "z_HSC.txt",
    "Y_uv.res",
    "J_uv.res",
    "H_uv.res",
    "K_uv.res",
]

bands = [
    # "CFHT_u",
    "CFHT_ustar",
    "HSC_g",
    "HSC_r",
    "HSC_i",
    "HSC_z",
    "UVISTA_Y",
    "UVISTA_J",
    "UVISTA_H",
    "UVISTA_Ks",
]
print("Calculating MW extinction correction")

# Load MW extinction law from Lephare code.
# fn = "/Users/alarcon/Downloads/lephare_dev/ext/MW_seaton.dat"
fn = "/lcrc/project/halotools/alarcon/data/DESC_mocks_data/MW_seaton.dat"
MW_seaton = np.loadtxt(fn).T
MW_seaton_func = interp1d(
    MW_seaton[0],
    MW_seaton[1],
    bounds_error=False,
    fill_value=(MW_seaton[1][0], MW_seaton[1][-1]),
)


def calculate_F(filter_path):
    """Calculate the average extinction of one filter.
    F = \int dλ filter(λ) * extinction_law(λ)

    then: mag_unobs = mag_obs - 0.4 * E(B-V) * F
    """
    filter_data = np.loadtxt(filter_path).T
    _t = np.linspace(filter_data[0][0], filter_data[0][-1], 5000)
    filter_func = interp1d(
        filter_data[0], filter_data[1], bounds_error=False, fill_value=0.0
    )
    return np.round(np.average(MW_seaton_func(_t), weights=filter_func(_t)), 3)


Ff = {
    band: calculate_F(filter_path + filter_filenames[i]) for i, band in enumerate(bands)
}

print(Ff)

print("Collecting COSMOS2020 columns")

dic = {band + "_FLUX": data[band + "_FLUX"].byteswap().newbyteorder() for band in bands}
dic.update(
    {
        band + "_FLUXERR": data[band + "_FLUXERR"].byteswap().newbyteorder()
        for band in bands
    }
)
dic.update(
    {
        "ID": data["ID"].byteswap().newbyteorder(),
        "ALPHA_J2000": data["ALPHA_J2000"].byteswap().newbyteorder(),
        "DELTA_J2000": data["DELTA_J2000"].byteswap().newbyteorder(),
        "FLAG_COMBINED": data["FLAG_COMBINED"].byteswap().newbyteorder(),
        "lp_type": data["lp_type"].byteswap().newbyteorder(),
        "EBV_MW": data["EBV_MW"].byteswap().newbyteorder(),
        "ez_z_phot": data["ez_z_phot"].byteswap().newbyteorder(),
        # 'lp_NbFilt': data['lp_NbFilt'].byteswap().newbyteorder(),
    }
)

df = pd.DataFrame(dic)
df = df[df.FLAG_COMBINED == 0]  # Mask out bad regions.
df = df[df.lp_type == 0]  # Select only galaxies.
df["nbands"] = np.sum(
    np.isfinite(df.loc[:, [band + "_FLUX" for band in bands]].values), axis=1
)
df = df[df.nbands == len(bands)]  # Select objects with all bands measured.


for band in bands:
    # Correct MW extinction from observed fluxes.
    # mag_corr = mag_obs - 0.4 * EBV * Ff
    # flux_corr = flux_obs * 10**(0.4 * EBV * Ff)
    factor = abs(df.loc[:, band + "_FLUX"].values) * 10 ** (
        0.4 * df.EBV_MW.values * Ff[band]
    )
    df.loc[:, band + "_FLUX"]


# Define magnitude and calculate colors.
mag_i = flux2lupmag(df.HSC_i_FLUX.values)
z_obs = df.ez_z_phot.values
colors_data = np.zeros((len(bands) - 1, len(df)))

for i in range(len(bands) - 1):
    colors_data[i] = flux2lupmag(df[bands[i] + "_FLUX"].values) - flux2lupmag(
        df[bands[i + 1] + "_FLUX"].values
    )

gal_mags = np.zeros((len(df), len(bands)))
for i in range(len(bands)):
    gal_mags[:, i] = flux2lupmag(df[bands[i] + "_FLUX"].values)

bins_mag = np.linspace(18, 23, 20)
bins_color = np.linspace(-1.0, 2.0, 20)

bins_LO_mag = bins_mag[:-1]
bins_HI_mag = bins_mag[1:]
ndsig_mag = np.ones_like(colors_data[0]) * np.diff(bins_mag)[0]

bins_LO_color = bins_color[:-1]
bins_HI_color = bins_color[1:]
ndsig_color = np.ones_like(colors_data[0]) * np.diff(bins_color)[0]

true_i_counts = []
true_color_counts = []

diff_i_counts = []
diff_color_counts = []

true_mask_mag = (mag_i > 18.0) & (mag_i < 23.0)


def get_counts(zmin, zmax):
    mask_z = (z_obs > zmin) & (z_obs < zmax)
    mask = mask_z & true_mask_mag
    true_counts_i = np.histogram(mag_i[mask], bins_mag)[0] / sum(mask)
    true_counts_color = np.array(
        [np.histogram(x, bins_color)[0] / sum(mask) for x in colors_data[:, mask]]
    )

    diff_counts_i, diff_counts_colors = calculate_1d_COSMOS_colors_counts_singlez_bin(
        gal_mags[mask_z],
        ndsig_mag[mask_z],
        ndsig_color[mask_z],
        bins_LO_mag,
        bins_HI_mag,
        bins_LO_color,
        bins_HI_color,
    )
    return true_counts_i, true_counts_color, diff_counts_i, diff_counts_colors


print("Calculating target data")

zbins = np.arange(0.1, 1.6, 0.2)
for i in range(len(zbins) - 1):
    print("Redshift bin", zbins[i], zbins[i + 1])
    res = get_counts(zbins[i], zbins[i + 1])
    true_i_counts.append(res[0])
    true_color_counts.append(res[1])
    diff_i_counts.append(res[2])
    diff_color_counts.append(res[3])


true_i_counts = np.array(true_i_counts)
true_color_counts = np.array(true_color_counts)
diff_i_counts = np.array(diff_i_counts)
diff_color_counts = np.array(diff_color_counts)

outpath = "/lcrc/project/halotools/alarcon/data/DESC_mocks_data/"
