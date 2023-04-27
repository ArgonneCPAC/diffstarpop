import os
import numpy as np
import h5py
from diffstar.fit_smah_helpers import get_header
from diffstar.utils import _get_dt_array
from diffstar.constants import TODAY
from diffstar.stars import fstar_tools

from diffstarpop.star_wrappers import sm_sfr_history_diffstar_scan_XsfhXmah_vmap
import warnings
from umachine_pyio.load_mock import load_mock_from_binaries
from astropy.cosmology import Planck15


smdpl_binaries_path = "/lcrc/project/halotools/UniverseMachine/SMDPL/sfh_z0_binaries/"
smdpl_params_path = "/lcrc/project/halotools/alarcon/results/SMDPL/"

H_SMDPL = 0.678


def load_SMDPL_data(subvols, data_drn=smdpl_binaries_path):
    galprops = ["halo_id", "mpeak_history_main_prog"]
    _halos = load_mock_from_binaries(subvols, root_dirname=data_drn, galprops=galprops)
    halo_ids = np.array(_halos["halo_id"])
    _mah = np.maximum.accumulate(_halos["mpeak_history_main_prog"], axis=1)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        log_mahs = np.where(_mah == 0, 0, np.log10(_mah))

    # Needed for .../SMDPL/sfr_catalogs
    # but not for .../SMDPL/sfh_z0_binaries
    # log_mahs = np.where(log_mahs > 0.0, log_mahs + np.log10(H_BPL), log_mahs)

    SMDPL_a = np.load("/lcrc/project/halotools/UniverseMachine/SMDPL/scale_list.npy")
    SMDPL_z = 1.0 / SMDPL_a - 1.0
    SMDPL_t = Planck15.age(SMDPL_z).value

    log_mah_fit_min = 10.0

    # From https://www.cosmosim.org/cms/simulations/smdpl/
    # The mass particle resolution is 9.63e7 Msun/h
    particle_mass_res = 9.63e7 / H_SMDPL
    # So we cut halos with M0 below 500 times the mass resolution.
    logmpeak_fit_min = np.log10(500 * particle_mass_res)
    logmpeak = log_mahs[:, -1]

    sel = logmpeak >= logmpeak_fit_min
    log_mahs = log_mahs[sel]
    halo_ids = halo_ids[sel]

    return halo_ids, log_mahs, SMDPL_t, log_mah_fit_min


def get_mah_params(runname, data_path=smdpl_params_path):

    fitting_data = dict()

    fn = os.path.join(data_path, runname)
    with h5py.File(fn, "r") as hdf:
        for key in hdf.keys():
            if key == "halo_id":
                fitting_data[key] = hdf[key][...]
            else:
                fitting_data["fit_" + key] = hdf[key][...]

    mah_params = np.array(
        [
            fitting_data["fit_logmp_fit"],
            fitting_data["fit_mah_logtc"],
            fitting_data["fit_early_index"],
            fitting_data["fit_late_index"],
        ]
    ).T
    return mah_params


def get_sfh_params(runname, data_path=smdpl_params_path):
    sfr_fitdata = dict()

    fn = os.path.join(data_path, runname)
    with h5py.File(fn, "r") as hdf:
        for key in hdf.keys():
            sfr_fitdata[key] = hdf[key][...]

    colnames = get_header()[1:].strip().split()
    sfr_colnames = colnames[1:6]
    q_colnames = colnames[6:10]

    u_sfr_fit_params = np.array([sfr_fitdata[key] for key in sfr_colnames]).T
    u_q_fit_params = np.array([sfr_fitdata[key] for key in q_colnames]).T

    return u_sfr_fit_params, u_q_fit_params


def calculate_histories():

    tarr = np.linspace(1.0, TODAY, 20)

    dt = _get_dt_array(tarr)
    lgt = np.log10(tarr)

    fstar_tdelay = 1.0
    index_select, index_high = fstar_tools(tarr, fstar_tdelay=fstar_tdelay)

    i = 0
    runname = "run1_SMDPL_diffmah_default_%d.h5" % i
    mah_params = get_mah_params(runname)
    runname = "run1_SMDPL_diffstar_default_%i.h5" % i
    sfr_ms_params, q_params = get_sfh_params(runname)

    _mstar, _sfr, _fstar = sm_sfr_history_diffstar_scan_XsfhXmah_vmap(
        tarr,
        lgt,
        dt,
        mah_params,
        sfr_ms_params,
        q_params,
        index_select,
        index_high,
        fstar_tdelay,
    )
    breakpoint()
