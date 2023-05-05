import os
import numpy as np
import h5py
from diffstar.fit_smah_helpers import get_header
from diffstar.utils import _get_dt_array
from diffstar.constants import TODAY
from diffstar.stars import fstar_tools
from diffmah.individual_halo_assembly import _calc_halo_history
from diffstarpop.utils import get_t50_p50
from diffstarpop.star_wrappers import sm_sfr_history_diffstar_scan_XsfhXmah_vmap
import warnings
from umachine_pyio.load_mock import load_mock_from_binaries
from astropy.cosmology import Planck15
from jax import vmap, jit as jjit, numpy as jnp
import resource
_calc_halo_history_vmap = jjit(vmap(_calc_halo_history, in_axes=(None, *[0]*6)))

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
            np.log10(fitting_data["fit_t0"]),
            fitting_data["fit_logmp_fit"],
            fitting_data["fit_mah_logtc"],
            fitting_data["fit_mah_k"],
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


def calculate_histories(binminds, binwidths):

    tarr = np.linspace(1.0, TODAY, 20)

    dt = _get_dt_array(tarr)
    lgt = np.log10(tarr)

    fstar_tdelay = 1.0
    index_select, index_high = fstar_tools(tarr, fstar_tdelay=fstar_tdelay)

    dmhdt = []
    log_mah = []
    mstar = []
    sfr = []
    fstar = []

    for i in range(120):
        runname = "run1_SMDPL_diffmah_default_%d.h5" % i
        mah_params = get_mah_params(runname)
        
        print(i, len(mah_params), resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)

        runname = "run1_SMDPL_diffstar_default_%i.h5" % i
        sfr_ms_params, q_params = get_sfh_params(runname)

        

        _res = _calc_halo_history_vmap(lgt, *mah_params.T)
        dmhdt.append(_res[0])
        log_mah.append(_res[1])

        _res = sm_sfr_history_diffstar_scan_XsfhXmah_vmap(
            tarr,
            lgt,
            dt,
            mah_params[:, [1,2,4,5]],
            sfr_ms_params,
            q_params,
            index_select,
            index_high,
            fstar_tdelay,
        )
        mstar.append(_res[0])
        sfr.append(_res[1])
        fstar.append(_res[2])

    dmhdt = np.concatenate(dmhdt)
    log_mah = np.concatenate(log_mah)
    mstar = np.concatenate(mstar)
    sfr = np.concatenate(sfr)
    fstar = np.concatenate(fstar)

    print("Calculating p50...")

    t50, p50 = get_t50_p50(tarr, 10**log_mah, 0.5, log_mah[:,-1], window_length = 101)
    
    # sFstar = fstar / mstar[:, index_select]
    ssfr = sfr / mstar
    weights_quench_bin = jnp.where(ssfr > 1e-11, 1.0, 0.0)
    logmpeak = log_mah[:, -1]

    return mstar, sfr, fstar, p50, weights_quench_bin, logmpeak 


def calculate_SMDPL_sumstats():
    logm0_binmids = np.linspace(11.5, 13.5, 5)
    logm0_bin_widths = np.ones_like(logm0_binmids) * 0.1

    mstar_histories, sfr_histories, fstar_histories, p50, weights_MS, logmpeak = calculate_histories(logm0_binmids, logm0_bin_widths)


    stats = []
    for i in range(len(logm0_binmids)):

        print("Calculating m0=[%.2f, %.2f]"%(logm0_binmids[i] - logm0_bin_widths[i], logm0_binmids[i] + logm0_bin_widths[i]))
        sel = (logmpeak > logm0_binmids[i] - logm0_bin_widths[i]) & (logmpeak < logm0_binmids[i] + logm0_bin_widths[i])

        _stats = calculate_sumstats_bin(
            mstar_histories[sel], 
            sfr_histories[sel], 
            # fstar_histories[sel], 
            p50[sel], 
            weights_MS[sel]
        )
        stats.append(_stats)

    print("Reshaping results")
        
    new_stats = []
    nres = len(_stats)
    for j in range(nres):
        _new_stats = []
        for i in range(len(logm0_binmids)):
            _new_stats.append(stats[i][j])
        new_stats.append(np.array(_new_stats))


    np.save(smdpl_params_path+"SMDPL_sfh_sumstats.npy", new_stats)
    return new_stats



def calculate_sumstats_bin(
    mstar_histories, sfr_histories, p50, weights_MS
):

    weights_Q = 1.0 - weights_MS

    # Clip weights. When all weights in a time
    # step are 0, Nans will occur in gradients.
    eps = 1e-10
    weights_Q = jnp.clip(weights_Q, eps, None)
    weights_MS = jnp.clip(weights_MS, eps, None)

    weights_early = jnp.where(p50 < 0.5, 1.0, 0.0)
    weights_late = 1.0 - weights_early
    weights_early = jnp.clip(weights_early, eps, None)
    weights_late = jnp.clip(weights_late, eps, None)

    mstar_histories = jnp.where(mstar_histories > 0.0, jnp.log10(mstar_histories), 0.0)
    sfr_histories = jnp.where(sfr_histories > 0.0, jnp.log10(sfr_histories), 0.0)
    # fstar_histories = jnp.where(fstar_histories > 0.0, jnp.log10(fstar_histories), 0.0)

    mean_sm = jnp.average(mstar_histories, axis=0)
    mean_sfr_MS = jnp.average(sfr_histories, weights=weights_MS, axis=0)
    mean_sfr_Q = jnp.average(sfr_histories, weights=weights_Q, axis=0)

    mean_sm_early = jnp.average(mstar_histories, weights=weights_early, axis=0)
    mean_sm_late = jnp.average(mstar_histories, weights=weights_late, axis=0)

    variance_sm = jnp.average(
        (mstar_histories - mean_sm[None, :]) ** 2, axis=0,
    )

    variance_sfr_MS = jnp.average(
        (sfr_histories - mean_sfr_MS[None, :]) ** 2, weights=weights_MS, axis=0,
    )
    variance_sfr_Q = jnp.average(
        (sfr_histories - mean_sfr_Q[None, :]) ** 2, weights=weights_Q, axis=0,
    )
    variance_sm_early = jnp.average(
        (mstar_histories - mean_sm[None, :]) ** 2, weights=weights_early, axis=0,
    )
    variance_sm_late = jnp.average(
        (mstar_histories - mean_sm[None, :]) ** 2, weights=weights_late, axis=0,
    )

    NHALO_MS = jnp.sum(weights_MS, axis=0)
    NHALO_Q = jnp.sum(weights_Q, axis=0)
    quench_frac = NHALO_Q / (NHALO_Q + NHALO_MS)

    mean_sfr_Q = jnp.where(quench_frac == 0.0, 0.0, mean_sfr_Q)
    variance_sfr_Q = jnp.where(quench_frac == 0.0, 0.0, variance_sfr_Q)
    mean_sfr_MS = jnp.where(quench_frac == 1.0, 0.0, mean_sfr_MS)
    variance_sfr_MS = jnp.where(quench_frac == 1.0, 0.0, variance_sfr_MS)

    NHALO_MS_early = jnp.sum(weights_MS * weights_early[:, None], axis=0)
    NHALO_Q_early = jnp.sum(weights_Q * weights_early[:, None], axis=0)
    quench_frac_early = NHALO_Q_early / (NHALO_Q_early + NHALO_MS_early)

    NHALO_MS_late = jnp.sum(weights_MS * weights_late[:, None], axis=0)
    NHALO_Q_late = jnp.sum(weights_Q * weights_late[:, None], axis=0)
    quench_frac_late = NHALO_Q_late / (NHALO_Q_late + NHALO_MS_late)

    _out = (
        mean_sm,
        variance_sm,
        mean_sfr_MS,
        mean_sfr_Q,
        variance_sfr_MS,
        variance_sfr_Q,
        quench_frac,
        mean_sm_early,
        mean_sm_late,
        variance_sm_early,
        variance_sm_late,
        quench_frac_early,
        quench_frac_late,
    )
    return _out



if __name__ == "__main__":
    calculate_SMDPL_sumstats()
