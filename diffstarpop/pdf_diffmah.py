import numpy as np
from diffmah.individual_halo_assembly import calc_halo_history
from jax import random as jran


def get_diffmah_grid(
    tarr,
    n_halos_per_bin,
    jran_key,
    logm0_binmids,
    logm0_bin_widths,
    logm0_halos,
    mah_tauc_halos,
    mah_early_halos,
    mah_late_halos,
    t0,
):
    """Get a grid of halo histories by downsampling the input diffmah parameters.

    Parameters
    ----------
    tarr : ndarray of shape (n_times, )
        Description.
    n_halos_per_bin : int
        Number of halos to select per bin
    jran_key : obj
        jran.PRNGKey(seed)
    logm0_binmids : ndarray of shape (n_bins, )
        Midpoint of the logarithmic halo mass bins
    logm0_bin_widths : ndarray of shape (n_bins, )
        logarithmic width of the halo mass bin
    logm0_halos : ndarray of shape (n_halos, )
        Description.
    mah_tauc_halos : ndarray of shape (n_halos, )
        Description.
    mah_early_halos : ndarray of shape (n_halos, )
        Description.
    mah_late_halos : ndarray of shape (n_halos, )
        Description.
    Returns
    -------
    dmhdt_grid : ndarray of shape (n_bins*n_halos_per_bin, n_times)
        Description.
    log_mah_grid : ndarray of shape (n_bins*n_halos_per_bin, n_times)
        Description.
    logm0_sample : ndarray of shape (n_bins*n_halos_per_bin, )
        Description.
    lgtc_sample : ndarray of shape (n_bins*n_halos_per_bin, )
        Description.
    early_indx_sample : ndarray of shape (n_bins*n_halos_per_bin, )
        Description.
    late_indx_sample : ndarray of shape (n_bins*n_halos_per_bin, )
        Description.
    """
    diffmah_params_grid = get_binned_halo_sample(
        n_halos_per_bin,
        jran_key,
        logm0_binmids,
        logm0_bin_widths,
        logm0_halos,
        mah_tauc_halos,
        mah_early_halos,
        mah_late_halos,
    )

    dmhdt_grid, log_mah_grid = calc_halo_history(tarr, t0, *diffmah_params_grid)
    return (dmhdt_grid, log_mah_grid, *diffmah_params_grid)


def get_binned_halo_sample(
    n_per_bin, jran_key, logm0_binmids, logm0_bin_widths, logm0_halos, *other_halo_props
):
    """Retrieve a downsampling of halos binned by the input logm0

    Parameters
    ----------
    n_per_bin : int
        Number of halos to select per bin
    jran_key : obj
        jran.PRNGKey(seed)
    logm0_binmids : ndarray of shape (n_bins, )
        Midpoint of the logarithmic halo mass bins
    logm0_bin_widths : ndarray of shape (n_bins, )
        logarithmic width of the halo mass bin
    logm0_halos : ndarray of shape (n_halos, )
        Description.
    other_halo_props : length m sequence of ndarrays of shape (n_halos, )
        Description.
    Returns
    -------
    binned_halo_sample : list of length m+1
        Each element is an ndarray of shape (n_per_bin*n_bins, )
        The first element is the array of logm0
        The remaining m elements are the input halo properties

    Notes
    -----
    The binned_halo_sample is what is used to generate a diffmah grid

    """
    binned_halos = get_binned_halos(
        logm0_binmids, logm0_bin_widths, logm0_halos, *other_halo_props
    )
    n_bins = len(logm0_binmids)
    ran_keys = jran.split(jran_key, n_bins)
    collector = []
    for ran_key_bin, halos_in_bin in zip(ran_keys, binned_halos):
        halo_sample = randomly_select_halos(n_per_bin, ran_key_bin, *halos_in_bin)
        collector.append(halo_sample)

    binned_halo_sample = [np.empty(0) for __ in range(len(collector[0]))]
    for halos_in_bin in collector:
        gen = zip(binned_halo_sample, halos_in_bin)
        binned_halo_sample = [np.concatenate((a, b)) for a, b in gen]
    return binned_halo_sample


def get_binned_halos(logm0_binmids, logm0_bin_widths, logm0_halos, *other_halo_props):
    """Retrieve a collection of halos binned by the input logm0

    Parameters
    ----------
    logm0_binmids : ndarray of shape (n_bins, )
        Midpoint of the logarithmic halo mass bins
    logm0_bin_widths : ndarray of shape (n_bins, )
        logarithmic width of the halo mass bin
    logm0_halos : ndarray of shape (n_halos, )
        Description.
    other_halo_props : length m sequence of ndarrays of shape (n_halos, )
        Description.
    Returns
    -------
    binned_halos : list of length n_bins
        Element i is a tuple of halo properties that fall within the ith bin
        The first element of each tuple is the value of logm0 of the halos
        The remaining m elements are the input halo properties
        So each tuple has length m+1

    """
    msg = "input logm0_binmids and logm0_bin_widths must have same length"
    assert len(logm0_binmids) == len(logm0_bin_widths), msg
    binned_halos = []
    for cen, w in zip(logm0_binmids, logm0_bin_widths):
        msk = np.abs(logm0_halos - cen) < w
        m0_haloprop_data = [h[msk] for h in other_halo_props]
        m0_data = (logm0_halos[msk], *m0_haloprop_data)
        binned_halos.append(m0_data)
    return binned_halos


def get_binned_halo_sample_p50(
    n_per_bin,
    jran_key,
    logm0_binmids,
    logm0_bin_widths,
    logm0_halos,
    p50_binmids,
    p50_bin_widths,
    p50_halos,
    *other_halo_props
):
    """Retrieve a downsampling of halos binned by the input logm0 and p50

    Parameters
    ----------
    n_per_bin : int
        Number of halos to select per bin
    jran_key : obj
        jran.PRNGKey(seed)
    logm0_binmids : ndarray of shape (n_bins_mass, )
        Midpoint of the logarithmic halo mass bins
    logm0_bin_widths : ndarray of shape (n_bins_mass, )
        logarithmic width of the halo mass bin
    logm0_halos : ndarray of shape (n_halos, )
        Logarithmic halo mass values for each halo.
    p50_binmids : ndarray of shape (n_bins_p50, )
        Midpoint of the formation time percentile bins.
    p50_bin_widths : ndarray of shape (n_bins_p50, )
        Width of the formation time percentile bins.
    p50_halos : ndarray of shape (n_halos, )
        Formation time values for each halo.
    other_halo_props : length m sequence of ndarrays of shape (n_halos, )
        Description.
    Returns
    -------
    binned_halo_sample : ndarray of shape (n_bins_mass, n_bins_p50, n_per_bin, m+1 )
        The first element is the array of logm0
        The remaining m elements are the input halo properties

    Notes
    -----
    The binned_halo_sample is what is used to generate a diffmah grid

    """

    other_halo_props = np.array(other_halo_props)
    n_bins_p = len(p50_binmids)
    n_bins_h = len(logm0_binmids)
    n_props = len(other_halo_props) + 1
    ran_keys = jran.split(jran_key, n_bins_p)

    result = []
    for cen, w, ran_key_bin in zip(p50_binmids, p50_bin_widths, ran_keys):
        msk = np.abs(p50_halos - cen) < w

        binned_halos = get_binned_halo_sample(
            n_per_bin,
            ran_key_bin,
            logm0_binmids,
            logm0_bin_widths,
            logm0_halos[msk],
            *other_halo_props[:, msk]
        )

        binned_halos = np.array(binned_halos).reshape((n_props, n_bins_h, n_per_bin))
        result.append(binned_halos)

    result = np.einsum("phmn->mpnh", np.array(result))
    return result


def randomly_select_halos(n_sample, jran_key, *halo_sample_properties):
    n_halos = halo_sample_properties[0].size
    indx_all = np.arange(n_halos).astype("i8")
    indx_select = jran.choice(jran_key, indx_all, shape=(n_sample,), replace=False)
    return [arr[np.array(indx_select)] for arr in halo_sample_properties]
