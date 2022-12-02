"""
"""
import numpy as np
from .. import triweight_burst as tb

ZZ = np.zeros(1)


def test_sfh_post_burst_vmap_integrates_to_correct_final_mstar():
    lgtarr = np.linspace(1, 11, 100)

    for mstar_tot in (1e5, 1e9, 1e12):
        cuml_mstar = []
        for lgtmax in lgtarr:
            lgtx = np.linspace(2, lgtmax, 500)
            sfh = tb._sfh_post_burst_vmap(10**lgtx, mstar_tot)
            result = np.trapz(sfh, x=10**lgtx)
            cuml_mstar.append(result)
        cuml_mstar = np.array(cuml_mstar)
        assert np.allclose(cuml_mstar[-1], mstar_tot, rtol=0.1)
        assert np.all(np.diff(cuml_mstar) / mstar_tot >= -0.1)


def test_sfh_post_burst_vmap_integrates_to_mstar_post_burst_vmap():
    lgtarr = np.linspace(1, 11, 200)

    BURST_KERN_LGTHI = tb.BURST_KERN_LGTLO + tb.BURST_KERN_DLGT
    test_msk = (lgtarr > tb.BURST_KERN_LGTLO + 0.1) & (lgtarr < BURST_KERN_LGTHI - 0.1)

    for mstar_tot in (1e5, 1e9, 1e12):
        cuml_mstar = []
        for lgtmax in lgtarr:
            lgtx = np.linspace(2, lgtmax, 200)
            sfh = tb._sfh_post_burst_vmap(10**lgtx, mstar_tot)
            result = np.trapz(sfh, x=10**lgtx)
            cuml_mstar.append(result)
        cuml_mstar = np.array(cuml_mstar)
        cuml_mstar2 = tb._mstar_post_burst_vmap(10**lgtarr, mstar_tot)
        assert np.allclose(cuml_mstar[test_msk], cuml_mstar2[test_msk], rtol=0.05)


def test_sfh_post_burst_vmap_has_correct_asymptotic_behavior():
    lgtarr = np.linspace(1, 11, 100)
    mstar_tot = 1e10
    sfh = tb._sfh_post_burst_vmap(10**lgtarr, mstar_tot)

    early_time_msk = lgtarr < tb.BURST_KERN_LGTLO
    assert np.allclose(sfh[early_time_msk], 0)

    late_time_msk = lgtarr > (tb.BURST_KERN_LGTLO + tb.BURST_KERN_DLGT)
    assert np.allclose(sfh[late_time_msk], 0, atol=0.0001)


def test_mstar_post_burst_vmap_has_correct_asymptotic_behavior():
    lgtarr = np.linspace(1, 11, 100)
    for mstar_tot in (1e5, 1e9, 1e12):
        smh = tb._mstar_post_burst_vmap(10**lgtarr, mstar_tot)

        early_time_msk = lgtarr < tb.BURST_KERN_LGTLO
        assert np.allclose(smh[early_time_msk], 0, atol=0.001)

        late_time_msk = lgtarr > tb.BURST_KERN_LGTLO + tb.BURST_KERN_DLGT
        assert np.allclose(smh[late_time_msk], mstar_tot, rtol=0.01)


def test_cuml_prob_lgage_is_properly_bounded():
    lgtarr = np.linspace(1, 11, 100)
    time_since_burst = 10**lgtarr
    p = tb._cuml_prob_age_vmap(time_since_burst)
    assert np.all(p >= 0)
    assert np.all(p <= 1)


def test_cuml_prob_lgage_has_correct_asymptotic_behavior():
    lgtarr = np.linspace(1, 11, 100)
    time_since_burst = 10**lgtarr
    p = tb._cuml_prob_age_vmap(time_since_burst)

    early_time_msk = lgtarr < tb.BURST_KERN_LGTLO
    assert np.allclose(p[early_time_msk], 0, atol=0.001)

    late_time_msk = lgtarr > tb.BURST_KERN_LGTLO + tb.BURST_KERN_DLGT
    assert np.allclose(p[late_time_msk], 1, atol=0.001)


def test_cuml_prob_lgage_is_monotonic():
    lgtarr = np.linspace(1, 11, 100)
    time_since_burst = 10**lgtarr
    p = tb._cuml_prob_age_vmap(time_since_burst)
    assert np.all(np.diff(p) >= 0)


def test_sfh_post_burst_vmap_vs_kern_args():

    res_kern = tb._sfh_post_burst_kern(1e9, 1e9)
    res_vmap = tb._sfh_post_burst_vmap(1e9 + ZZ, 1e9)
    assert np.allclose(res_kern, res_vmap)


def test_mstar_post_burst_vmap_vs_kern_args():

    res_kern = tb._mstar_post_burst_kern(1e9, 1e9)
    res_vmap = tb._mstar_post_burst_vmap(1e9 + ZZ, 1e9)
    assert np.allclose(res_kern, res_vmap)


def test_sfh_post_burst_is_never_nan():
    assert np.isfinite(tb._sfh_post_burst_kern(-1e9, 1e9))
    assert np.isfinite(tb._sfh_post_burst_vmap(-1e9 + ZZ, 1e9))


def test_mstar_post_burst_is_never_nan():
    assert np.isfinite(tb._mstar_post_burst_kern(-1e9, 1e9))
    assert np.isfinite(tb._mstar_post_burst_vmap(-1e9 + ZZ, 1e9))
