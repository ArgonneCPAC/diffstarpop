"""
"""
import numpy as np
from ..triweight_burst import _cuml_prob_age, _sfh_post_burst, _mstar_post_burst
from ..triweight_burst import BURST_KERN_LGTLO, BURST_KERN_DLGT


def test_sfh_post_burst_integrates_to_correct_final_mstar():
    lgtarr = np.linspace(1, 11, 100)

    for mstar_tot in (1e5, 1e9, 1e12):
        cuml_mstar = []
        for lgtmax in lgtarr:
            lgtx = np.linspace(2, lgtmax, 500)
            sfh = _sfh_post_burst(10 ** lgtx, mstar_tot)
            result = np.trapz(sfh, x=10 ** lgtx)
            cuml_mstar.append(result)
        cuml_mstar = np.array(cuml_mstar)
        assert np.allclose(cuml_mstar[-1], mstar_tot, rtol=0.1)
        assert np.all(np.diff(cuml_mstar) / mstar_tot >= -0.1)


def test_sfh_post_burst_integrates_to_mstar_post_burst():
    lgtarr = np.linspace(1, 11, 200)

    BURST_KERN_THI = BURST_KERN_LGTLO + BURST_KERN_DLGT
    test_msk = (lgtarr > BURST_KERN_LGTLO + 0.1) & (lgtarr < BURST_KERN_THI - 0.1)

    for mstar_tot in (1e5, 1e9, 1e12):
        cuml_mstar = []
        for lgtmax in lgtarr:
            lgtx = np.linspace(2, lgtmax, 200)
            sfh = _sfh_post_burst(10 ** lgtx, mstar_tot)
            result = np.trapz(sfh, x=10 ** lgtx)
            cuml_mstar.append(result)
        cuml_mstar = np.array(cuml_mstar)
        cuml_mstar2 = _mstar_post_burst(10 ** lgtarr, mstar_tot)
        assert np.allclose(cuml_mstar[test_msk], cuml_mstar2[test_msk], rtol=0.05)


def test_sfh_post_burst_has_correct_asymptotic_behavior():
    lgtarr = np.linspace(1, 11, 100)
    mstar_tot = 1e10
    sfh = _sfh_post_burst(10 ** lgtarr, mstar_tot)

    early_time_msk = lgtarr < BURST_KERN_LGTLO
    assert np.allclose(sfh[early_time_msk], 0)

    late_time_msk = lgtarr > (BURST_KERN_LGTLO + BURST_KERN_DLGT)
    assert np.allclose(sfh[late_time_msk], 0, atol=0.0001)


def test_mstar_post_burst_has_correct_asymptotic_behavior():
    lgtarr = np.linspace(1, 11, 100)
    for mstar_tot in (1e5, 1e9, 1e12):
        smh = _mstar_post_burst(10 ** lgtarr, mstar_tot)

        early_time_msk = lgtarr < BURST_KERN_LGTLO
        assert np.allclose(smh[early_time_msk], 0, atol=0.001)

        late_time_msk = lgtarr > BURST_KERN_LGTLO + BURST_KERN_DLGT
        assert np.allclose(smh[late_time_msk], mstar_tot, rtol=0.01)


def test_cuml_prob_lgage_is_properly_bounded():
    lgtarr = np.linspace(1, 11, 100)
    time_since_burst = 10 ** lgtarr
    p = _cuml_prob_age(time_since_burst)
    assert np.all(p >= 0)
    assert np.all(p <= 1)


def test_cuml_prob_lgage_has_correct_asymptotic_behavior():
    lgtarr = np.linspace(1, 11, 100)
    time_since_burst = 10 ** lgtarr
    p = _cuml_prob_age(time_since_burst)

    early_time_msk = lgtarr < BURST_KERN_LGTLO
    assert np.allclose(p[early_time_msk], 0, atol=0.001)

    late_time_msk = lgtarr > BURST_KERN_LGTLO + BURST_KERN_DLGT
    assert np.allclose(p[late_time_msk], 1, atol=0.001)


def test_cuml_prob_lgage_is_monotonic():
    lgtarr = np.linspace(1, 11, 100)
    time_since_burst = 10 ** lgtarr
    p = _cuml_prob_age(time_since_burst)
    assert np.all(np.diff(p) >= 0)
