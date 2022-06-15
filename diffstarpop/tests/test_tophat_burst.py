"""
"""
from ..tophat_burst import _tophat_burst_sfh, _tophat_burst_smh
from ..tophat_burst import DT_BURST

EPSILON = 0.1


def test_tophat_burst_sfh():
    tburst_yr = 500_000
    t0 = tburst_yr - DT_BURST
    thalf = t0 + DT_BURST / 2
    sfr_burst = 1.0
    assert _tophat_burst_sfh(t0 - EPSILON, tburst_yr, sfr_burst) == 0.0
    assert _tophat_burst_sfh(t0, tburst_yr, sfr_burst) == sfr_burst
    assert _tophat_burst_sfh(t0 + EPSILON, tburst_yr, sfr_burst) == sfr_burst
    assert _tophat_burst_sfh(thalf, tburst_yr, sfr_burst) == sfr_burst
    assert _tophat_burst_sfh(tburst_yr - EPSILON, tburst_yr, sfr_burst) == sfr_burst
    assert _tophat_burst_sfh(tburst_yr, tburst_yr, sfr_burst) == sfr_burst
    assert _tophat_burst_sfh(tburst_yr + EPSILON, tburst_yr, sfr_burst) == 0.0


def test_tophat_burst_smh():
    tburst_yr = 500_000
    t0 = tburst_yr - DT_BURST
    thalf = t0 + DT_BURST / 2
    sfr_burst = 1.0
    mstar_burst = sfr_burst * DT_BURST
    assert _tophat_burst_smh(t0 - EPSILON, tburst_yr, sfr_burst) == 0.0
    assert _tophat_burst_smh(t0, tburst_yr, sfr_burst) == 0.0
    assert _tophat_burst_smh(thalf, tburst_yr, sfr_burst) == mstar_burst / 2
    assert _tophat_burst_smh(tburst_yr, tburst_yr, sfr_burst) == mstar_burst
    assert _tophat_burst_smh(tburst_yr + EPSILON, tburst_yr, sfr_burst) == mstar_burst
