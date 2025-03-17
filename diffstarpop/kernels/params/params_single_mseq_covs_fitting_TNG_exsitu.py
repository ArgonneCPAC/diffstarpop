from collections import OrderedDict, namedtuple

RHO_BOUNDS = (-0.3, 0.3)

SFH_PDF_QUENCH_MU_PDICT = OrderedDict(
    mean_ulgm_int=11.827,
    mean_ulgm_slp=0.096,
    mean_ulgy_int=0.090,
    mean_ulgy_slp=0.520,
    mean_ul_int=-1.101,
    mean_ul_slp=0.132,
    mean_utau_int=0.100,
    mean_utau_slp=-7.166,
    mean_uqt_int=0.901,
    mean_uqt_slp=-0.678,
    mean_uqs_int=-0.447,
    mean_uqs_slp=-0.585,
    mean_udrop_int=-2.601,
    mean_udrop_slp=1.228,
    mean_urej_int=-2.458,
    mean_urej_slp=0.439,
)
SFH_PDF_QUENCH_MU_BOUNDS_PDICT = OrderedDict(
    mean_ulgm_int=(11.0, 13.0),
    mean_ulgm_slp=(-20.0, 20.0),
    mean_ulgy_int=(-1.0, 3.5),
    mean_ulgy_slp=(-20.0, 20.0),
    mean_ul_int=(-3.0, 5.0),
    mean_ul_slp=(-20.0, 20.0),
    mean_utau_int=(-25.0, 50.0),
    mean_utau_slp=(-20.0, 20.0),
    mean_uqt_int=(0.0, 2.0),
    mean_uqt_slp=(-20.0, 20.0),
    mean_uqs_int=(-5.0, 2.0),
    mean_uqs_slp=(-20.0, 20.0),
    mean_udrop_int=(-3.0, 2.0),
    mean_udrop_slp=(-20.0, 20.0),
    mean_urej_int=(-10.0, 2.0),
    mean_urej_slp=(-20.0, 20.0),
)

SFH_PDF_QUENCH_COV_MS_BLOCK_PDICT = OrderedDict(
    std_ulgm_int=0.119,
    std_ulgm_slp=0.039,
    std_ulgy_int=0.224,
    std_ulgy_slp=-0.139,
    std_ul_int=0.275,
    std_ul_slp=-0.096,
    std_utau_int=3.298,
    std_utau_slp=-0.299,
    rho_ulgy_ulgm_int=1.724,
    rho_ulgy_ulgm_slp=-0.766,
    rho_ul_ulgm_int=-0.743,
    rho_ul_ulgm_slp=-0.391,
    rho_ul_ulgy_int=10.510,
    rho_ul_ulgy_slp=0.110,
    rho_utau_ulgm_int=8.971,
    rho_utau_ulgm_slp=-0.058,
    rho_utau_ulgy_int=4.023,
    rho_utau_ulgy_slp=3.676,
    rho_utau_ul_int=-0.489,
    rho_utau_ul_slp=0.853,
)
SFH_PDF_QUENCH_COV_MS_BLOCK_BOUNDS_PDICT = OrderedDict(
    std_ulgm_int=(0.01, 1.0),
    std_ulgm_slp=(-1.00, 1.0),
    std_ulgy_int=(0.01, 1.0),
    std_ulgy_slp=(-1.00, 1.0),
    std_ul_int=(0.01, 1.0),
    std_ul_slp=(-1.00, 1.0),
    std_utau_int=(1.00, 12.0),
    std_utau_slp=(-3.00, 3.0),
    rho_ulgy_ulgm_int=(-20.0, 20.0),
    rho_ulgy_ulgm_slp=(-20.0, 20.0),
    rho_ul_ulgm_int=(-20.0, 20.0),
    rho_ul_ulgm_slp=(-20.0, 20.0),
    rho_ul_ulgy_int=(-20.0, 20.0),
    rho_ul_ulgy_slp=(-20.0, 20.0),
    rho_utau_ulgm_int=(-20.0, 20.0),
    rho_utau_ulgm_slp=(-20.0, 20.0),
    rho_utau_ulgy_int=(-20.0, 20.0),
    rho_utau_ulgy_slp=(-20.0, 20.0),
    rho_utau_ul_int=(-20.0, 20.0),
    rho_utau_ul_slp=(-20.0, 20.0),
)

SFH_PDF_QUENCH_COV_Q_BLOCK_PDICT = OrderedDict(
    std_uqt_int=0.082,
    std_uqt_slp=-0.104,
    std_uqs_int=0.600,
    std_uqs_slp=0.026,
    std_udrop_int=0.394,
    std_udrop_slp=0.169,
    std_urej_int=0.662,
    std_urej_slp=-0.122,
    rho_uqs_uqt_int=4.107,
    rho_uqs_uqt_slp=0.291,
    rho_udrop_uqt_int=-0.725,
    rho_udrop_uqt_slp=0.586,
    rho_udrop_uqs_int=8.910,
    rho_udrop_uqs_slp=0.078,
    rho_urej_uqt_int=-12.203,
    rho_urej_uqt_slp=-0.498,
    rho_urej_uqs_int=13.302,
    rho_urej_uqs_slp=0.165,
    rho_urej_udrop_int=-0.639,
    rho_urej_udrop_slp=0.454,
)
SFH_PDF_QUENCH_COV_Q_BLOCK_BOUNDS_PDICT = OrderedDict(
    std_uqt_int=(0.01, 0.5),
    std_uqt_slp=(-1.00, 1.0),
    std_uqs_int=(0.01, 1.0),
    std_uqs_slp=(-1.00, 1.0),
    std_udrop_int=(0.01, 2.0),
    std_udrop_slp=(-1.00, 1.0),
    std_urej_int=(0.01, 2.0),
    std_urej_slp=(-1.00, 1.0),
    rho_uqs_uqt_int=(-20.0, 20.0),
    rho_uqs_uqt_slp=(-20.0, 20.0),
    rho_udrop_uqt_int=(-20.0, 20.0),
    rho_udrop_uqt_slp=(-20.0, 20.0),
    rho_udrop_uqs_int=(-20.0, 20.0),
    rho_udrop_uqs_slp=(-20.0, 20.0),
    rho_urej_uqt_int=(-20.0, 20.0),
    rho_urej_uqt_slp=(-20.0, 20.0),
    rho_urej_uqs_int=(-20.0, 20.0),
    rho_urej_uqs_slp=(-20.0, 20.0),
    rho_urej_udrop_int=(-20.0, 20.0),
    rho_urej_udrop_slp=(-20.0, 20.0),
)
SFH_PDF_FRAC_QUENCH_PDICT = OrderedDict(
    frac_quench_x0=11.863,
    frac_quench_k=4.579,
    frac_quench_ylo=0.01,
    frac_quench_yhi=0.99,
)
SFH_PDF_FRAC_QUENCH_BOUNDS_PDICT = OrderedDict(
    frac_quench_x0=(10.0, 13.0),
    frac_quench_k=(0.01, 5.0),
    frac_quench_ylo=(0.0, 0.5),
    frac_quench_yhi=(0.5, 1.0),
)

BOUNDING_MEAN_VALS_PDICT = OrderedDict(
    mean_ulgm=(11.0, 13.0),
    mean_ulgy=(-1.0, 3.5),
    mean_ul=(-3.0, 5.0),
    mean_utau=(-25.0, 50.0),
    mean_uqt=(0.0, 2.0),
    mean_uqs=(-5.0, 2.0),
    mean_udrop=(-3.0, 2.0),
    mean_urej=(-10.0, 2.0),
)

BOUNDING_STD_VALS_PDICT = OrderedDict(
    std_ulgm=(0.01, 1.0),
    std_ulgy=(0.01, 1.0),
    std_ul=(0.01, 1.0),
    std_utau=(1.00, 12.0),
    std_uqt=(0.01, 0.5),
    std_uqs=(0.01, 1.0),
    std_udrop=(0.01, 2.0),
    std_urej=(0.01, 2.0),
)

BOUNDING_RHO_VALS_PDICT = OrderedDict(
    rho_ulgy_ulgm=RHO_BOUNDS,
    rho_ul_ulgm=RHO_BOUNDS,
    rho_ul_ulgy=RHO_BOUNDS,
    rho_utau_ulgm=RHO_BOUNDS,
    rho_utau_ulgy=RHO_BOUNDS,
    rho_utau_ul=RHO_BOUNDS,
    rho_uqs_uqt=RHO_BOUNDS,
    rho_udrop_uqt=RHO_BOUNDS,
    rho_udrop_uqs=RHO_BOUNDS,
    rho_urej_uqt=RHO_BOUNDS,
    rho_urej_uqs=RHO_BOUNDS,
    rho_urej_udrop=RHO_BOUNDS,
)

SFH_PDF_QUENCH_PDICT = SFH_PDF_FRAC_QUENCH_PDICT.copy()
SFH_PDF_QUENCH_PDICT.update(SFH_PDF_QUENCH_MU_PDICT)
SFH_PDF_QUENCH_PDICT.update(SFH_PDF_QUENCH_COV_MS_BLOCK_PDICT)
SFH_PDF_QUENCH_PDICT.update(SFH_PDF_QUENCH_COV_Q_BLOCK_PDICT)

SFH_PDF_QUENCH_BOUNDS_PDICT = SFH_PDF_FRAC_QUENCH_BOUNDS_PDICT.copy()
SFH_PDF_QUENCH_BOUNDS_PDICT.update(SFH_PDF_QUENCH_MU_BOUNDS_PDICT)
SFH_PDF_QUENCH_BOUNDS_PDICT.update(SFH_PDF_QUENCH_COV_MS_BLOCK_BOUNDS_PDICT)
SFH_PDF_QUENCH_BOUNDS_PDICT.update(SFH_PDF_QUENCH_COV_Q_BLOCK_BOUNDS_PDICT)

BOUNDING_VALS_PDICT = BOUNDING_MEAN_VALS_PDICT.copy()
BOUNDING_VALS_PDICT.update(BOUNDING_STD_VALS_PDICT.copy())
BOUNDING_VALS_PDICT.update(BOUNDING_RHO_VALS_PDICT.copy())

QseqParams = namedtuple("QseqParams", list(SFH_PDF_QUENCH_PDICT.keys()))
SFH_PDF_QUENCH_PARAMS = QseqParams(**SFH_PDF_QUENCH_PDICT)
SFH_PDF_QUENCH_PBOUNDS = QseqParams(**SFH_PDF_QUENCH_BOUNDS_PDICT)

_UPNAMES = ["u_" + key for key in QseqParams._fields]
QseqUParams = namedtuple("QseqUParams", _UPNAMES)

BoundingParams = namedtuple("BoundingParams", list(BOUNDING_VALS_PDICT.keys()))
BOUNDING_VALS = BoundingParams(**BOUNDING_VALS_PDICT)
