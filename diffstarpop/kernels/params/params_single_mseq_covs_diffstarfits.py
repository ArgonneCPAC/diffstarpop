from collections import OrderedDict, namedtuple

RHO_BOUNDS = (-0.3, 0.3)

SFH_PDF_QUENCH_MU_PDICT = OrderedDict(
    mean_ulgm_int=12.16,
    mean_ulgm_slp=0.11,
    mean_ulgy_int=-0.04,
    mean_ulgy_slp=-0.17,
    mean_ul_int=0.09,
    mean_ul_slp=0.22,
    mean_utau_int=1.36,
    mean_utau_slp=-12.57,
    mean_uqt_int=0.93,
    mean_uqt_slp=-0.19,
    mean_uqs_int=-0.36,
    mean_uqs_slp=0.68,
    mean_udrop_int=-1.84,
    mean_udrop_slp=-0.44,
    mean_urej_int=0.02,
    mean_urej_slp=-1.09,
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
    std_ulgm_int=0.22,
    std_ulgm_slp=0.11,
    std_ulgy_int=0.32,
    std_ulgy_slp=0.03,
    std_ul_int=0.32,
    std_ul_slp=0.10,
    std_utau_int=6.85,
    std_utau_slp=0.96,
    rho_ulgy_ulgm_int=0.001,
    rho_ulgy_ulgm_slp=0.001,
    rho_ul_ulgm_int=0.001,
    rho_ul_ulgm_slp=0.001,
    rho_ul_ulgy_int=0.001,
    rho_ul_ulgy_slp=0.001,
    rho_utau_ulgm_int=0.001,
    rho_utau_ulgm_slp=0.001,
    rho_utau_ulgy_int=0.001,
    rho_utau_ulgy_slp=0.001,
    rho_utau_ul_int=0.001,
    rho_utau_ul_slp=0.001,
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
    std_uqt_int=0.09,
    std_uqt_slp=0.02,
    std_uqs_int=0.55,
    std_uqs_slp=0.02,
    std_udrop_int=0.73,
    std_udrop_slp=-0.07,
    std_urej_int=1.12,
    std_urej_slp=-0.26,
    rho_uqs_uqt_int=0.001,
    rho_uqs_uqt_slp=0.001,
    rho_udrop_uqt_int=0.001,
    rho_udrop_uqt_slp=0.001,
    rho_udrop_uqs_int=0.001,
    rho_udrop_uqs_slp=0.001,
    rho_urej_uqt_int=0.001,
    rho_urej_uqt_slp=0.001,
    rho_urej_uqs_int=0.001,
    rho_urej_uqs_slp=0.001,
    rho_urej_udrop_int=0.001,
    rho_urej_udrop_slp=0.001,
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
    frac_quench_x0=12.07,
    frac_quench_k=2.87,
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
