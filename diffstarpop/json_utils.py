import numpy as np
import json
from .pdf_quenched import DEFAULT_SFH_PDF_QUENCH_PARAMS
from .pdf_mainseq import DEFAULT_SFH_PDF_MAINSEQ_PARAMS
from .pdf_model_assembly_bias_shifts import (
    DEFAULT_R_QUENCH_PARAMS,
    DEFAULT_R_MAINSEQ_PARAMS,
)

KEYS_PDF_Q = list(DEFAULT_SFH_PDF_QUENCH_PARAMS.keys())
KEYS_PDF_MS = list(DEFAULT_SFH_PDF_MAINSEQ_PARAMS.keys())
KEYS_R_Q = list(DEFAULT_R_QUENCH_PARAMS.keys())
KEYS_R_MS = list(DEFAULT_R_MAINSEQ_PARAMS.keys())

KEYS_ALL = KEYS_PDF_Q + KEYS_PDF_MS + KEYS_R_Q + KEYS_R_MS

N_PDF_Q = len(DEFAULT_SFH_PDF_QUENCH_PARAMS)
N_PDF_MS = len(DEFAULT_SFH_PDF_MAINSEQ_PARAMS)
N_R_Q = len(DEFAULT_R_QUENCH_PARAMS)
N_R_MS = len(DEFAULT_R_MAINSEQ_PARAMS)

DICT_NAMES = [
    "DEFAULT_SFH_PDF_QUENCH_PARAMS",
    "DEFAULT_SFH_PDF_MAINSEQ_PARAMS",
    "DEFAULT_R_QUENCH_PARAMS",
    "DEFAULT_R_MAINSEQ_PARAMS",
]


def print_default_dicts(dictionary, name):
    print(f"{name} = OrderedDict(")
    for key in dictionary.keys():
        print(f"    {key}={dictionary[key]:.3f},")
    print(")")


def print_all_default_dicts(dict_list):
    for i, dictionary in enumerate(dict_list):
        print_default_dicts(dictionary, DICT_NAMES[i])


def round_to_N(params, N=3):
    output = []
    for x in params:
        if x != 0.0:
            output.append(round(x, -(np.floor(np.log10(abs(x)))).astype(int) + N))
        else:
            output.append(0.0)
    return np.array(output)


def write_params_json(file_path, params):
    params = round_to_N(params, N=3)
    output = dict(zip(KEYS_ALL, params.astype(float)))
    with open(file_path, "w") as fp:
        json.dump(output, fp)


def load_params(file_path):
    with open(file_path, "r") as fp:
        params_dict_all = json.load(fp)
    all_params = np.array(list(params_dict_all.values()))

    pars_PDF_Q = all_params[0:N_PDF_Q]
    pars_PDF_MS = all_params[N_PDF_Q : N_PDF_Q + N_PDF_MS]
    pars_R_Q = all_params[N_PDF_Q + N_PDF_MS : N_PDF_Q + N_PDF_MS + N_R_Q]
    pars_R_MS = all_params[
        N_PDF_Q + N_PDF_MS + N_R_Q : N_PDF_Q + N_PDF_MS + N_R_Q + N_R_MS
    ]
    output = (
        all_params,
        dict(zip(KEYS_PDF_Q, pars_PDF_Q)),
        dict(zip(KEYS_PDF_MS, pars_PDF_MS)),
        dict(zip(KEYS_R_Q, pars_R_Q)),
        dict(zip(KEYS_R_MS, pars_R_MS)),
    )
    return output
