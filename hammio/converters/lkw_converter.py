import os
import glob
import itertools

import pandas as pd

DEFAUT_FORMAT_STRING = "r{r}a{a}occ{occ1}f{forcing}tdimer{tdimer}tinter{tinter}.json"
DEFAULT_PARAMETERS = (
    ("r", (2.0,)),
    ("a", (0.0,0.1,0.2,0.4,0.6,0.8)),
    ("occ", (1.0,1.5)),
    ("f", (0.5,)),
    ("tdimer", (-1,2,4)),
    ("tinter", (-1,1,2))
)

def load_lkw_directory(dirname, parameters = None,
                       supplemental_values = None):
        parameters = parameters or DEFAULT_PARAMETERS
        param_names = [p for (p, _) in parameters]
        param_vals = [v for (_, v) in parameters]
        supplemental_values = supplemental_values or {}
        format_string = "".join([_ + "{" + _ + "}" for _ in param_names]) + ".json"
        dfs = []
        for i, v in enumerate(itertools.product(*param_vals)):
            print(i)
            fn = os.path.join(dirname, format_string.format(**dict(zip(param_names, v))))
            if not os.path.isfile(fn): continue
            dftmp = pd.read_json(fn)
            for n, v in zip(param_names, v):
                dftmp[n] = v
            dfs.append(dftmp)
        df = pd.concat(dfs)
        for n, v in supplemental_values.items():
            df[n] = v
        return df
