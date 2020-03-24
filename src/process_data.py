#!/usr/bin/env python3
"""
Process the raw data to data format used in pipeline.

process_data.py
transformers-cyberbullying-participants 
3/24/20
Copyright (c) Gilles Jacobs. All rights reserved.  
"""

import numpy as np
import pandas as pd
from pathlib import Path
import importlib.util


def in_set(row):
    if np.isnan(row["y_holdout_true"]):
        return "holdin"
    else:
        return "holdout"


def load_paths_settings(settings_fp):
    """
    Handload a settings file as settings
    :param settings_fp:
    :return:
    """
    mod_name = settings_fp.stem
    spec = importlib.util.spec_from_file_location(mod_name, settings_fp)
    settings = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(settings)
    raw_fp = settings.RAW_DATA_FP
    opt_fp = settings.PROC_DATA_FP
    return raw_fp, opt_fp


if __name__ == "__main__":

    for settings_fp in Path(".").glob("**/settings_*.py"):

        raw_fp, opt_fp = load_paths_settings(settings_fp)

        df = pd.read_csv(raw_fp, sep="\t", index_col="Unnamed: 0")

        # indicate in which split in first experiments instance is
        df["previous_experiments_set"] = df.apply(lambda row: in_set(row), axis=1)

        # make labels column
        df["labels"] = df["vectorfile_label"].astype(int)

        # drop unneeded columns
        df = df.drop(["y_holdout_true", "y_holdout_pred", "vectorfile_label"], axis=1)

        # write tsv
        print(f"Writing pre-processed data to {opt_fp}")
        df.to_csv(opt_fp, sep="\t", index=False)
        print("Pre-processing of raw data complete.")
