#!/usr/bin/env python3
"""
Crossvalidation text-classification pipeline full run-through.

xval_en.py
transformers-cyberbullying-participants 
3/24/20
Copyright (c) Gilles Jacobs. All rights reserved.  
"""
import settings
import settings_en
import numpy as np
import pandas as pd
from pathlib import Path
from ast import literal_eval

if __name__ == "__main__":

    # load dataset
    df = pd.read_csv(settings_en.PROC_DATA_FP, sep="\t")
    # drop useless columns
    df = df.drop(labels=["previous_experiments_set"], axis=1)
    # load splits metadata
    split_df = pd.read_csv(
        Path(settings.SPLIT_DIR) / "splits_en.csv",
        converters={"train_idc": literal_eval, "eval_idc": literal_eval},
    )

    pass
