#!/usr/bin/env python3
"""
Language-specific settings for English pipeline.
Contains dataset location, model settings, etc.

settings_en.py
transformers-cyberbullying-participants 
3/24/20
Copyright (c) Gilles Jacobs. All rights reserved.  
"""
from sklearn.model_selection import StratifiedKFold

SPLIT_DIR = "../data/split"
MODEL_DIR = (
    "/home/gilles/shares/lt3_sentivent/transformers-cyberbullying-participants/models/"
)
RANDOM_STATE = 42

K_FOLDS = 5
CV = StratifiedKFold(n_splits=K_FOLDS, shuffle=False, random_state=RANDOM_STATE)

LANGUAGE = "nl"
RAW_DATA_FP = f"../data/raw/fulldataset_{LANGUAGE}.tsv"
PROC_DATA_FP = f"../data/processed/{LANGUAGE}.tsv"
