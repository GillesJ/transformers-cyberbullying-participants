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

LANGUAGE = "en"
RAW_DATA_FP = f"../data/raw/fulldataset_{LANGUAGE}.tsv"
PROC_DATA_FP = f"../data/processed/{LANGUAGE}.tsv"

MODEL_SETTINGS = {
    "model_type": "roberta",
    "model_name": "roberta-large",
    "train_args": {
        "reprocess_input_data": True,
        "overwrite_output_dir": True,
        "num_train_epochs": 4,
        "n_gpu": 1,
        "save_steps": 16384,
        "save_model_every_epoch": False,
        "fp16": False,
        "manual_seed": RANDOM_STATE,
    },
}

MODEL_SETTINGS = {
    "model_type": "xlmroberta",
    "model_name": "xlm-roberta-large",
    "train_args": {
        "reprocess_input_data": True,
        "overwrite_output_dir": True,
        "num_train_epochs": 4,
        "n_gpu": 1,
        "save_steps": 16384,
        "save_model_every_epoch": False,
        "fp16": False,
        "manual_seed": RANDOM_STATE,
    },
}
