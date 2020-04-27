#!/usr/bin/env python3
"""
Language-specific settings for English pipeline.
Contains dataset location, model settings, etc.

settings.py
transformers-cyberbullying-participants 
3/24/20
Copyright (c) Gilles Jacobs. All rights reserved.  
"""
from sklearn.model_selection import StratifiedKFold

SPLIT_DIR = "../data/split"
MODEL_DIR = (
    "/home/gilles/shares/lt3_amica/transformers-cyberbullying-participants/models/"
)
RANDOM_STATE = 42

K_FOLDS = 5
CV = StratifiedKFold(n_splits=K_FOLDS, shuffle=False, random_state=RANDOM_STATE)

LANGUAGE = "en"
LABELMAP = dict(((0, "no_bullying"), (1, "harasser"), (2, "victim"), (3, "defender")))
RAW_DATA_FP = f"../data/raw/fulldataset_{LANGUAGE}.tsv"
PROC_DATA_FP = f"../data/processed/{LANGUAGE}.tsv"

MODEL_SETTINGS = {
    "model_type": "roberta",
    "model_name": "roberta-large",
    "train_args": {
        "max_seq_length": 256,
        "reprocess_input_data": True,
        "overwrite_output_dir": True,
        "num_train_epochs": 4,
        "train_batch_size": 64,
        "n_gpu": 4,
        "save_steps": 65536,
        "save_model_every_epoch": False,
        "fp16": True,
        "fp16_opt_level": "O1",
        "manual_seed": RANDOM_STATE,
        "do_lower_case": False,
        "use_early_stopping": True,
        "early_stopping_patience": 3,
        "early_stopping_delta": 0.01,
    },
}

# MODEL_SETTINGS = { # OOM on 3x V100
#     "model_type": "xlmroberta",
#     "model_name": "xlm-roberta-large-finetuned-conll02-dutch",
#     "train_args": {
#         "max_seq_length": 256,
#         "reprocess_input_data": True,
#         "overwrite_output_dir": True,
#         "num_train_epochs": 16,
#         "train_batch_size": 32,
#         "n_gpu": 3,
#         "save_steps": 65536,
#         "save_model_every_epoch": False,
#         "fp16": True,
#         "fp16_opt_level": "O1",
#         "manual_seed": RANDOM_STATE,
#     },
# }

# MODEL_SETTINGS = {
#     "model_type": "bert",
#     "model_name": "bert-base-dutch-cased",
#     "train_args": {
#         "max_seq_length": 256,
#         "reprocess_input_data": True,
#         "overwrite_output_dir": True,
#         "num_train_epochs": 4,
#         "train_batch_size": 32,
#         "n_gpu": 2,
#         "save_steps": 65536,
#         "save_model_every_epoch": False,
#         "fp16": True,
#         "fp16_opt_level": "O1",
#         "manual_seed": RANDOM_STATE,
#     },
# }

# MODEL_SETTINGS = {
#     "model_type": "xlnet",
#     "model_name": "xlnet-base-cased",
#     "train_args": {
#         "max_seq_length": 256,
#         "reprocess_input_data": True,
#         "overwrite_output_dir": True,
#         "num_train_epochs": 4,
#         "train_batch_size": 32,
#         "n_gpu": 2,
#         "save_steps": 65536,
#         "save_model_every_epoch": False,
#         "fp16": True,
#         "fp16_opt_level": "O1",
#         "manual_seed": RANDOM_STATE,
#     },
# }

# MODEL_SETTINGS = {
#     "model_type": "bert",
#     "model_name": "bert-base-uncased",
#     "train_args": {
#         "max_seq_length": 256,
#         "reprocess_input_data": True,
#         "overwrite_output_dir": True,
#         "num_train_epochs": 16,
#         "train_batch_size": 32,
#         "n_gpu": 16,
#         "save_steps": 65536,
#         "save_model_every_epoch": False,
#         "fp16": True,
#         "fp16_opt_level": "O1",
#         "manual_seed": RANDOM_STATE,
#         "do_lower_case": True,
#         "use_early_stopping": True,
#         "early_stopping_patience": 3,
#         "early_stopping_delta": 0.001,
#     },
# }
