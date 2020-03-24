#!/usr/bin/env python3
"""
Language-specific settings for English pipeline.
Contains dataset location, model settings, etc.

settings_en.py
transformers-cyberbullying-participants 
3/24/20
Copyright (c) Gilles Jacobs. All rights reserved.  
"""
LANGUAGE = "en"
RAW_DATA_FP = f"../data/raw/fulldataset_{LANGUAGE}.tsv"
PROC_DATA_FP = f"../data/processed/{LANGUAGE}.tsv"

MODEL_DIRP = f"../models/{LANGUAGE}/"
