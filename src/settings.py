#!/usr/bin/env python3
"""
Settings shared between language-specific pipelines and models.

settings.py
transformers-cyberbullying-participants 
3/24/20
Copyright (c) Gilles Jacobs. All rights reserved.  
"""
from sklearn.model_selection import StratifiedKFold

SPLIT_DIR = "../data/split"

RANDOM_STATE = 42

K_FOLDS = 5
CV = StratifiedKFold(n_splits=K_FOLDS, shuffle=False, random_state=RANDOM_STATE)
