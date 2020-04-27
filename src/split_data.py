#!/usr/bin/env python3
"""
Split the data and write a Dataframe with that split metadata
All split indices index to the original full dataset in processed/raw datadirs.

split_data.py
transformers-cyberbullying-participants 
3/24/20
Copyright (c) Gilles Jacobs. All rights reserved.  
"""
import importlib.util
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt


def load_paths_settings(settings_fp):
    """
    Handload a settings file as settings
    :param settings_fp:
    :return:
    """
    mod_name = settings_fp.stem
    spec = importlib.util.spec_from_file_location(mod_name, settings_fp)
    return spec


if __name__ == "__main__":

    # run for each language-specific settings file
    for lang_settings_fp in Path(".").glob("**/settings_*.py"):

        print(f"Loading settings file {lang_settings_fp.name}")
        # load language-specific settings module
        spec = load_paths_settings(lang_settings_fp)
        lang_settings = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(lang_settings)

        # load dataset into frame
        data_fp = lang_settings.PROC_DATA_FP
        print(f"Loading and splitting dataset {data_fp}")
        df = pd.read_csv(data_fp, sep="\t")

        # Collect all split data + metadata into a df
        split_df = pd.DataFrame(columns=["split_name", "train_idc", "eval_idc",])
        # Add holdin dev - holdout test split and save indices for reference to preprocessed full dataset

        dev_df = df[df["previous_experiments_set"] == "holdin"]
        holdout_df = df[df["previous_experiments_set"] == "holdout"]
        dev_idc = dev_df.index.to_numpy()
        holdout_idc = holdout_df.index.to_numpy()
        print(
            f"Holdin-holdout: {dev_idc.shape[0]} holdin inst. and {holdout_idc.shape[0]} holdout inst."
        )
        split_df = split_df.append(
            {
                "split_name": "dev-holdout",
                "train_idc": dev_idc.tolist(),
                "eval_idc": holdout_idc.tolist(),
            },
            ignore_index=True,
        )
        # Make KFolds and collect fold splits
        X = dev_df["text"].to_numpy()
        y = dev_df["labels"].to_numpy()

        for i, (dev_train_idc, dev_eval_idc) in enumerate(lang_settings.CV.split(X, y)):
            print(
                f"Fold {i}: {dev_train_idc.shape[0]} train inst. and {dev_eval_idc.shape[0]} eval inst."
            )
            # get back original full dataset IDCs before dev holdin - holdout test split
            train_idc = dev_idc[dev_train_idc]
            eval_idc = dev_idc[dev_eval_idc]
            # test that all dev_idc have been split correctly
            assert np.array_equal(
                np.sort(np.concatenate([train_idc, eval_idc])), dev_idc
            )
            # collect split metadata
            split_df = split_df.append(
                {
                    "split_name": f"fold_{i}",
                    "train_idc": train_idc.tolist(),
                    "eval_idc": eval_idc.tolist(),
                },
                ignore_index=True,
            )

        split_fp = (
            Path(lang_settings.SPLIT_DIR) / f"splits_{lang_settings.LANGUAGE}.csv"
        )
        print(f"Writing split metadata to {split_fp}")
        split_df.to_csv(split_fp, index=False)
