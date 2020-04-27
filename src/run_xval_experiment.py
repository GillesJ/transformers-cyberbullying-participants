#!/usr/bin/env python3
"""
Crossvalidation text-classification pipeline full run-through.

run_xval_experiment.py
transformers-cyberbullying-participants 
3/24/20
Copyright (c) Gilles Jacobs. All rights reserved.  
"""
import settings
import numpy as np
import socket
import pandas as pd
from pathlib import Path
from ast import literal_eval
from simpletransformers.classification import ClassificationModel
import json
import operator
from functools import reduce


def train_eval(train_df, eval_df, output_dirp):
    """
    Train and eval test a model
    :param train_df:
    :param eval_df:
    :param output_dirp:
    :return:
    """
    print(train_df.head())

    # Define model
    model = ClassificationModel(
        settings.MODEL_SETTINGS["model_type"],
        settings.MODEL_SETTINGS["model_name"],
        num_labels=4,
        args=settings.MODEL_SETTINGS["train_args"],
    )

    # Write train and eval
    Path(output_dirp).mkdir(parents=True, exist_ok=True)
    train_df.to_csv(Path(output_dirp) / "trainset.tsv", sep="\t", index=False)
    eval_df.to_csv(Path(output_dirp) / "testset.tsv", sep="\t", index=False)

    # # Reload train and eval for testing
    # train_df = pd.read_csv(Path(output_dirp) / "trainset.tsv", sep="\t", converters={"labels": literal_eval})
    # eval_df = pd.read_csv(Path(output_dirp) / "testset.tsv", sep="\t", converters={"labels": literal_eval})

    # Set tensorflow_dir in model args to run dir
    model.args["tensorboard_dir"] = Path(output_dirp) / "tensorboard/"
    model.args["cache_dir"] = (
        Path(output_dirp) / "cache/"
    )  # to ensure no weights are shared
    model.args["output_dir"] = output_dirp  # is redundant

    # Train the model
    print(f"Training model with args: {model.args}")
    model.train_model(train_df, output_dir=output_dirp)

    # Evaluate the model on eval set
    result, model_outputs, _ = model.eval_model(eval_df)

    # Write model result and outputs
    eval_df["y_pred"] = model_outputs.tolist()
    predictions_fp = Path(output_dirp) / "testset_with_predictions.tsv"
    eval_df.to_csv(predictions_fp, sep="\t", index=False)

    with open(Path(output_dirp) / "result.json", "wt") as result_out:
        json.dump(result, result_out)

    return result, model_outputs


if __name__ == "__main__":

    # set experiment output dir
    m_name = settings.MODEL_SETTINGS["model_name"]
    n_epochs = settings.MODEL_SETTINGS["train_args"]["num_train_epochs"]
    seq_length = settings.MODEL_SETTINGS["train_args"]["max_seq_length"]
    modelname = f"{settings.LANGUAGE}_{m_name}_epochs-{n_epochs}_seq-{seq_length}"
    experiment_dirp = Path(settings.MODEL_DIR) / modelname

    # load dataset
    df = pd.read_csv(settings.PROC_DATA_FP, sep="\t")
    # drop useless columns
    df = df.drop(labels=["previous_experiments_set"], axis=1)

    # edit the labels so they are 0...n as required by simpletransformers
    df["labels_original"] = df["labels"]
    df["labels"] = df["labels"].astype(int) + 1

    # load splits metadata
    split_df = pd.read_csv(
        Path(settings.SPLIT_DIR) / f"splits_{settings.LANGUAGE}.csv",
        converters={"train_idc": literal_eval, "eval_idc": literal_eval},
    )

    print(
        f"Cross-validating across {settings.K_FOLDS} folds with model:\n{settings.MODEL_SETTINGS}"
    )

    experiment_df = split_df  # experiment_df collects all metadata of split runs

    for i, row in experiment_df.iterrows():
        # split the data
        split_name = row["split_name"]
        train_idc = row["train_idc"]
        eval_idc = row["eval_idc"]
        train_df = df.iloc[train_idc]
        eval_df = df.iloc[eval_idc]

        # set split run output path
        run_dirp = experiment_dirp / split_name

        # add run metadata to experiment_df
        experiment_df.at[i, "run_dirp"] = f"{socket.gethostname()}:{run_dirp}"

        # run split train-eval
        print(
            f"{split_name.upper()}: {train_df.shape[0]} train inst. and {eval_df.shape[0]} eval inst."
        )
        result, model_outputs = train_eval(train_df, eval_df, run_dirp)

        # collect result
        print(f"{split_name.upper()}: {result}")
        for k, v in result.items():
            experiment_df.at[i, k] = v

    # collect results
    results_df = experiment_df[["split_name", "run_dirp"] + list(result.keys())]
    results_df = results_df.set_index("split_name")
    # average fold results
    print("--------------------")
    for k in result.keys():
        fold_mean = results_df[results_df.index.str.match("fold")][k].mean()
        results_df.at["all_fold_mean", k] = fold_mean
        print(f"Crossvalidation {k}: {fold_mean}")
        print(f"Holdout {k}: {results_df.loc['dev-holdout', k]}")

    # Write experiment results
    results_fp = experiment_dirp / "results.tsv"
    results_df.to_csv(results_fp, sep="\t")

    # write model settings
    with open(experiment_dirp / "model_settings.json", "wt") as ms_out:
        json.dump(settings.MODEL_SETTINGS, ms_out)

    print(
        f"Crossvalidation and holdout testing finished. All results and metadata in {experiment_dirp}"
    )
