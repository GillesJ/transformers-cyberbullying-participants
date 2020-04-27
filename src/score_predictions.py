#!/usr/bin/env python3
"""
Computes several scores given a model directory containing training-test run subdirs with a `testset_with_predictions.tsv` file.
Output:
- In each run subdir: a processed `testset_with_predictions.tsv` file with human-readable predictions per instance for error analysis.
- In each run subdir: scores with metrics (json)
- In root model dir: summary results.tsv file with holdout scores + averaged crossvalidation scores.
- In root model dir: class_scores.tsv overview of scores by class/type in holdout test and crossvalidation.

score_predictions
sentivent_event_sentence_classification 
12/12/19
Copyright (c) Gilles Jacobs. All rights reserved.  
"""
import settings_en as settings
import operator
from functools import reduce
import pandas as pd
from pprint import pprint
from ast import literal_eval
import numpy as np
from pathlib import Path
import json
from sklearn.metrics import (
    precision_recall_fscore_support,
    accuracy_score,
    roc_auc_score,
)


def average_fold_df(fold_df):
    type_cols = ["precision", "recall", "f1"]
    fold_types_df = fold_df[type_cols]
    fold_no_types_df = fold_df.drop(type_cols, axis=1)
    fold_avg_df = fold_no_types_df.mean(axis=0)
    type_avg_scores = []
    for c in type_cols:
        score_dicts = fold_types_df[c].to_list()
        avg = {
            key: np.mean([d.get(key) for d in score_dicts])
            for key in reduce(operator.or_, (d.keys() for d in score_dicts))
        }
        avg["metric"] = c
        type_avg_scores.append(avg)
    fold_types_avg_df = pd.DataFrame(type_avg_scores).set_index("metric").transpose()

    return fold_avg_df, fold_types_avg_df


def flatten_embedded_dict(dictionary):
    return {
        (outerKey, innerKey): values
        for outerKey, innerDict in dictionary.items()
        for innerKey, values in innerDict.items()
    }


def get_score(y_true, y_pred):
    scores = {}

    scores["accuracy"] = accuracy_score(y_true, y_pred)
    for avg in [None, "micro", "macro", "weighted", "samples"]:
        if avg:
            avg_suffix = f"_{avg}"
            try:
                (
                    scores[f"precision{avg_suffix}"],
                    scores[f"recall{avg_suffix}"],
                    scores[f"f1{avg_suffix}"],
                    _,
                ) = precision_recall_fscore_support(y_true, y_pred, average=avg)
            except:
                (
                    scores[f"precision{avg_suffix}"],
                    scores[f"recall{avg_suffix}"],
                    scores[f"f1{avg_suffix}"],
                ) = (None, None, None)
            try:
                scores[f"roc_auc{avg_suffix}"] = roc_auc_score(
                    y_true, y_pred, average=avg
                )
            except:
                scores[f"roc_auc{avg_suffix}"] = None
        else:
            p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred)
            scores["precision"] = p.tolist()
            scores["recall"] = r.tolist()
            scores["f1"] = f1.tolist()
            try:
                scores["roc_auc"] = roc_auc_score(y_true, y_pred)
            except:
                scores["roc_auc"] = None

    return scores


def make_score_summaries(model_dirp):
    # Load runs + paths
    run_pred_fps = list(model_dirp.rglob("fold*/testset_with_predictions.tsv")) + list(
        model_dirp.rglob("dev-holdout/testset_with_predictions.tsv")
    )

    all_scores = {}

    model_dirn = str(model_dirp.name)

    # Iterate over run dirs and process testset results + collect fold scores
    for testset_fp in run_pred_fps:

        run_name = testset_fp.parts[-2]
        print(
            f"-------\nProcessing {run_name.upper()} for {model_dirn.upper()}\n-------"
        )

        # Set output path
        testset_proc_fp = (
            testset_fp.parent
            / f"{testset_fp.stem}_processed{''.join(testset_fp.suffixes)}"
        )

        # Load preds
        testset_df = pd.read_csv(
            testset_fp, sep="\t", converters={"y_pred": literal_eval},
        )

        # Get the predictions
        y_true = np.array(testset_df["labels"].to_list())
        y_pred = np.array(testset_df["y_pred"].to_list())
        # go from probas to single label
        y_pred_max = np.argmax(y_pred, axis=-1)  # strongest label wins
        testset_df["y_pred_max"] = y_pred_max

        # Collect scores
        scores = get_score(y_true, y_pred_max)
        pprint(scores)

        all_scores[run_name] = scores

        # Get label counts in train and test
        # Load full dataset for comparative stats
        trainset_fp = testset_fp.parent / "trainset.tsv"
        trainset_df = pd.read_csv(trainset_fp, sep="\t",)

        train_labels = np.array(trainset_df["labels"].to_list())
        test_labels = np.array(testset_df["labels"].to_list())

        train_label_sum = np.bincount(train_labels)
        test_label_sum = np.bincount(test_labels, minlength=train_label_sum.shape[0])
        train_label_pct = np.divide(
            np.multiply(train_label_sum, 100), np.sum(train_label_sum)
        )
        test_label_pct = np.divide(
            np.multiply(test_label_sum, 100), np.sum(test_label_sum)
        )

        try:
            delta = train_label_pct - test_label_pct
        except:
            print("Train and test labels do not correspond in presence.")
            delta = None

        data = {
            "label_cnt_train": train_label_sum,
            "label_pct_train": train_label_pct,
            "label_cnt_test_true": test_label_sum,
            "label_pct_test_true": test_label_pct,
            "label_cnt_test_pred": np.bincount(
                y_pred_max, minlength=train_label_sum.shape[0]
            ),
            "test_train_pct_delta": delta,
            "f1": np.pad(
                scores["f1"],
                (0, train_label_sum.shape[0] - len(scores["f1"])),
                "constant",
            ),
            "recall": np.pad(
                scores["recall"],
                (0, train_label_sum.shape[0] - len(scores["f1"])),
                "constant",
            ),
            "precision": np.pad(
                scores["precision"],
                (0, train_label_sum.shape[0] - len(scores["f1"])),
                "constant",
            ),
        }
        label_df = pd.DataFrame(data)

        label_df.update(
            label_df.select_dtypes(include=np.number).applymap("{:,g}".format)
        )  # format human readable

        # write all scores as json, scores by label as tsv, and confusion matrix
        label_df.to_csv(testset_fp.parent / "label_scores.tsv", sep="\t")

        with open(testset_fp.parent / "scores.json", "wt") as scores_out:
            json.dump(scores, scores_out, indent=4, sort_keys=True)

    # isolate folds for mean + std in crossvalidation
    fold_scores = [all_scores[k] for k in all_scores if k != "dev-holdout"]
    fold_df = pd.DataFrame(fold_scores)

    # std + mean in one df
    xval_mean = fold_df.dropna(axis=1).mean()
    xval_std = fold_df.std(axis=0)

    summary_df = pd.DataFrame(all_scores)
    summary_df["xval_mean"] = xval_mean
    summary_df["xval_std"] = xval_std
    # write summaries
    summary_df.update(
        summary_df.select_dtypes(include=np.number).applymap("{:,g}".format)
    )
    summary_df.to_csv(model_dirp / "score_summary.tsv", sep="\t")

    print(f"{model_dirn.upper()} holdout and crossvalidation scores summary:")
    print(summary_df)

    return summary_df


if __name__ == "__main__":

    # 1. Set model path and load labels and predictions
    model_dirn = "en_roberta-base_epochs-4_seq-256"
    model_dirpath = Path(settings.MODEL_DIR) / "extra-test" / model_dirn
    print(model_dirpath)
    # 3.Score preds and write summary
    summary_df = make_score_summaries(model_dirpath)

    print(
        f"{model_dirn} f1_macro xval: {summary_df.at['f1_macro', 'xval_mean']} f1_macro holdout {summary_df.at['f1_macro', 'dev-holdout']}"
    )
