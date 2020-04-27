from pathlib import Path
import pandas as pd
import numpy as np
from ast import literal_eval


def map_label(x):
    map = dict(((0, "not_bullying"), (1, "harasser"), (2, "victim"), (3, "defender")))
    return map[x]


def make_qualerr_file(fp):
    # Load preds
    testset_df = pd.read_csv(fp, sep="\t", converters={"y_pred": literal_eval},)

    # Get the predictions
    y_pred = np.array(testset_df["y_pred"].to_list())
    # go from probas to single label
    y_pred_max = np.argmax(y_pred, axis=-1)  # strongest label wins
    testset_df["y_pred_max"] = y_pred_max
    testset_df["error?"] = np.where(
        testset_df["labels"] != testset_df["y_pred_max"], "yes", "no"
    )
    out_df = testset_df[["id", "text", "annotation"]]
    out_df["label_true"] = testset_df["text_label"]
    out_df["label_pred"] = testset_df["y_pred_max"].apply(map_label)
    out_df["error"] = testset_df["error?"]
    print(out_df.head(5))
    out_fp = fp.replace(".tsv", "_error_analysis.tsv")
    out_df.to_csv(out_fp, sep="\t", index=False)


if __name__ == "__main__":

    fps = [
        "/home/gilles/shares/lt3_amica/transformers-cyberbullying-participants/models/nl_pdelobelle/robBERT-base_epochs-4_seq-256/dev-holdout/testset_with_predictions.tsv",
        "/home/gilles/shares/lt3_amica/transformers-cyberbullying-participants/models/en_roberta-base_epochs-4_seq-256/dev-holdout/testset_with_predictions.tsv",
    ]
    for fp in fps:
        make_qualerr_file(fp)
