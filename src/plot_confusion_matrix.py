import numpy as np
from sklearn.metrics import confusion_matrix
import itertools
import settings
from pathlib import Path
import matplotlib as mpl

mpl.use("pgf")

# I make my own newfig and savefig functions
def figsize(scale):
    fig_width_pt = 469.755  # Get this from LaTeX using \the\textwidth
    inches_per_pt = 1.0 / 72.27  # Convert pt to inch
    golden_mean = (np.sqrt(5.0) - 1.0) / 2.0  # Aesthetic ratio (you could change this)
    fig_width = fig_width_pt * inches_per_pt * scale  # width in inches
    fig_height = fig_width * golden_mean  # height in inches
    fig_size = [fig_width, fig_height]
    return fig_size


pgf_with_latex = {  # setup matplotlib to use latex for output
    "pgf.texsystem": "pdflatex",  # change this if using xetex or lautex
    "text.usetex": True,  # use LaTeX to write all text
    "font.family": "serif",
    "font.serif": [],  # blank entries should cause plots to inherit fonts from the document
    "font.sans-serif": [],
    "font.monospace": [],
    "axes.labelsize": 8,  # LaTeX default is 10pt font.
    "font.size": 8,
    "legend.fontsize": 8,  # Make the legend/label fonts a little smaller
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "figure.figsize": figsize(0.9),  # default fig size of 0.9 textwidth
    "pgf.preamble": [
        r"\usepackage[utf8x]{inputenc}",  # use utf8 fonts becasue your computer can handle it :)
        r"\usepackage[T1]{fontenc}",  # plots will be generated using this preamble
    ],
}

mpl.rcParams.update(pgf_with_latex)

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from ast import literal_eval


def plot_confusion_matrix(
    cm, classes, normalize=False, title="Confusion matrix", cmap=plt.cm.Blues
):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """

    if normalize:
        cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        cm_norm = np.round(cm_norm, decimals=3) * 100
        print("Normalized confusion matrix")
        print(cm_norm)

        plt.imshow(cm_norm, interpolation="nearest", cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        for i, j in itertools.product(range(cm_norm.shape[0]), range(cm_norm.shape[1])):
            thresh = cm_norm.max() / 2
            plt.text(
                j,
                i,
                "{}%\n($n={}$)".format(round(cm_norm[i, j], 1), cm[i, j]),
                horizontalalignment="center",
                verticalalignment="center",
                color="white" if cm_norm[i, j] > thresh else "black",
            )

    else:
        print("Confusion matrix, without normalization")
        print(cm)

        plt.imshow(cm, interpolation="nearest", cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            thresh = cm.max() / 2
            plt.text(
                j,
                i,
                cm[i, j],
                horizontalalignment="center",
                verticalalignment="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    plt.tight_layout()
    plt.ylabel("True label")
    plt.xlabel("Predicted label")


def savefig_confusion_matrix(y_true, y_pred, fp, title):
    # plot confusion matrix
    cm_norm_plot_fp = str(fp).replace(".pdf", "_norm.pdf")

    # Compute confusion matrix
    cnf_matrix = confusion_matrix(y_true, y_pred)
    # np.set_printoptions(precision=2)
    LABELMAP = dict(
        ((0, "Not Bullying"), (1, "Harasser"), (2, "Victim"), (3, "Bystander-Defender"))
    )
    class_names = [LABELMAP[val] for val in np.unique(y_true)]
    # Plot non-normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=class_names, title=title)
    plt.tight_layout()
    plt.savefig(fp)
    plt.gcf().clear()
    plt.clf()
    plt.close("all")

    # Plot normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(
        cnf_matrix, classes=class_names, normalize=True, title=title,
    )
    plt.tight_layout()
    plt.savefig(cm_norm_plot_fp)
    plt.gcf().clear()
    plt.clf()
    plt.close("all")

    print("Saved confusion matrix plot to {}".format(fp))


if __name__ == "__main__":

    dirs = [
        ("en_roberta-base_epochs-4_seq-256", "English RoBERTa"),
        ("nl_pdelobelle/robBERT-base_epochs-4_seq-256", "Dutch RoBERT"),
    ]  # best model dirs + titles for fig

    for (dirn, title) in dirs:
        holdout_dirp = Path(settings.MODEL_DIR) / f"{dirn}/dev-holdout/"

        # Load preds
        holdout_df = pd.read_csv(
            holdout_dirp / "testset_with_predictions.tsv",
            sep="\t",
            converters={"y_pred": literal_eval},
        )

        # Get the predictions
        y_true = np.array(holdout_df["labels"].to_list())
        y_pred = np.array(holdout_df["y_pred"].to_list())
        # go from probas to single label
        y_pred_max = np.argmax(y_pred, axis=-1)  # strongest label wins
        savefig_confusion_matrix(
            y_true, y_pred_max, holdout_dirp / "holdout_confusion_matrix.pdf", title
        )
