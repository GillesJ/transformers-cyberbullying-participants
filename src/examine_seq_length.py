from transformers import RobertaTokenizer, BertTokenizer
import pandas as pd
import numpy as np
from scipy import stats
import settings_nl as settings
from pathlib import Path
import matplotlib.pyplot as plt

# load devset text
data_fp = settings.PROC_DATA_FP
df = pd.read_csv(data_fp, sep="\t")
dev_df = df[df["previous_experiments_set"] == "holdin"]
text_df = dev_df[["text"]]

# load tokenizers used
tokenizers = {
    "en": [
        RobertaTokenizer.from_pretrained("roberta-base", unk_token="<unk>"),
        BertTokenizer.from_pretrained("bert-base-uncased", unk_token="<unk>"),
    ],
    "nl": [
        RobertaTokenizer.from_pretrained("pdelobelle/robBERT-base", unk_token="<unk>"),
        BertTokenizer.from_pretrained("bert-base-dutch-cased", unk_token="<unk>"),
    ],
}

for tokenizer in tokenizers[settings.LANGUAGE]:

    def tokenize_len(s):
        tokens = tokenizer.tokenize(s)
        f.write(" ".join(tokens) + "\n")
        # print(tokens)
        return len(tokens)

    # Check token length
    with open(
        Path(settings.MODEL_DIR) / f"{type(tokenizer).__name__}_tokens", "wt"
    ) as f:
        token_cnt = text_df.text.map(tokenize_len)
    print(f"Tokenizing all devset text with {type(tokenizer).__name__}.")
    token_counts = sorted(token_cnt.tolist(), reverse=True)
    # see the percentile at token count values in test experiments
    for max_seq_v in [128, 256]:
        print(
            f"Percentile of instances @ {max_seq_v} sequence token length: {stats.percentileofscore(token_counts, max_seq_v)}"
        )

    # see token counts at certain percentiles
    for pctl in [95, 99, 99.5, 99.9]:
        print(
            f"{pctl}th percentile: {np.percentile(token_counts, pctl)} sequence token length"
        )
    # hist = token_cnt.hist(bins=256)
    # plt.yscale("log")
    # plt.savefig(f"{type(tokenizer).__name__}_token_cnt_hist.pdf")
    # plt.clf()
