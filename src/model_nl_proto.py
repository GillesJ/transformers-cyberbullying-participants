#!/usr/bin/env python3
"""
model_nl_proto
transformers-cyberbullying-participants 
3/26/20
Copyright (c) Gilles Jacobs. All rights reserved.  
"""
from pathlib import Path
import settings_nl as settings
import pandas as pd
import numpy as np
import json, re
from tqdm import tqdm_notebook
from uuid import uuid4

## Torch Modules
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader


# ## Mount Drive into Colab
# from google.colab import drive
#
# drive.mount("/content/drive")

## PyTorch Transformer
from transformers import RobertaModel, RobertaTokenizer
from transformers import RobertaForSequenceClassification, RobertaConfig
from transformers import BertTokenizer, BertModel


## Check if Cuda is Available
print(torch.cuda.is_available())

## Test loading of RobBERT
cache_dir = Path(settings.MODEL_DIR) / "robbertcache/"
tokenizer = RobertaTokenizer.from_pretrained(
    "pdelobelle/robBERT-base", cache_dir=cache_dir
)
model = RobertaForSequenceClassification.from_pretrained(
    "pdelobelle/robBERT-base", cache_dir=cache_dir
)

## Test loading of BERTje
cache_dir = Path(settings.MODEL_DIR) / "bertjecache/"
tokenizer = BertTokenizer.from_pretrained("bert-base-dutch-cased", cache_dir=cache_dir)
model = BertModel.from_pretrained("bert-base-dutch-cased", cache_dir=cache_dir)

## Data prep
dataset_path = "drive/My Drive/Datasets/2017-06-custom-intent-engines/"

dataset = pd.DataFrame(columns=["utterance", "label"])
for intent in [
    "AddToPlaylist",
    "BookRestaurant",
    "GetWeather",
    "PlayMusic",
    "RateBook",
    "SearchCreativeWork",
    "SearchScreeningEvent",
]:
    with open(
        dataset_path + intent + "/train_" + intent + ".json", encoding="cp1251"
    ) as data_file:
        data = json.load(data_file)
    print("Class: {}, # utterances: {}".format(intent, len(data[intent])))
    texts = []
    for i in range(len(data[intent])):
        text = ""
        for j in range(len(data[intent][i]["data"])):
            text += data[intent][i]["data"][j]["text"]
        dataset = dataset.append(
            {"utterance": text, "label": intent}, ignore_index=True
        )
dataset.tail()

label_to_ix = {}
for label in dataset.label:
    for word in label.split():
        if word not in label_to_ix:
            label_to_ix[word] = len(label_to_ix)
label_to_ix

## Load RobertaConfig
config = RobertaConfig.from_pretrained("roberta-base")
config.num_labels = len(list(label_to_ix.values()))
config

## Load model and tokenizer
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
model = RobertaForSequenceClassification(config)


def prepare_features(
    seq_1,
    max_seq_length=300,
    zero_pad=False,
    include_CLS_token=True,
    include_SEP_token=True,
):
    ## Tokenzine Input
    tokens_a = tokenizer.tokenize(seq_1)

    ## Truncate
    if len(tokens_a) > max_seq_length - 2:
        tokens_a = tokens_a[0 : (max_seq_length - 2)]
    ## Initialize Tokens
    tokens = []
    if include_CLS_token:
        tokens.append(tokenizer.cls_token)
    ## Add Tokens and separators
    for token in tokens_a:
        tokens.append(token)

    if include_SEP_token:
        tokens.append(tokenizer.sep_token)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    ## Input Mask
    input_mask = [1] * len(input_ids)
    ## Zero-pad sequence lenght
    if zero_pad:
        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
    return torch.tensor(input_ids).unsqueeze(0), input_mask


msg = "My dog is cute!"
prepare_features(msg)

## Dataset Loader Classes
class Intents(Dataset):
    def __init__(self, dataframe):
        self.len = len(dataframe)
        self.data = dataframe

    def __getitem__(self, index):
        utterance = self.data.utterance[index]
        label = self.data.label[index]
        X, _ = prepare_features(utterance)
        y = label_to_ix[self.data.label[index]]
        return X, y

    def __len__(self):
        return self.len


## amke splits
train_size = 0.8
train_dataset = dataset.sample(frac=train_size, random_state=200).reset_index(drop=True)
test_dataset = dataset.drop(train_dataset.index).reset_index(drop=True)
training_set = Intents(train_dataset)
testing_set = Intents(test_dataset)
