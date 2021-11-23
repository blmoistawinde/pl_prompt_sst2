import os
import re
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch import nn, optim
from scipy.io import loadmat
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import pipeline
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from collections import defaultdict
from openprompt.data_utils import InputExample
from openprompt import PromptDataLoader

class PromptDataset(Dataset):
    def __init__(self, input_dir, split="train"):
        assert split in {"train", "dev", "test"}
        self.input_dir = input_dir
        self.data = []
        input_dir2 = os.path.join(input_dir, split)
        for i, fname in enumerate(os.listdir(input_dir2)):
            label = int(fname[-5])   # "xxx_0.txt"/"xxx_1.txt"
            text = open(os.path.join(input_dir2, fname), encoding="utf-8").read()
            input_example = InputExample(text_a = text, label=label, guid=f"{split}_{i}")
            self.data.append(input_example)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int):
        return self.data[index]

    def __iter__(self):
        return iter(self.data)

class PromptDataModule(pl.LightningDataModule):
    def __init__(self, bs, input_dir, template, tokenizer, tokenizer_wrapper_class, max_seq_length=256, truncate_method='tail'):
        super().__init__()
        self.bs = bs
        self.input_dir = input_dir
        self.template = template
        self.tokenizer = tokenizer 
        self.tokenizer_wrapper_class = tokenizer_wrapper_class
        self.max_seq_length = max_seq_length
        self.truncate_method = truncate_method
    
    def setup(self, stage):
        if stage == "fit":
            self.train_set = PromptDataset(self.input_dir, "train")
            self.val_set = PromptDataset(self.input_dir, "dev")
        elif stage == "test":
            self.test_set = PromptDataset(self.input_dir, "test")

    def train_dataloader(self):
        return PromptDataLoader(dataset=self.train_set, template=self.template, tokenizer=self.tokenizer, tokenizer_wrapper_class=self.tokenizer_wrapper_class, max_seq_length=self.max_seq_length, batch_size=self.bs, shuffle=True, teacher_forcing=False, predict_eos_token=False, truncate_method=self.truncate_method)

    def val_dataloader(self):
        return PromptDataLoader(dataset=self.val_set, template=self.template, tokenizer=self.tokenizer, tokenizer_wrapper_class=self.tokenizer_wrapper_class, max_seq_length=self.max_seq_length, batch_size=self.bs, shuffle=False, teacher_forcing=False, predict_eos_token=False, truncate_method=self.truncate_method)

    def test_dataloader(self):
        return PromptDataLoader(dataset=self.test_set, template=self.template, tokenizer=self.tokenizer, tokenizer_wrapper_class=self.tokenizer_wrapper_class, max_seq_length=self.max_seq_length, batch_size=self.bs, shuffle=False, teacher_forcing=False, predict_eos_token=False, truncate_method=self.truncate_method)
