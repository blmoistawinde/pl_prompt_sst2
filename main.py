import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import argparse
import torch
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch import nn, optim
from data import PromptDataModule
from model import PromptClassifierPL
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

def main(args):
    prompt_config = json.load(open(args.prompt_config_dir, encoding="utf-8"))
    template_text = prompt_config['template_text']
    label_words = prompt_config['label_words']
    model = PromptClassifierPL(template_text, label_words, **vars(args))
    data_module = PromptDataModule(args.bs, args.input_dir, model.template, model.tokenizer, model.wrapper_cls, args.max_seq_length)
    import ipdb; ipdb.set_trace()
    early_stop_callback = EarlyStopping(
        monitor='val_f1',
        patience=args.patience,
        mode="max"
    )
    
    trainer = pl.Trainer(gpus=1, callbacks=[early_stop_callback], val_check_interval=1.0, max_epochs=100, min_epochs=1, accumulate_grad_batches=args.accumulation, gradient_clip_val=args.gradient_clip_val, deterministic=True, log_every_n_steps=100)

    trainer.fit(model, data_module)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser = PromptClassifierPL.add_model_specific_args(parser)
    parser.add_argument("--input_dir", type=str, default="./sst2")
    parser.add_argument("--prompt_config_dir", type=str, default="./sst2/prompt_config.json")
    parser.add_argument("--bs", type=int, default=32)
    parser.add_argument("--max_seq_length", type=int, default=64)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--accumulation", type=int, default=2)
    parser.add_argument("--gradient_clip_val", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=666)
    args = parser.parse_args()
    # seed_everything(args.seed, workers=True)
    seed_everything(args.seed)
    main(args)