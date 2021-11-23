import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch import nn, optim
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoModel
from transformers import AdamW, get_linear_schedule_with_warmup
from argparse import ArgumentParser
from sklearn.metrics import f1_score, precision_score, recall_score

from openprompt.plms import load_plm
from openprompt.prompts import ManualTemplate
from openprompt.prompts import ManualVerbalizer
from openprompt import PromptForClassification

class LightningInterface(pl.LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.best_f1 = 0.
        self.criterion = nn.CrossEntropyLoss()

    def training_step(self, batch, batch_nb, optimizer_idx=0):
        inputs = batch
        labels = inputs['label']
        y_hat = self(inputs)
        loss = self.criterion(y_hat, labels)
        tensorboard_logs = {'train_loss': loss}
        # self.log('lr', self.trainer.lr_schedulers[0]['scheduler'].get_last_lr()[0], on_step=True)
        # self.log('lr', self.optimizers().param_groups[0]['lr'], on_step=True)
        return {'loss': loss, 'log': tensorboard_logs}

    # def training_epoch_end(self, output) -> None:
    #     self.log('lr', self.hparams.lr)
    
    def validation_step(self, batch, batch_nb):
        inputs = batch
        labels = inputs['label']
        y_hat = self(inputs)
        yy, yy_hat = labels.detach().cpu().numpy(), y_hat.sigmoid().detach().cpu().numpy()
        return {'val_loss': self.criterion(y_hat, labels), "labels": yy, "probs": yy_hat}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        all_labels = np.concatenate([x['labels'] for x in outputs])
        all_probs = np.concatenate([x['probs'] for x in outputs])
        all_preds = all_probs.argmax(1)
        acc = np.mean(all_labels == all_preds)
        p = precision_score(all_labels, all_preds)
        r = recall_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds)
        self.best_f1 = max(self.best_f1, f1)
        # return {'val_loss': avg_loss, 'val_acc': avg_acc, 'hp_metric': self.best_acc}
        tensorboard_logs = {'val_loss': avg_loss, 'val_acc': acc, 'val_p': p, 'val_r': r, 'val_f1': f1, 'hp_metric': self.best_f1}
        # import pdb; pdb.set_trace()
        self.log_dict(tensorboard_logs)
        self.log("best_f1", self.best_f1, prog_bar=True, on_epoch=True)
        return {'val_loss': avg_loss, 'log': tensorboard_logs}

    def test_step(self, batch, batch_nb):
        inputs = batch
        labels = inputs['label']
        y_hat = self(inputs)
        yy, yy_hat = labels.detach().cpu().numpy(), y_hat.sigmoid().detach().cpu().numpy()
        return {'test_loss': self.criterion(y_hat, labels), "labels": yy, "probs": yy_hat}
    
    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        all_labels = np.concatenate([x['labels'] for x in outputs])
        all_probs = np.concatenate([x['probs'] for x in outputs])
        all_preds = all_probs.argmax(1)
        acc = np.mean(all_labels == all_preds)
        p = precision_score(all_labels, all_preds)
        r = recall_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds)
        return {'test_loss': avg_loss, 'test_acc': acc, 'test_p': p, 'test_r': r, 'test_f1': f1}

    @staticmethod
    def add_model_specific_args(parent_parser: ArgumentParser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        return parser


class PromptClassifierPL(LightningInterface):
    def __init__(self, template_text, label_words, model_class="bert", model_name_or_path="prajjwal1/bert-tiny", lr=2e-4, freeze_plm=False, **kwargs):
        super().__init__()
        self.freeze_plm = freeze_plm
        self.template_text = template_text
        self.model_class = model_class
        self.model_name_or_path = model_name_or_path
        plm, self.tokenizer, self.model_config, self.wrapper_cls = load_plm(model_class, model_name_or_path)
        # TODO: currently support using manual template and manual verbalizer
        self.template = ManualTemplate(tokenizer=self.tokenizer, text=template_text)
        self.verbalizer = ManualVerbalizer(self.tokenizer, num_classes=len(label_words), label_words=label_words)
        self.model = PromptForClassification(plm=plm,template=self.template, verbalizer=self.verbalizer, freeze_plm=self.freeze_plm)
        del plm
        self.lr = lr
        self.save_hyperparameters()
        print(self.hparams)

    def forward(self, x):
        x = self.model(x)
        return x

    @staticmethod
    def add_model_specific_args(parent_parser: ArgumentParser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser = LightningInterface.add_model_specific_args(parser)
        parser.add_argument("--lr", type=float, default=2e-4)
        parser.add_argument("--model_class", type=str, default="bert")
        parser.add_argument("--model_name_or_path", type=str, default="prajjwal1/bert-tiny")
        parser.add_argument("--freeze_plm", action="store_true")
        return parser

    def configure_optimizers(self):
        no_decay = ['bias', 'LayerNorm.weight']
        # it's always good practice to set no decay to biase and LayerNorm parameters
        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.lr)
        return optimizer
