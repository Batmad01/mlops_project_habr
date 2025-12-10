import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.optim import AdamW
from torchmetrics import F1Score, HammingDistance, Precision, Recall
from transformers import AutoModel


# Model
class RuBertpl(pl.LightningModule):
    def __init__(self, model_name, num_labels, lr=1e-5, weight_decay=1e-2):
        super().__init__()
        self.save_hyperparameters()

        self.model = AutoModel.from_pretrained(model_name)
        self.model.train()

        self.dropout = nn.Dropout(0.2)
        self.classifier = nn.Linear(self.model.config.hidden_size, num_labels)
        self.loss_fn = nn.BCEWithLogitsLoss()

        self.f1 = F1Score(task="multilabel", num_labels=num_labels, average="micro")
        self.prec = Precision(task="multilabel", num_labels=num_labels, average="micro")
        self.rec = Recall(task="multilabel", num_labels=num_labels, average="micro")
        self.hamming = HammingDistance(task="multilabel", num_labels=num_labels)

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        outputs = self.model(
            input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids
        )
        x = outputs.last_hidden_state[:, 0, :]
        x = self.dropout(x)
        return self.classifier(x)

    def training_step(self, batch, batch_idx):
        y = batch["labels"]
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        token_type_ids = batch["token_type_ids"]

        if token_type_ids is not None:
            logits = self(input_ids, attention_mask, token_type_ids)
        else:
            logits = self(input_ids, attention_mask)

        loss = self.loss_fn(logits, y)
        preds = torch.sigmoid(logits)

        self.log("train_loss", loss, prog_bar=True)
        self.log("train_f1", self.f1(preds, y.int()), prog_bar=True)
        self.log("train_precision", self.prec(preds, y.int()))
        self.log("train_recall", self.rec(preds, y.int()))
        self.log("train_hamming", self.hamming(preds, y.int()))

        return loss

    def validation_step(self, batch, batch_idx):
        y = batch["labels"]
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        token_type_ids = batch["token_type_ids"]

        if token_type_ids is not None:
            logits = self(input_ids, attention_mask, token_type_ids)
        else:
            logits = self(input_ids, attention_mask)

        loss = self.loss_fn(logits, y)
        preds = torch.sigmoid(logits)

        self.log("val_loss", loss, prog_bar=True)
        self.log("val_f1", self.f1(preds, y.int()), prog_bar=True)
        self.log("val_precision", self.prec(preds, y.int()))
        self.log("val_recall", self.rec(preds, y.int()))
        self.log("val_hamming", self.hamming(preds, y.int()))

        return loss

    def configure_optimizers(self):
        optimizer = AdamW(
            self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay
        )
        return optimizer
