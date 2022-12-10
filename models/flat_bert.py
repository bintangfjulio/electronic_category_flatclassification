import torch.nn as nn
import torch
import pytorch_lightning as pl

from sklearn.metrics import classification_report
from transformers import BertModel

class Flat_BERT(pl.LightningModule):
    def __init__(self, lr, num_classes, dropout=0.1, embedding_size=768, hidden_size=768):
        super(Flat_BERT, self).__init__()
        self.bert = BertModel.from_pretrained('indolem/indobert-base-uncased')
        self.pre_classifier = nn.Linear(embedding_size, hidden_size)
        self.classifier = nn.Linear(hidden_size, num_classes)
        self.dropout = nn.Dropout(dropout)
        self.lr = lr      
        self.criterion = nn.BCEWithLogitsLoss()
        self.tanh = nn.Tanh()

    def forward(self, input_ids):
        bert_output = self.bert(input_ids=input_ids)
        bert_last_hiddenstate = bert_output[0]
        pooler = bert_last_hiddenstate[:, 0]
        pooler = self.pre_classifier(pooler)
        pooler = self.tanh(pooler)
        pooler = self.dropout(pooler)
        logits = self.classifier(pooler)

        return logits

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

        return optimizer

    def training_step(self, train_batch, batch_idx):
        input_ids, target = train_batch

        output = self(input_ids=input_ids)
        loss = self.criterion(output.cpu(), target=target.float().cpu())

        preds = output.argmax(1).cpu()
        target = target.argmax(1).cpu()
        report = classification_report(target, preds, output_dict=True, zero_division=0)

        self.log_dict({'train_loss': loss, 'train_accuracy': report["accuracy"]}, prog_bar=True, on_epoch=True)

        return loss

    def validation_step(self, valid_batch, batch_idx):
        input_ids, target = valid_batch

        output = self(input_ids=input_ids)
        loss = self.criterion(output.cpu(), target=target.float().cpu())

        preds = output.argmax(1).cpu()
        target = target.argmax(1).cpu()
        report = classification_report(target, preds, output_dict=True, zero_division=0)

        self.log_dict({'val_loss': loss, 'val_accuracy': report["accuracy"]}, prog_bar=True, on_epoch=True)

        return loss

    def test_step(self, test_batch, batch_idx):
        input_ids, target = test_batch

        output = self(input_ids=input_ids)
        loss = self.criterion(output.cpu(), target=target.float().cpu())

        preds = output.argmax(1).cpu()
        target = target.argmax(1).cpu()
        report = classification_report(target, preds, output_dict=True, zero_division=0)

        self.log_dict({'test_loss': loss, 'test_accuracy': report["accuracy"]}, prog_bar=True, on_epoch=True)

        return loss