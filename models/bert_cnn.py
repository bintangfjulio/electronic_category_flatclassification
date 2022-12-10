import torch.nn as nn
import torch
import pytorch_lightning as pl
import torch.nn.functional as F

from sklearn.metrics import classification_report
from transformers import BertModel

class BERT_CNN(pl.LightningModule):
    def __init__(self, lr, num_classes, dropout=0.1, embedding_size=768, window_sizes=[1, 2, 3, 4, 5], filters_in=4, filters_out=32):
        super(BERT_CNN, self).__init__()
        self.bert = BertModel.from_pretrained('indolem/indobert-base-uncased', output_hidden_states=True)
        self.conv_layers = nn.ModuleList([nn.Conv2d(filters_in, filters_out, (window_size, embedding_size)) for window_size in window_sizes])
        self.lr = lr
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(len(window_sizes) * filters_out, num_classes)
        self.sigmoid = nn.Sigmoid()
        self.criterion = nn.BCELoss()

    def forward(self, input_ids):
        bert_output = self.bert(input_ids=input_ids)
        bert_hidden_states = bert_output[2]
        bert_hidden_states = torch.stack(bert_hidden_states, dim=1)
        bert_hidden_states = bert_hidden_state[:, -4:]

        pooler = [F.relu(conv_layer(bert_hidden_states)).squeeze(3) for conv_layer in self.conv_layers] 
        max_pooler = [F.max_pool1d(output, output.size(2)).squeeze(2) for output in pooler]  

        flatten = torch.cat(max_pooler, dim=1) 

        fully_connected = self.dropout(flatten)
        fully_connected = self.classifier(fully_connected)
        output = self.sigmoid(fully_connected)

        return output

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
        x_input_ids, target = test_batch

        output = self(input_ids=input_ids)
        loss = self.criterion(output.cpu(), target=target.float().cpu())

        preds = output.argmax(1).cpu()
        target = target.argmax(1).cpu()
        report = classification_report(target, preds, output_dict=True, zero_division=0)

        self.log_dict({'test_loss': loss, 'test_accuracy': report["accuracy"]}, prog_bar=True, on_epoch=True)

        return loss
