import torch.nn as nn
import torch
import pytorch_lightning as pl

from sklearn.metrics import classification_report
from transformers import BertModel

class Flat_BERTLSTM(pl.LightningModule):
    def __init__(self, lr, num_classes, dropout=0.1, embedding_size=768, hidden_size=768, num_layers=2):
        super(Flat_BERTLSTM, self).__init__() 
        self.lr = lr
        self.num_classes = num_classes
        self.dropout = nn.Dropout(dropout)
        self.bert = BertModel.from_pretrained('indolem/indobert-base-uncased') 
        self.criterion = nn.BCELoss()

        self.lstm = nn.LSTM(input_size=embedding_size, 
                            hidden_size=hidden_size, 
                            num_layers=num_layers,
                            batch_first=True,
                            bidirectional=True)
        
        self.classifier = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, input_ids):
        bert_output = self.bert(input_ids=input_ids)
        bert_last_hidden_state = bert_ouput[0]

        _, (last_hidden_state, _) = self.lstm(bert_last_hidden_state) 
        
        last_hidden_state_left = last_hidden_state[-2]
        last_hidden_state_right = last_hidden_state[-1]
        last_hidden_state_output = torch.cat([last_hidden_state_left, last_hidden_state_right], dim=-1)
            
        output = self.dropout(last_hidden_state_output)
        output = self.classifier(output) 
        
        return output

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
