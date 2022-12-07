import torch.nn as nn
import torch
import pytorch_lightning as pl
import torch.nn.functional as F

from sklearn.metrics import classification_report
from transformers import BertModel

class IndoBERT_CNN(pl.LightningModule):
    def __init__(self, lr, num_classes, dropout=0.1, embedding_size=768, filters_in=4, filters_out=32):
        super(IndoBERT_CNN, self).__init__()
        self.lr = lr
        self.dropout = nn.Dropout(dropout)
        self.bert = BertModel.from_pretrained('indolem/indobert-base-uncased', output_hidden_states=True)
        
        conv_layers = 5
        self.conv1 = nn.Conv2d(filters_in, filters_out, (1, embedding_size))
        self.conv2 = nn.Conv2d(filters_in, filters_out, (2, embedding_size))
        self.conv3 = nn.Conv2d(filters_in, filters_out, (3, embedding_size))
        self.conv4 = nn.Conv2d(filters_in, filters_out, (4, embedding_size))
        self.conv5 = nn.Conv2d(filters_in, filters_out, (5, embedding_size))
        self.classifier = nn.Linear(conv_layers * filters_out, num_classes)
        
        self.sigmoid = nn.Sigmoid()
        self.criterion = nn.BCELoss()

    def forward(self, input_ids, attention_mask):
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        bert_hidden_state = bert_output[2]
        bert_hidden_state = torch.stack(bert_hidden_state, dim=1)
        bert_hidden_state = bert_hidden_state[:, -4:]
        
        pooler = [
            F.relu(self.conv1(bert_hidden_state).squeeze(3)),
            F.relu(self.conv2(bert_hidden_state).squeeze(3)),
            F.relu(self.conv3(bert_hidden_state).squeeze(3)),
            F.relu(self.conv4(bert_hidden_state).squeeze(3)),
            F.relu(self.conv5(bert_hidden_state).squeeze(3))
        ]
        
        pooler = [ F.max_pool1d(output, output.size(2)).squeeze(2) for output in pooler ]
        flatten = torch.cat(pooler, dim=1) 
        flatten = self.dropout(flatten)
        dense = self.classifier(flatten)
        output = self.sigmoid(dense)
        
        return output

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

        return optimizer
    
    def training_step(self, train_batch, batch_idx):
        x_input_ids, x_attention_mask, flat_target, hierarchy_target = train_batch

        output = self(input_ids=x_input_ids, attention_mask=x_attention_mask)
        loss = self.criterion(output.cpu(), target=flat_target.float().cpu())
        
        preds = output.argmax(1).cpu()
        flat_target = flat_target.argmax(1).cpu()
        report = classification_report(flat_target, preds, output_dict=True, zero_division=0)

        self.log_dict({'train_loss': loss, 'train_accuracy': report["accuracy"]}, prog_bar=True, on_epoch=True)

        return loss

    def validation_step(self, valid_batch, batch_idx):
        x_input_ids, x_attention_mask, flat_target, hierarchy_target = valid_batch

        output = self(input_ids=x_input_ids, attention_mask=x_attention_mask)
        loss = self.criterion(output.cpu(), target=flat_target.float().cpu())
        
        preds = output.argmax(1).cpu()
        flat_target = flat_target.argmax(1).cpu()
        report = classification_report(flat_target, preds, output_dict=True, zero_division=0)
        
        self.log_dict({'val_loss': loss, 'val_accuracy': report["accuracy"]}, prog_bar=True, on_epoch=True)

        return loss

    def test_step(self, test_batch, batch_idx):
        x_input_ids, x_attention_mask, flat_target, hierarchy_target = test_batch

        output = self(input_ids=x_input_ids, attention_mask=x_attention_mask)
        loss = self.criterion(output.cpu(), target=flat_target.float().cpu())
        
        preds = output.argmax(1).cpu()
        flat_target = flat_target.argmax(1).cpu()
        report = classification_report(flat_target, preds, output_dict=True, zero_division=0)

        self.log_dict({'test_loss': loss, 'test_accuracy': report["accuracy"]}, prog_bar=True, on_epoch=True)

        return loss
