import torch
import torch.nn as nn
import pytorch_lightning as pl

from transformers import BertModel

class BERT_BiLSTM(pl.LightningModule):
    def __init__(self, num_classes, dropout=0.1, embedding_size=768, hidden_size=768, num_layers=2):
        super(BERT_BiLSTM, self).__init__()
        self.bert = BertModel.from_pretrained('indolem/indobert-base-uncased')  
        self.lstm = nn.LSTM(embedding_size, hidden_size, num_layers, bidirectional=True, batch_first=True)
        self.dropout = nn.Dropout(dropout) 
        self.fully_connected = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, input_ids):
        bert_output = self.bert(input_ids=input_ids)
        bert_last_hidden_state = bert_output[0]
        hidden_state_without_cls = bert_last_hidden_state[:, 1:]
        
        _, (lstm_last_hidden_state, _) = self.lstm(hidden_state_without_cls)

        last_hidden_state_LEFT = lstm_last_hidden_state[-2]
        last_hidden_state_RIGHT = lstm_last_hidden_state[-1]
        lstm_output = torch.cat([last_hidden_state_LEFT, last_hidden_state_RIGHT], dim=-1)

        logits = self.fully_connected(self.dropout(lstm_output))
        
        return logits
