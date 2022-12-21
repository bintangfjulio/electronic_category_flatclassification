import torch
import torch.nn as nn
import pytorch_lightning as pl

from transformers import BertModel

class BERT_LSTM(pl.LightningModule):
    def __init__(self, num_classes, bidirectional, dropout=0.1, input_size=768, hidden_size=768, num_layers=2):
        super(BERT_LSTM, self).__init__()
        self.bert = BertModel.from_pretrained('indolem/indobert-base-uncased')  
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, bidirectional=bidirectional, batch_first=True)
        self.dropout = nn.Dropout(dropout) 
        self.bidirectional = bidirectional
        
        if bidirectional:
            self.fully_connected = nn.Linear(hidden_size * 2, num_classes)    
        else: 
            self.fully_connected = nn.Linear(hidden_size, num_classes)

    def forward(self, input_ids):
        bert_output = self.bert(input_ids=input_ids)
        bert_last_hidden_state = bert_output[0]
        
        _, (lstm_last_hidden_state, _) = self.lstm(bert_last_hidden_state)
        
        if self.bidirectional:
            last_hidden_state_LEFT = lstm_last_hidden_state[-2]
            last_hidden_state_RIGHT = lstm_last_hidden_state[-1]
            lstm_output = torch.cat([last_hidden_state_LEFT, last_hidden_state_RIGHT], dim=-1)   
        else:
            lstm_output = lstm_last_hidden_state[-1]

        output = self.fully_connected(self.dropout(lstm_output))
        
        return output
