import torch
import torch.nn as nn
import pytorch_lightning as pl

from transformers import BertModel

class BERT_LSTM(pl.LightningModule):
    def __init__(self, num_classes, bidirectional, dropout=0.1, input_size=768, hidden_size=768, num_layers=2):
        super(BERT_LSTM, self).__init__()
        self.pretrained_bert = BertModel.from_pretrained('indolem/indobert-base-uncased')  
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, bidirectional=bidirectional, batch_first=True)
        self.dropout = nn.Dropout(dropout) 
        self.bidirectional = bidirectional
        
        if bidirectional:
            self.output_layer = nn.Linear(hidden_size * 2, num_classes)    
        else: 
            self.output_layer = nn.Linear(hidden_size, num_classes)

    def forward(self, input_ids):
        bert_output = self.pretrained_bert(input_ids=input_ids)
        bert_output_layer = bert_output[0]
        
        _, (lstm_output_layer, _) = self.lstm(bert_output_layer)
        
        if self.bidirectional:
            output_layer_LEFT = lstm_output_layer[-2]
            output_layer_RIGHT = lstm_output_layer[-1]
            lstm_output = torch.cat([output_layer_LEFT, output_layer_RIGHT], dim=-1)   
        else:
            lstm_output = lstm_output_layer[-1]

        output = self.output_layer(self.dropout(lstm_output))
        
        return output
