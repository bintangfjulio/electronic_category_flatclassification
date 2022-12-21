import torch.nn as nn
import torch
import pytorch_lightning as pl
import torch.nn.functional as F

from transformers import BertModel

class BERT_CNN(pl.LightningModule):
    def __init__(self, num_classes, dropout=0.1, input_size=768, window_sizes=[1, 2, 3, 4, 5], in_channels=4, out_channels=32):
        super(BERT_CNN, self).__init__()
        self.bert = BertModel.from_pretrained('indolem/indobert-base-uncased', output_hidden_states=True)
        self.conv_layers = nn.ModuleList([nn.Conv2d(in_channels, out_channels, (window_size, input_size)) for window_size in window_sizes])
        self.dropout = nn.Dropout(dropout)
        self.fully_connected = nn.Linear(len(window_sizes) * out_channels, num_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_ids):
        bert_output = self.bert(input_ids=input_ids)
        bert_hidden_states = bert_output[2]
        bert_hidden_states = torch.stack(bert_hidden_states, dim=1)
        bert_hidden_states = bert_hidden_states[:, -4:]

        pooler = [F.relu(conv_layer(bert_hidden_states).squeeze(3)) for conv_layer in self.conv_layers] 
        max_pooled = [F.max_pool1d(output, output.size(2)).squeeze(2) for output in pooler]  

        flattened = torch.cat(max_pooled, dim=1) 
        fully_connected_layer = self.fully_connected(self.dropout(flattened))
        output = self.sigmoid(fully_connected_layer)

        return output
