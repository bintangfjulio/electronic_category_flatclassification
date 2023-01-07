import torch.nn as nn
import torch
import pytorch_lightning as pl
import torch.nn.functional as F

from transformers import BertModel

class BERT_CNN(pl.LightningModule):
    def __init__(self, num_classes, dropout=0.1, input_size=768, window_sizes=[1, 2, 3, 4, 5], in_channels=4, out_channels=32):
        super(BERT_CNN, self).__init__()
        self.pretrained_bert = BertModel.from_pretrained('indolem/indobert-base-uncased', output_hidden_states=True)
        self.convolutional_layers = nn.ModuleList([nn.Conv2d(in_channels, out_channels, (window_size, input_size)) for window_size in window_sizes])
        self.dropout = nn.Dropout(dropout)
        self.output_layer = nn.Linear(len(window_sizes) * out_channels, num_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_ids):
        bert_output = self.pretrained_bert(input_ids=input_ids)
        bert_hidden_layers = bert_output[2]
        bert_hidden_layers = torch.stack(bert_hidden_layers, dim=1)
        bert_hidden_layers = bert_hidden_layers[:, -4:]

        pooling_layer = [F.relu(convolutional_layer(bert_hidden_layers).squeeze(3)) for convolutional_layer in self.convolutional_layers] 
        max_pooling_layer = [F.max_pool1d(filtered_features, filtered_features.size(2)).squeeze(2) for filtered_features in pooling_layer]  

        flatten_layer = torch.cat(max_pooling_layer, dim=1) 
        fully_connected_layer = self.output_layer(self.dropout(flatten_layer))
        probabilities = self.sigmoid(fully_connected_layer)
        
        return probabilities
