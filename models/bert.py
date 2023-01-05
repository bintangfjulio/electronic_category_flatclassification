import torch.nn as nn
import pytorch_lightning as pl

from transformers import BertModel

class BERT(pl.LightningModule):
    def __init__(self, num_classes, dropout=0.1, input_size=768, hidden_size=768):
        super(BERT, self).__init__()
        self.pretrained_bert = BertModel.from_pretrained('indolem/indobert-base-uncased')
        self.hidden_layer = nn.Linear(input_size, hidden_size)
        self.output_layer = nn.Linear(hidden_size, num_classes)
        self.dropout = nn.Dropout(dropout)   
        self.tanh = nn.Tanh()

    def forward(self, input_ids):
        bert_output = self.pretrained_bert(input_ids=input_ids)
        bert_output_layer = bert_output[0]
        cls_hidden_layer = bert_output_layer[:, 0]
        pooled_output =  self.tanh(self.hidden_layer(cls_hidden_layer))
        output = self.output_layer(self.dropout(pooled_output))

        return output
