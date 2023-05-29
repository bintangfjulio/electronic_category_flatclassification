import torch
import torch.nn as nn

from transformers import BertModel

class BERT(nn.Module):
    def __init__(self, num_classes, bert_model, dropout, level=None, input_size=768, hidden_size=768):
        super(BERT, self).__init__()
        self.pretrained_bert = BertModel.from_pretrained(bert_model)
        self.dropout = nn.Dropout(dropout) 
        self.hidden_layer = nn.Linear(768, 768)
        self.level = level
        self.relu = nn.ReLU()

        if self.level is None:
           self.output_layer = nn.Linear(768, num_classes)

    def forward(self, input_ids):
        bert_output = self.pretrained_bert(input_ids=input_ids)
        last_hidden_state = bert_output[0]
        pooler = last_hidden_state[:, 0]
        pooled_output = self.relu(self.hidden_layer(pooler))
        pooler = self.dropout(pooler)

        if self.level is not None:
            return logits

        preds = self.output_layer(logits)
        
        return preds
