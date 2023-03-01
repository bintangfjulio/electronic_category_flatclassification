import torch.nn as nn

from transformers import BertModel

class BERT(nn.Module):
    def __init__(self, num_classes, dropout=0.1, input_size=768, hidden_size=768):
        super(BERT, self).__init__()
        self.pretrained_bert = BertModel.from_pretrained('indolem/indobert-base-uncased')
        self.pooler_layer = nn.Linear(input_size, hidden_size)
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(dropout) 
        self.output_layer = nn.Linear(hidden_size, num_classes)  

    def forward(self, input_ids):
        bert_output = self.pretrained_bert(input_ids=input_ids)
        last_hidden_state = bert_output[0]
        cls_state = last_hidden_state[:, 0]
        pooler = self.tanh(self.pooler_layer(cls_state))
        preds = self.output_layer(self.dropout(pooler))

        return preds