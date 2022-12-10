import torch.nn as nn
import torch
import pytorch_lightning as pl

from sklearn.metrics import classification_report
from transformers import BertModel

class Flat_BERTLSTM(pl.LightningModule):
    def __init__(self, lr, num_classes, dropout=0.1, bert_embedding_size=768, lstm_hidden_size=1024, lstm_num_layers=2):
        super(Flat_BERTLSTM, self).__init__() 
        self.lr = lr
        self.num_classes = num_classes
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()    
        self.criterion = nn.BCELoss() 

        self.bert = BertModel.from_pretrained('indolem/indobert-base-uncased') 
        self.lstm = nn.LSTM(input_size=bert_embedding_size,
                            hidden_size=lstm_hidden_size,
                            num_layers=lstm_num_layers)

        self.relu_dimension = [1024, 256, 128, 64] 
        self.max_sequent_length = 128
        self.hidden_to_dense = nn.Linear(self.bert_hidden_size + self.lstm_hidden_size, self.relu_dimension[0])
        
        modules = []

        for i in range(len(self.relu_dimension) - 1):
            modules.append(nn.Linear(self.relu_dimension[i], self.relu_dimension[i + 1]))
            modules.append(nn.ReLU())
            modules.append(nn.Dropout(dropout))

        dense_to_label = nn.Linear(self.relu_dimension[-1], self.num_classes)
        modules.append(dense_to_label)

        self.classifier = nn.Sequential(*modules)

