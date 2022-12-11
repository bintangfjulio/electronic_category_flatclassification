import torch
import os
import re
import pytorch_lightning as pl
import multiprocessing
import string
import pandas as pd

from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from transformers import BertTokenizer
from torch.utils.data import TensorDataset, DataLoader

class Flat_Preprocessor(pl.LightningDataModule):
    def __init__(self, batch_size, dataset, target_names):
        super(Flat_Preprocessor, self).__init__()
        self.batch_size = batch_size
        self.dataset = dataset
        self.target_names = target_names
        self.stop_words = StopWordRemoverFactory().get_stop_words()
        self.stemmer = StemmerFactory().create_stemmer()
        self.tokenizer = BertTokenizer.from_pretrained('indolem/indobert-base-uncased')

    def setup(self, stage=None):
        train_flat_set, valid_flat_set, test_flat_set = self.preprocessor()
        if stage == "fit":
            self.train_flat_set = train_flat_set
            self.valid_flat_set = valid_flat_set
        elif stage == "test":
            self.test_flat_set = test_flat_set

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_flat_set,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=multiprocessing.cpu_count()
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.valid_flat_set,
            batch_size=self.batch_size,
            num_workers=multiprocessing.cpu_count()
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.test_flat_set,
            batch_size=self.batch_size,
            num_workers=multiprocessing.cpu_count()
        )
    
    def preprocessor(self):
        if os.path.exists("datasets/train_flat_set.pt") and os.path.exists("datasets/valid_flat_set.pt") and os.path.exists("datasets/test_flat_set.pt"):
            print("\nLoading Data...")
            train_flat_set = torch.load("datasets/train_flat_set.pt")
            valid_flat_set = torch.load("datasets/valid_flat_set.pt")
            test_flat_set = torch.load("datasets/test_flat_set.pt")
            print('[ Loading Completed ]\n')

        else:
            print("\nPreprocessing Data...")
            train_flat_set, valid_flat_set, test_flat_set = self.preprocessing_data(self.dataset, self.target_names)
            print('[ Preprocessing Completed ]\n')

        return train_flat_set, valid_flat_set, test_flat_set

    def balancing_data(self, dataset, target_names):
        labels = target_names

        name = []
        root = []
        node = []
        leaf = []

        for i, label in enumerate(labels):
            length = 0

            for data in dataset.values.tolist():
                if str(data[3]) == label:
                    if length < 2000:
                        name.append(data[0])
                        root.append(data[1])
                        node.append(data[2])
                        leaf.append(data[3])
                        length += 1

            if length != 2000:
                for i in range(length):
                    name.pop()
                    root.pop()
                    node.pop()
                    leaf.pop()

        balanced_dataset = pd.DataFrame({'nama': name, 'root': root, 'node': node, 'leaf': leaf})

        return balanced_dataset
    
    def get_maxlength(self, dataset, extra_space=5):
        sentences_token = []
        for data in dataset.values.tolist():
            data = str(data[0]).split()
            sentences_token.append(data)

        token_length = [len(token) for token in sentences_token]
        max_length = max(token_length) + extra_space
        print("Max Length:", max_length)
        
        return max_length

    def preprocessing_data(self, dataset, target_names): 
        dataset = self.balancing_data(dataset, target_names)

        label_idx = dataset['leaf'].unique().tolist()
        dataset['leaf'] = dataset['leaf'].map(lambda x: label_idx.index(x))

        max_length = self.get_maxlength(dataset)

        input_ids, target = [], []

        for data in dataset.values.tolist():
            name = self.data_cleaning(str(data[0])) 

            binary_label = [0] * len(target_names)
            binary_label[int(data[3])] = 1

            token = self.tokenizer(text=name, max_length=max_length, padding="max_length", truncation=True)  
            
            input_ids.append(token['input_ids'])
            target.append(binary_label)

        input_ids = torch.tensor(input_ids)
        target = torch.tensor(target)
        
        tensor_dataset = TensorDataset(input_ids, target)

        train_valid_size = round(len(tensor_dataset) * 0.8)
        test_size = len(tensor_dataset) - train_valid_size

        train_valid_flat_set, test_flat_set = torch.utils.data.random_split(tensor_dataset, [train_valid_size, test_size])

        train_size = round(len(train_valid_flat_set) * 0.9)
        valid_size = len(train_valid_flat_set) - train_size

        train_flat_set, valid_flat_set = torch.utils.data.random_split(train_valid_flat_set, [train_size, valid_size])

        torch.save(train_flat_set, "datasets/train_flat_set.pt")
        torch.save(valid_flat_set, "datasets/valid_flat_set.pt")
        torch.save(test_flat_set, "datasets/test_flat_set.pt")

        return train_flat_set, valid_flat_set, test_flat_set

    def data_cleaning(self, text):
        text = text.lower()
        text = re.sub('\n', ' ', text)
        text = re.sub(r'@\w+', '', text)
        text = re.sub(r'http\S+', '', text)
        text = text.translate(str.maketrans('', '', string.punctuation))
        text = re.sub("'", '', text)
        text = re.sub(r'\d+', '', text)
        text = ' '.join([word for word in text.split() if word not in self.stop_words])
        text = text.strip()
        text = self.stemmer.stem(text)

        return text
