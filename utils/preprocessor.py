import torch
import os
import re
import pytorch_lightning as pl
import multiprocessing
import string

from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from transformers import BertTokenizer
from torch.utils.data import TensorDataset, DataLoader

class Preprocessor(pl.LightningDataModule):
    def __init__(self, batch_size, dataset, hierarchy='datasets/label_hierarchy.tree'):
        super(Preprocessor, self).__init__()
        self.batch_size = batch_size
        self.dataset = dataset
        self.hierarchy = hierarchy
        self.stop_words = StopWordRemoverFactory().get_stop_words()
        self.stemmer = StemmerFactory().create_stemmer()
        self.tokenizer = BertTokenizer.from_pretrained('indolem/indobert-base-uncased')

    def setup(self, stage=None):
        train_set, valid_set, test_set = self.preprocessor()
        if stage == "fit":
            self.train_set = train_set
            self.valid_set = valid_set
        elif stage == "test":
            self.test_set = test_set

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_set,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=multiprocessing.cpu_count()
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.valid_set,
            batch_size=self.batch_size,
            num_workers=multiprocessing.cpu_count()
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.test_set,
            batch_size=self.batch_size,
            num_workers=multiprocessing.cpu_count()
        )
    
    def preprocessor(self):
        if os.path.exists("datasets/train_set.pt") and os.path.exists("datasets/valid_set.pt") and os.path.exists("datasets/test_set.pt"):
            print("\nLoading Data...")
            train_set = torch.load("datasets/train_set.pt")
            valid_set = torch.load("datasets/valid_set.pt")
            test_set = torch.load("datasets/test_set.pt")
            print('[ Loading Completed ]\n')

        else:
            print("\nPreprocessing Data...")
            train_set, valid_set, test_set = self.preprocessing_data(self.dataset)
            print('[ Preprocessing Completed ]\n')

        return train_set, valid_set, test_set
    
    def get_maxlength(self, dataset, extra_space=5):
        sentences_token = []
        for data in dataset.values.tolist():
            data = str(data[0]).split()
            sentences_token.append(data)

        token_length = [len(token) for token in sentences_token]
        max_length = max(token_length) + extra_space
        print("Max Length:", max_length)
        
        return max_length
      
    def generate_hierarchy(self):
        hierarchy_of_level = {}
        level_of_hierarchy = {}
        
        with open(self.hierarchy, "r") as raw_hierarchy:
            for row in raw_hierarchy:
                row = row[:-1].lower()
                category = row.split(" > ")[-1]
                level = len(row.split(" > "))
                
                if category not in hierarchy_of_level:
                    hierarchy_of_level[category] = level
                    
                    if level not in level_of_hierarchy:
                        level_of_hierarchy[level] = []
                        
                    level_of_hierarchy[level] += [category]
                    
        return hierarchy_of_level, level_of_hierarchy

    def preprocessing_data(self, dataset): 
        max_length = self.get_maxlength(dataset)
        # hierarchy_of_level, level_of_hierarchy = self.generate_hierarchy()
        
        x_input_ids, x_attention_mask, flat_target, hierarchy_target = [], [], [], []

        for data in dataset.values.tolist():
            product_name = self.data_cleaning(str(data[0])) 
            token = self.tokenizer(text=product_name,  
                                max_length=max_length, 
                                padding="max_length", 
                                truncation=True)  
            
            flat_label = data[3].split(" > ")[-1]
            flat_binary_label = [0] * len(level_of_hierarchy[3])
            flat_binary_label[level_of_hierarchy[3].index(flat_label.lower())] = 1     
            
#             hierarchy_binary_label = []
            
#             leaf_path = data[3]
#             for level, category in enumerate(leaf_path.split(" > "), 1):
#                 binary = [0] * len(level_of_hierarchy[level])
#                 binary[level_of_hierarchy[level].index(category.lower())] = 1
#                 hierarchy_binary_label.append(binary)
            
            x_input_ids.append(token['input_ids'])
            x_attention_mask.append(token['attention_mask'])
            flat_target.append(flat_binary_label)
#             hierarchy_target.append(hierarchy_binary_label)

        x_input_ids = torch.tensor(x_input_ids)
        x_attention_mask = torch.tensor(x_attention_mask)
        flat_target = torch.tensor(flat_target)
        # hierarchy_target = torch.tensor(hierarchy_target)
        
        tensor_dataset = TensorDataset(x_input_ids, x_attention_mask, flat_target)

        train_valid_size = round(len(tensor_dataset) * 0.8)
        test_size = len(tensor_dataset) - train_valid_size

        train_valid_set, test_set = torch.utils.data.random_split(
            tensor_dataset, [
                train_valid_size, test_size
            ]
        )

        train_size = round(len(train_valid_set) * 0.9)
        valid_size = len(train_valid_set) - train_size

        train_set, valid_set = torch.utils.data.random_split(
            train_valid_set, [
                train_size, valid_size
            ]
        )

        torch.save(train_set, "datasets/train_set.pt")
        torch.save(valid_set, "datasets/valid_set.pt")
        torch.save(test_set, "datasets/test_set.pt")

        return train_set, valid_set, test_set

    def data_cleaning(self, text):
        text = text.lower()
        text = re.sub('\n', ' ', text)
        text = re.sub(r'@\w+', '', text)
        text = re.sub(r'http\S+', '', text)
        text = text.translate(str.maketrans('', '', string.punctuation))
        text = re.sub("'", '', text)
        text = re.sub(r'\d+', '', text)
        text = ' '.join([word for word in text.split() if word not in self.stop_words])
        text = self.stemmer.stem(text.strip())

        return text
