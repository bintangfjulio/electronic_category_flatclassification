import torch
import os
import re
import string
import multiprocessing
import requests
import pytorch_lightning as pl
import pandas as pd

from tqdm import tqdm
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from transformers import BertTokenizer
from torch.utils.data import TensorDataset, DataLoader
from utils.tree_creator import Tree_Creator

class Preprocessor(pl.LightningDataModule):
    def __init__(self, batch_size, method):
        super(Preprocessor, self).__init__()
        self.batch_size = batch_size
        self.method = method
        self.dataset = pd.read_csv('datasets/small_product_tokopedia.csv')
        self.hierarchy_tree = 'datasets/small_hierarchy.tree'
        self.stop_words = StopWordRemoverFactory().get_stop_words()
        self.stemmer = StemmerFactory().create_stemmer()
        self.tokenizer = BertTokenizer.from_pretrained('indolem/indobert-base-uncased')

        # url = 'https://github.com/bintangfjulio/product_categories_classification/releases/download/0.0/product_tokopedia.csv'
        # file = requests.get(url, allow_redirects=True)
        # open('datasets/product_tokopedia.csv', 'wb').write(file.content)
        # self.dataset = pd.read_csv('datasets/product_tokopedia.csv')

        # if not os.path.exists(self.hierarchy_tree):
        #     Tree_Creator(dataset=self.dataset)
    
    def preprocessor(self, level='all'):
        if not os.path.exists(f"datasets/{self.method}_level_{str(level)}_train_set.pt") and not os.path.exists(f"datasets/{self.method}_level_{str(level)}_valid_set.pt") and not os.path.exists(f"datasets/{self.method}_level_{str(level)}_test_set.pt"):
            print("\nPreprocessing Data...")
            max_length = self.get_max_length(self.dataset)
            # train_set, test_set = self.train_test_split(self.dataset)
            self.preprocessing_data(dataset=self.dataset, max_length=max_length, method=self.method, level=level)
            print('[ Preprocessing Completed ]\n')
        
        print("\nLoading Data...")
        train_set = torch.load(f"datasets/{self.method}_level_{str(level)}_train_set.pt")
        valid_set = torch.load(f"datasets/{self.method}_level_{str(level)}_valid_set.pt")
        test_set = torch.load(f"datasets/{self.method}_level_{str(level)}_test_set.pt")
        print('[ Loading Completed ]\n')

        return train_set, valid_set, test_set
    
    def get_max_length(self, dataset, extra_length=10):
        sentences_token = []
        
        for row in dataset.values.tolist():
            row = str(row[0]).split()
            sentences_token.append(row)

        token_length = [len(token) for token in sentences_token]
        max_length = max(token_length) + extra_length
        
        return max_length
    
    def preprocessing_data(self, dataset, max_length, method, level): 
        level_on_nodes_indexed, idx_on_section, section_on_idx = self.generate_hierarchy()
    
        # for queue, dataset in enumerate(set_queue):
        input_ids, binary_target, categorical_target = [], [], []

        # if queue == 0:
        #     print("\nOn Queue Train & Validation Set...")

        # elif queue == 1:
        #     print("\nOn Queue Test Set...")

        progress_preprocessing = tqdm(dataset.values.tolist())

        for row in progress_preprocessing:
            text = self.text_cleaning(str(row[0]))
            token = self.tokenizer(text=text, max_length=max_length, padding="max_length", truncation=True)  

            if method == 'flat':
                last_node = row[3].split(" > ")[-1].lower()

                flat_binary_label = [0] * len(level_on_nodes_indexed[2])
                flat_binary_label[level_on_nodes_indexed[2][last_node]] = 1
                
                flat_categorical_label = level_on_nodes_indexed[2][last_node]

                input_ids.append(token['input_ids'])
                binary_target.append(flat_binary_label)
                categorical_target.append(flat_categorical_label)
            
            elif method == 'level':
                node_on_level = row[3].split(" > ")[level].lower()
                member_on_level = level_on_nodes_indexed[level]
                node_idx = member_on_level[node_on_level]

                leveled_binary_label = [0] * len(member_on_level)
                leveled_binary_label[node_idx] = 1

                leveled_categorical_label = node_idx

                input_ids.append(token['input_ids'])
                binary_target.append(leveled_binary_label)
                categorical_target.append(leveled_categorical_label)

            elif method == 'section':
                nodes = row[3].lower().split(" > ")
                
                binary_encoded = {}
                categorical_encoded = {}

                for node in nodes:
                    section_idx = section_on_idx[node]
                    nodes_on_section = idx_on_section[section_idx]
                    node_idx = nodes_on_section.index(node)

                    sectioned_binary_label = [0] * len(nodes_on_section)
                    sectioned_binary_label[node_idx] = 1

                    sectioned_categorical_label = node_idx

                    binary_encoded[section_idx] = sectioned_binary_label
                    categorical_encoded[section_idx] = sectioned_categorical_label

                input_ids.append(token['input_ids'])
                binary_target.append(binary_encoded)
                categorical_target.append(categorical_encoded)
                
        if method == 'section':
            pass

        else:
            train_set, valid_set, test_set = self.dataset_splitting(input_ids, binary_target, categorical_target)
            torch.save(train_set, f"datasets/{self.method}_level_{str(level)}_train_set.pt")
            torch.save(valid_set, f"datasets/{self.method}_level_{str(level)}_valid_set.pt")
            torch.save(test_set, f"datasets/{self.method}_level_{str(level)}_test_set.pt")

    def generate_hierarchy(self):
        section_parent_child = {}
        level_on_nodes = {}

        with open(self.hierarchy_tree, "r") as tree:
            for path in tree:
                nodes = path[:-1].lower().split(" > ")

                # arrange section
                for level, node in enumerate(nodes):
                    if level > 0:
                        parent = nodes[level - 1]
                        try:
                            section_parent_child[parent].add(node)
                        except:
                            section_parent_child[parent] = set()
                            section_parent_child[parent].add(node)

                # arrange level
                level = len(nodes) - 1
                last_node = nodes[-1]

                if level not in level_on_nodes:
                    level_on_nodes[level] = []

                level_on_nodes[level] += [last_node]

        # arrange section
        set_root_section = {'root': set(level_on_nodes[0])}
        set_root_section.update(section_parent_child)

        section_on_idx = {}
        idx_on_section = {}

        for idx, (_, node_members) in enumerate(set_root_section.items()):
            idx_on_section[idx] = list(node_members)

            for node in node_members:
                section_on_idx[node] = idx

        # arrange level
        level_on_nodes_indexed = {}

        for level, node_members in level_on_nodes.items():
            node_with_idx = {}

            for idx, node in enumerate(node_members):
                node_with_idx[node] = idx
            
            level_on_nodes_indexed[level] = node_with_idx

        return level_on_nodes_indexed, idx_on_section, section_on_idx

    def text_cleaning(self, text):
        text = text.lower()
        text = re.sub(r"[^A-Za-z0-9(),!?\'\-`]", " ", text)
        text = re.sub('\n', ' ', text)
        text = re.sub(r'@\w+', '', text)
        text = re.sub(r'http\S+', '', text)
        text = text.translate(str.maketrans('', '', string.punctuation))
        text = re.sub("'", '', text)
        text = re.sub(r'\d+', '', text)
        text = ' '.join([word for word in text.split() if word not in self.stop_words and len(word) > 1])
        text = self.stemmer.stem(text.strip())

        return text

    # def train_test_split(self, dataset):
    #     dataset = dataset.sample(frac=1)
    #     dataset_size = dataset.shape[0]
    #     train_size = int(dataset_size * 0.8)

    #     train_set = dataset.iloc[:train_size, :]
    #     test_set = dataset.iloc[train_size:, :]

    #     train_set = pd.DataFrame(train_set)
    #     test_set = pd.DataFrame(test_set)

    #     return train_set, test_set

    def dataset_splitting(self, input_ids, binary_target, categorical_target):
        input_ids = torch.tensor(input_ids)
        binary_target = torch.tensor(binary_target)
        categorical_target = torch.tensor(categorical_target)
        
        tensor_dataset = TensorDataset(input_ids, binary_target, categorical_target)

        train_valid_size = round(len(tensor_dataset) * 0.8)
        test_size = len(tensor_dataset) - train_valid_size

        train_valid_set, test_set = torch.utils.data.random_split(tensor_dataset, [train_valid_size, test_size])

        train_size = round(len(train_valid_set) * 0.9)
        valid_size = len(train_valid_set) - train_size

        train_set, valid_set = torch.utils.data.random_split(train_valid_set, [train_size, valid_size])

        return train_set, valid_set, test_set
    
    def count_flat_classes(self):
        level_on_nodes_indexed, _, _ = self.generate_hierarchy()
        num_classes = len(level_on_nodes_indexed[2])

        return num_classes

    # Overriding Pytorch Lightning Data Loader for Flat Fine Tuning
    def setup(self, stage=None):
        flat_train_set, flat_valid_set, flat_test_set = self.preprocessor()   
        if stage == "fit":
            self.flat_train_set = flat_train_set
            self.flat_valid_set = flat_valid_set
        elif stage == "test":
            self.flat_test_set = flat_test_set

    def train_dataloader(self):
        return DataLoader(
            dataset=self.flat_train_set,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=multiprocessing.cpu_count()
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.flat_valid_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=multiprocessing.cpu_count()
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.flat_test_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=multiprocessing.cpu_count()
        )

    # Custom Data Loader for Level Fine Tuning 
    def level_dataloader(self, stage, level):
        level_train_set, level_valid_set, level_test_set = self.preprocessor(level=level) 
        
        if stage == 'fit':
            train_dataloader = DataLoader(dataset=level_train_set,
                                        batch_size=self.batch_size,
                                        shuffle=True,
                                        num_workers=multiprocessing.cpu_count())

            val_dataloader = DataLoader(dataset=level_valid_set,
                                        batch_size=self.batch_size,
                                        shuffle=False,
                                        num_workers=multiprocessing.cpu_count())

            return train_dataloader, val_dataloader

        elif stage == 'test':
            test_dataloader = DataLoader(dataset=level_test_set,
                                        batch_size=self.batch_size,
                                        shuffle=False,
                                        num_workers=multiprocessing.cpu_count())

            return test_dataloader