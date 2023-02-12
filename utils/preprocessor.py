import torch
import os
import re
import string
import multiprocessing
import requests
import pytorch_lightning as pl
import pandas as pd

from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from transformers import BertTokenizer
from torch.utils.data import TensorDataset, DataLoader
from utils.tree_creator import Tree_Creator

import sys
import pickle

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
    
    def preprocessor(self):
        if not os.path.exists(f"datasets/{self.method}_train_set.pt") and not os.path.exists(f"datasets/{self.method}_valid_set.pt") and not os.path.exists(f"datasets/{self.method}_test_set.pt"):
            print("\nPreprocessing Data...")
            max_length = self.get_max_length(self.dataset)
            train_set, test_set = self.train_test_split(self.dataset)
            self.preprocessing_data(set_queue=[train_set, test_set], max_length=max_length, method=self.method)
            print('[ Preprocessing Completed ]\n')
        
        print("\nLoading Data...")
        train_set = torch.load(f"datasets/{self.method}_train_set.pt")
        valid_set = torch.load(f"datasets/{self.method}_valid_set.pt")
        test_set = torch.load(f"datasets/{self.method}_test_set.pt")
        print('[ Loading Completed ]\n')

        return train_set, valid_set, test_set
    
    def get_max_length(self, dataset, extra_length=10):
        sentences_token = []
        
        for data in dataset.values.tolist():
            data = str(data[0]).split()
            sentences_token.append(data)

        token_length = [len(token) for token in sentences_token]
        max_length = max(token_length) + extra_length
        
        return max_length
    
    def preprocessing_data(self, set_queue, max_length, method): 
        section_parent_child, level_of_nodes_indexed = self.generate_hierarchy()
        parents_idx = {parent: index for index, parent in enumerate(section_parent_child.keys())}        
        section_by_hierarchy = [[] for i in range(len(parents_idx))]

        for queue, dataset in enumerate(set_queue):
            input_ids, binary_target, categorical_target = [], [], []

            for data in dataset.values.tolist():
                text = self.text_cleaning(str(data[0]))
                token = self.tokenizer(text=text, max_length=max_length, padding="max_length", truncation=True)  

                if method == 'flat':
                    last_node = data[3].split(" > ")[-1].lower()

                    flat_binary_label = [0] * len(level_of_nodes_indexed[3])
                    flat_binary_label[level_of_nodes_indexed[3][last_node]] = 1
                    
                    flat_categorical_label = level_of_nodes_indexed[3][last_node]

                    input_ids.append(token['input_ids'])
                    binary_target.append(flat_binary_label)
                    categorical_target.append(flat_categorical_label)

                elif method == 'section':
                    nodes = data[3].lower().split(" > ")

                    for depth, node in enumerate(nodes[:-1]):
                        child = nodes[depth + 1]
                        child_of_parent = list(section_parent_child[node])
                        child_idx = child_of_parent.index(child)

                        hierarchical_binary_label = [0] * len(child_of_parent)
                        hierarchical_binary_label[child_idx] = 1

                        hierarchical_categorical_label = child_idx
                        
                        parent_idx = parents_idx[node]

                        if 'input_ids' not in section_by_hierarchy[parent_idx]:
                            section_by_hierarchy[parent_idx] = {'input_ids': [], 'binary_target': [], 'categorical_target': []}
                        
                        section_by_hierarchy[parent_idx]['input_ids'].append(token['input_ids'])
                        section_by_hierarchy[parent_idx]['binary_target'].append(hierarchical_binary_label)
                        section_by_hierarchy[parent_idx]['categorical_target'].append(hierarchical_categorical_label)

                else:
                    pass
        
            if method == 'flat':
                if queue == 0:
                    train_set, valid_set = self.train_valid_split(input_ids, binary_target, categorical_target)
                    torch.save(train_set, f"datasets/{self.method}_train_set.pt")
                    torch.save(valid_set, f"datasets/{self.method}_valid_set.pt")
                    
                elif queue == 1:
                    input_ids = torch.tensor(input_ids)
                    binary_target = torch.tensor(binary_target)
                    categorical_target = torch.tensor(categorical_target)
                    test_set = TensorDataset(input_ids, binary_target, categorical_target)
                    torch.save(test_set, f"datasets/{self.method}_test_set.pt")

            elif method == 'level':
                pass

            elif method == 'section':
                pass
                # sectioned_dataset = []

                # for section in section_by_hierarchy:
                #     sectioned_input_ids = section['input_ids']
                #     sectioned_target = section['target']
                    
                #     train_set, valid_set, test_set = self.dataset_splitting(sectioned_input_ids, sectioned_target)
                #     sectioned_dataset.append([train_set, valid_set, test_set])

                # with open("datasets/hierarchical_dataset.pkl", "wb") as hierarchical_preprocessed:
                #     pickle.dump(sectioned_dataset, hierarchical_preprocessed, protocol=pickle.HIGHEST_PROTOCOL)

    def generate_hierarchy(self):
        section_parent_child = {}
        level_of_nodes = {}

        with open(self.hierarchy_tree, "r") as tree:
            for path in tree:
                nodes = path[:-1].lower().split(" > ")

                for depth, node in enumerate(nodes):
                    if depth > 0:
                        parent = nodes[depth - 1]
                        try:
                            section_parent_child[parent].add(node)
                        except:
                            section_parent_child[parent] = set()
                            section_parent_child[parent].add(node)

                level = len(nodes)
                last_node = nodes[-1]

                if level not in level_of_nodes:
                    level_of_nodes[level] = []

                level_of_nodes[level] += [last_node]

        level_of_nodes_indexed = {}

        for level, node_members in level_of_nodes.items():
            node_with_idx = {}

            for node_idx, node in enumerate(node_members):
                node_with_idx[node] = node_idx
            
            level_of_nodes_indexed[level] = node_with_idx

        return section_parent_child, level_of_nodes_indexed

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

    def train_test_split(self, dataset):
        dataset = dataset.sample(frac=1)
        dataset_size = dataset.shape[0]
        train_size = int(dataset_size * 0.7)

        train_set = dataset.iloc[:train_size, :]
        test_set = dataset.iloc[train_size:, :]

        train_set = pd.DataFrame(train_set)
        test_set = pd.DataFrame(test_set)

        return train_set, test_set

    def train_valid_split(self, input_ids, binary_target, categorical_target):
        input_ids = torch.tensor(input_ids)
        binary_target = torch.tensor(binary_target)
        categorical_target = torch.tensor(categorical_target)
        
        tensor_dataset = TensorDataset(input_ids, binary_target, categorical_target)

        train_size = round(len(tensor_dataset) * 0.9)
        valid_size = len(tensor_dataset) - train_size

        train_set, valid_set = torch.utils.data.random_split(tensor_dataset, [train_size, valid_size])

        return train_set, valid_set

    # def setup(self, stage=None):
    #     flat_train_set, flat_valid_set, flat_test_set = self.preprocessor()   
    #     if stage == "fit":
    #         self.flat_train_set = flat_train_set
    #         self.flat_valid_set = flat_valid_set
    #     elif stage == "test":
    #         self.flat_test_set = flat_test_set

    # def train_dataloader(self):
    #     return DataLoader(
    #         dataset=self.flat_train_set,
    #         batch_size=self.batch_size,
    #         shuffle=True,
    #         num_workers=multiprocessing.cpu_count()
    #     )

    # def val_dataloader(self):
    #     return DataLoader(
    #         dataset=self.flat_valid_set,
    #         batch_size=self.batch_size,
    #         shuffle=False,
    #         num_workers=multiprocessing.cpu_count()
    #     )

    # def test_dataloader(self):
    #     return DataLoader(
    #         dataset=self.flat_test_set,
    #         batch_size=self.batch_size,
    #         shuffle=False,
    #         num_workers=multiprocessing.cpu_count()
    #     )