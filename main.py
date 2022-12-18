import pandas as pd
import os
import argparse

from utils.tree_generator import Tree_Generator
from utils.preprocessor import Preprocessor
from utils.trainer import Trainer

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['bert', 'bert-cnn', 'bert-bilstm', 'bert-lstm'], required=True, help='Model choices to fine tune')
    parser.add_argument('--method', choices=['flat', 'hierarchy'], required=True, help='Fine tuning method choices')
    
    args = parser.parse_args()
    config = vars(args)
    
    model_path = config['model']
    method = config['method']

    dataset = pd.read_csv('datasets/electronic_product_tokopedia.csv')
    num_classes = len(dataset['leaf'].drop_duplicates().values.tolist())
    hierarchy_tree = 'datasets/labels_hierarchy.tree'

    if not os.path.exists(hierarchy_tree):
        Tree_Generator(dataset=dataset)

    module = Preprocessor(batch_size=32, dataset=dataset, num_classes=num_classes, hierarchy_tree=hierarchy_tree) 
    Trainer(model_path=model_path, module=module, num_classes=num_classes, method=method)
