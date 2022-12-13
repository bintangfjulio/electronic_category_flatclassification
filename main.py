import pandas as pd
import os

from utils.tree_generator import Tree_Generator
from utils.preprocessor import Preprocessor
from utils.trainer import Trainer

if __name__ == "__main__":
    models = ['bert', 'bert-cnn']
    dataset = pd.read_csv('datasets/electronic_product_tokopedia.csv')
    num_classes = len(dataset['leaf'].drop_duplicates().values.tolist())
    hierarchy = 'datasets/labels_hierarchy.tree'

    if not os.path.exists(hierarchy):
        Tree_Generator(dataset=dataset)

    module = Preprocessor(batch_size=32, dataset=dataset, num_classes=num_classes, hierarchy=hierarchy) 
    module.preprocessor() # delete soon

    for model_path in models:
        Trainer(model_path=model_path, module=module, num_classes=num_classes, flat=True, hierarchy=False)