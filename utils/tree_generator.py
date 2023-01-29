import pandas as pd

class Tree_Generator:
    def __init__(self, dataset):
        hierarchy = []
        
        for root in dataset['root'].drop_duplicates().values.tolist():
            hierarchy.append(root)

        for node in dataset['node'].drop_duplicates().values.tolist():
            hierarchy.append(node)

        for leaf in dataset['leaf'].drop_duplicates().values.tolist():
            hierarchy.append(leaf)

        tree = pd.DataFrame([*zip(hierarchy)])
        tree = tree.sort_values(tree.columns[0])

        tree.to_csv('datasets/labels_hierarchy.tree', header=None, index=False, encoding='utf-8')
