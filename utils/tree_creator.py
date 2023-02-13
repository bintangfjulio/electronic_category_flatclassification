import pandas as pd

class Tree_Creator(object):
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

        tree.to_csv('datasets/hierarchy_path.tree', header=None, index=False, encoding='utf-8')