import pandas as pd

class Tree_Generator:
    def __init__(self, dataset):
        hierarchy = []

        dataset = dataset[['root', 'node', 'leaf']]
        dataset = dataset.drop_duplicates()

        target_root = dataset['root'].drop_duplicates().values.tolist()
        target_node = dataset['node'].drop_duplicates().values.tolist()

        for root in target_root:
            for node in target_node:
                for data in dataset.values.tolist():
                    hierarchy.append(data[0])
                    if data[0] == root:
                        if data[1] not in hierarchy:
                            hierarchy.append(data[1])
                    if data[1] == node:
                        if data[2] not in hierarchy:
                            hierarchy.append(data[2])

        tree = pd.DataFrame([*zip(hierarchy)])
        tree = tree.drop_duplicates()
        tree = tree.sort_values(tree.columns[0])

        tree.to_csv('datasets/labels_hierarchy.tree', header=None, index=False, encoding='utf-8')