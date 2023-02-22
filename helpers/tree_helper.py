import os

class Tree_Helper(object):
    def __init__(self, tree_file, dataset=None):
        self.tree_file = tree_file
        self.dataset = dataset

        if not os.path.exists(tree_file):
            hierarchy_path = []

            for column_idx, column in enumerate(self.dataset.columns.tolist()):
                if column_idx > 0:
                    for row in self.dataset[column].drop_duplicates().values.tolist():
                        hierarchy_path.append(row)

            hierarchy_path.sort()

            with open(self.tree_file, "w") as tree_file:
                for path in hierarchy_path:
                    tree_file.write(path + "\n")

    def generate_hierarchy(self):
        section_parent_child = {}
        level_on_nodes = {}

        with open(self.tree_file, "r") as tree:
            for path in tree:
                nodes = path[:-1].lower().split(" > ")

                for level, node in enumerate(nodes):
                    if level > 0:
                        parent = nodes[level - 1]
                        try:
                            section_parent_child[parent].add(node)
                        except:
                            section_parent_child[parent] = set()
                            section_parent_child[parent].add(node)

                level = len(nodes) - 1
                last_node = nodes[-1]

                if level not in level_on_nodes:
                    level_on_nodes[level] = []

                level_on_nodes[level] += [last_node]

        set_root_section = {'root': set(level_on_nodes[0])}
        set_root_section.update(section_parent_child)

        section_on_idx = {}
        idx_on_section = {}

        for idx, (_, node_members) in enumerate(set_root_section.items()):
            idx_on_section[idx] = list(node_members)

            for node in node_members:
                section_on_idx[node] = idx

        level_on_nodes_indexed = {}

        for level, node_members in level_on_nodes.items():
            node_with_idx = {}

            for idx, node in enumerate(node_members):
                node_with_idx[node] = idx
            
            level_on_nodes_indexed[level] = node_with_idx

        return level_on_nodes_indexed, idx_on_section, section_on_idx
