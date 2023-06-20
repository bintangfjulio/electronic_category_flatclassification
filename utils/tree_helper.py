class Tree_Helper(object):
    def __init__(self, tree_file):
        self.tree_file = tree_file
        self.level_on_nodes = None
        self.idx_on_section = None 
        self.section_on_idx = None
        self.section_parent_child = None
            
    def create_tree_file(self, dataset):
        hierarchy_path = []

        for column_idx, column in enumerate(dataset.columns.tolist()):
            if column_idx > 0:
                for row in dataset[column].drop_duplicates().values.tolist():
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

        section_idx = list(idx_on_section.keys())
        
        for section in section_idx:
            idx_on_section[section].sort()

        for level in range(len(level_on_nodes)):
            level_on_nodes[level].sort()

        self.level_on_nodes= level_on_nodes
        self.idx_on_section = idx_on_section
        self.section_on_idx = section_on_idx
        self.section_parent_child = set_root_section
    
    def get_hierarchy(self):
        return self.level_on_nodes, self.idx_on_section, self.section_on_idx, self.section_parent_child