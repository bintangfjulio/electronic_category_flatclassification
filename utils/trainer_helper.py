from utils.tree_helper import Tree_Helper
from models.flat_bert_cnn import Flat_Trainer
from models.level_bert_cnn import Level_Trainer
from models.section_bert_cnn import Section_Trainer

class Trainer_Helper(object):
    def __init__(self, method, dataset, bert_model, seed, max_epochs, lr, dropout, patience):
        tree = Tree_Helper(tree_file=f'datasets/{dataset}_hierarchy.tree', dataset=self.dataset)

        if method == 'flat':
            self.trainer = Flat_Trainer(tree=tree,
                                        bert_model=bert_model,
                                        seed=seed,
                                        max_epochs=max_epochs,
                                        lr=lr,
                                        dropout=dropout,
                                        patience=patience)

        elif method == 'level':
            self.trainer = Level_Trainer(tree=tree,
                                        bert_model=bert_model,
                                        seed=seed,
                                        max_epochs=max_epochs,
                                        lr=lr,
                                        dropout=dropout,
                                        patience=patience)

        elif method == 'section':
            self.trainer = Section_Trainer(tree=tree,
                                        bert_model=bert_model,
                                        seed=seed,
                                        max_epochs=max_epochs,
                                        lr=lr,
                                        dropout=dropout,
                                        patience=patience)

    def fit(self, datamodule):
        self.trainer.fit(datamodule=datamodule)

    def test(self, datamodule):
        self.trainer.test(datamodule=datamodule)

    def create_graph(self):
        self.trainer.create_graph()