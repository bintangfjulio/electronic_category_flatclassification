from utils.tree_helper import Tree_Helper
from models.flat_bert_cnn import Flat_Trainer
from models.level_bert_cnn import Level_Trainer

class Trainer_Helper(object):
    def __init__(self, method, dataset, bert_model, seed, max_epochs, lr, dropout):
        tree = Tree_Helper(tree_file=f'datasets/{dataset}_hierarchy.tree')

        if method == 'flat':
            self.trainer = Flat_Trainer(tree=tree,
                                        bert_model=bert_model,
                                        seed=seed,
                                        max_epochs=max_epochs,
                                        lr=lr,
                                        dropout=dropout)

        elif method == 'level':
            self.trainer = Level_Trainer(tree=tree,
                                        bert_model=bert_model,
                                        seed=seed,
                                        max_epochs=max_epochs,
                                        lr=lr,
                                        dropout=dropout)

        elif method == 'section':
            pass

    def fit(self, datamodule):
        self.trainer.fit(datamodule=datamodule)

    def test(self, datamodule):
        self.trainer.test(datamodule=datamodule)
        