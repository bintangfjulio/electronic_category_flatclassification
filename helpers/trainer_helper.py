from utils.level_finetuning import Level_FineTuning
from utils.flat_finetuning import Flat_FineTuning
from helpers.tree_helper import Tree_Helper

class Trainer_Helper(object):
    def __init__(self, method, dataset, seed, max_epochs, lr, early_stop_patience):
        tree = Tree_Helper(tree_file=f'datasets/{dataset}_hierarchy.tree')

        if method == 'flat':
            self.trainer = Flat_FineTuning(seed=seed, 
                                        tree=tree,
                                        max_epochs=max_epochs,
                                        lr=lr, 
                                        early_stop_patience=early_stop_patience)

        elif method == 'level':
            self.trainer = Level_FineTuning(seed=seed, 
                                        tree=tree,
                                        max_epochs=max_epochs,
                                        lr=lr, 
                                        early_stop_patience=early_stop_patience)

        elif method == 'section':
            pass

    def fit(self, model, datamodule):
        self.trainer.fit(model=model, datamodule=datamodule)

    def test(self, model, datamodule):
        self.trainer.test(model=model, datamodule=datamodule)