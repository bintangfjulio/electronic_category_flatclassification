from utils.flat_tuning import Level_FineTuning
from helpers.tree_helper import Tree_Helper

class Trainer_Helper(object):
    def __init__(self, method, tree_file, seed, device, max_epochs, lr, early_stop_patience):
        tree = Tree_Helper(tree_file=tree_file)

        if method == 'flat':
            pass

        elif method == 'level':
            self.trainer = Level_FineTuning(seed=seed, 
                                tree=tree,
                                device=device, 
                                max_epochs=max_epochs,
                                lr=lr, 
                                early_stop_patience=early_stop_patience)

        elif method == 'section':
            pass

    def fit(self, model, datamodule):
        self.trainer.fit(model=model, datamodule=datamodule)

    def test(self, model, datamodule):
        self.trainer.test(model=model, datamodule=datamodule)
