import argparse

from utils.preprocessor import Preprocessor
from helpers.trainer_helper import Trainer_Helper

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default='datasets/small_product_tokopedia.csv')
    parser.add_argument("--tree_file", type=str, default='datasets/hierarchy_path.tree')
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument('--method', choices=['flat', 'level', 'section'], required=True, help='Fine tuning method choices')
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max_epochs", type=int, default=50)
    parser.add_argument("--lr", type=int, default=2e-5)
    parser.add_argument("--early_stop_patience", type=int, default=5)
    parser.add_argument('--model', choices=['bert', 'bert-cnn', 'bert-bilstm', 'bert-lstm'], required=True, help='Model choices to fine tune')
    
    args = parser.parse_args()
    config = vars(args)

    datamodule = Preprocessor(dataset=config['dataset'], tree_file=config['tree_file'], batch_size=config['batch_size'], method=config['method']) 
    trainer = Trainer_Helper(method=config['method'], tree_file=config['tree_file'], seed=config['seed'], device=config['device'], max_epochs=config['max_epochs'], lr=config['lr'], early_stop_patience=config['early_stop_patience'])

    trainer.fit(model=config['model'], datamodule=datamodule)
    trainer.test(model=config['model'], datamodule=datamodule)