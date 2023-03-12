import argparse

from utils.preprocessor import Preprocessor
from helpers.trainer_helper import Trainer_Helper

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=['small', 'large'], required=True, help='Dataset size to fine tuning')
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument('--method', choices=['flat', 'level', 'section'], required=True, help='Fine tuning method choices')
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--early_stop_patience", type=int, default=5)
    parser.add_argument('--model', choices=['bert', 'bert-cnn', 'bert-bilstm', 'bert-lstm'], required=True, help='Model choices to fine tune')
    
    args = parser.parse_args()
    config = vars(args)

    datamodule = Preprocessor(dataset=config['dataset'], batch_size=config['batch_size'], method=config['method']) 
    trainer = Trainer_Helper(method=config['method'], dataset=config['dataset'], seed=config['seed'], max_epochs=config['max_epochs'], lr=config['lr'], early_stop_patience=config['early_stop_patience'])

    trainer.fit(model=config['model'], datamodule=datamodule)
    trainer.test(model=config['model'], datamodule=datamodule)
