import argparse

from utils.preprocessor import Preprocessor
from utils.trainer_helper import Trainer_Helper

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', choices=['flat', 'level', 'section'], required=True, help='Fine-tune method choices')
    parser.add_argument("--dataset", type=str, default='large')
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--bert_model", type=str, default='indolem/indobert-base-uncased')
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=5)
    config = vars(parser.parse_args())

    datamodule = Preprocessor(method=config['method'], dataset=config['dataset'], batch_size=config['batch_size'], bert_model=config['bert_model']) 
    trainer = Trainer_Helper(method=config['method'], dataset=config['dataset'], bert_model=config['bert_model'], seed=config['seed'], max_epochs=config['max_epochs'], lr=config['lr'], dropout=config['dropout'], patience=config['patience'])

    trainer.fit(datamodule=datamodule)
    trainer.test(datamodule=datamodule)
    trainer.create_graph()