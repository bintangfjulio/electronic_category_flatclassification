import argparse

from utils.preprocessor import Preprocessor
from utils.trainer import Trainer

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['bert', 'bert-cnn', 'bert-bilstm', 'bert-lstm'], required=True, help='Model choices to fine tune')
    parser.add_argument('--method', choices=['flat', 'level', 'section'], required=True, help='Fine tuning method choices')
    parser.add_argument('--loss', choices=['binary', 'categorical'], required=True, help='Log loss types')
    args = parser.parse_args()
    config = vars(args)

    module = Preprocessor(batch_size=32, method=config['method']) 
    module.preprocessor()

    Trainer(module=module, model_path=config['model'], method=config['method'], loss=config['loss'])