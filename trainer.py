import pytorch_lightning as pl
import pandas as pd

from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from utils.preprocessor import Preprocessor
from models.bert_cnn import BERT_CNN

if __name__ == "__main__":
    pl.seed_everything(1234, workers=True)
    
    dataset = pd.read_csv('datasets/product_tokopedia.csv')
    num_classes = len(dataset['leaf'].drop_duplicates().values.tolist())
    
    module = Preprocessor(batch_size=32, dataset=dataset) 
    model = BERT_CNN(lr=2e-5, num_classes=num_classes)

    checkpoint_callback = ModelCheckpoint(dirpath='./checkpoints/flat_bertcnn_results', monitor='val_loss')
    logger = TensorBoardLogger("logs", name="flat_bertcnn_results")
    early_stop_callback = EarlyStopping(monitor='val_loss', 
                                        min_delta=0.00, 
                                        check_on_train_epoch_end=1, 
                                        patience=10)

    trainer = pl.Trainer(
        accelerator='gpu',
        max_epochs=100,
        default_root_dir="./checkpoints/flat_bertcnn_results",
        callbacks = [checkpoint_callback, early_stop_callback],
        logger=logger
        deterministic=True)

    trainer.fit(model, datamodule=module)
    trainer.test(model=model, datamodule=module, ckpt_path='best')
