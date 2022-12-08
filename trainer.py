import pytorch_lightning as pl
import pandas as pd

from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from utils.preprocessor import Preprocessor
from models.indobert_cnn import IndoBERT_CNN

if __name__ == "__main__":
    pl.seed_everything(42, workers=True)
    
    dataset = pd.read_csv('datasets/product_tokopedia.csv')
    num_classes = len(dataset['leaf'].drop_duplicates().values.tolist())
    
    module = Preprocessor(batch_size=32, dataset=dataset) 
    model = IndoBERT_CNN(lr=2e-5, num_classes=num_classes)

    checkpoint_callback = ModelCheckpoint(dirpath='./checkpoints/flat_indobertcnn_results', monitor='val_loss')
    logger = TensorBoardLogger("logs", name="flat_indobertcnn_results")
    early_stop_callback = EarlyStopping(monitor='val_loss', 
                                        min_delta=0.00, 
                                        check_on_train_epoch_end=1, 
                                        patience=3)

    trainer = pl.Trainer(
        accelerator='gpu',
        max_epochs=100,
        default_root_dir="./checkpoints/flat_indobertcnn_results",
        callbacks = [checkpoint_callback, early_stop_callback],
        logger=logger,
        log_every_n_steps=5)

    trainer.fit(model, datamodule=module)
    trainer.test(model=model, datamodule=module, ckpt_path='best')
