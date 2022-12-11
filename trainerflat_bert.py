import pytorch_lightning as pl
import pandas as pd

from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from utils.flat_preprocessor import Flat_Preprocessor
from models.flat_bert import Flat_BERT

if __name__ == "__main__":
    pl.seed_everything(1234, workers=True)
    
    dataset = pd.read_csv('datasets/product_tokopedia.csv')
    target_names = dataset['leaf'].drop_duplicates().values.tolist()
    
    module = Flat_Preprocessor(batch_size=32, dataset=dataset, target_names=target_names) 
    model = Flat_BERT(lr=2e-5, num_classes=len(target_names))

    checkpoint_callback = ModelCheckpoint(dirpath='./checkpoints/flat_bert_results', monitor='val_loss')
    logger = TensorBoardLogger("logs", name="flat_bert_results")
    early_stop_callback = EarlyStopping(monitor='val_loss', min_delta=0.00, check_on_train_epoch_end=1, patience=3)

    trainer = pl.Trainer(
        accelerator='gpu',
        max_epochs=30,
        default_root_dir="./checkpoints/flat_bert_results",
        callbacks = [checkpoint_callback, early_stop_callback],
        logger=logger,
        deterministic=True)

    trainer.fit(model, datamodule=module)
    trainer.test(model=model, datamodule=module, ckpt_path='best')
