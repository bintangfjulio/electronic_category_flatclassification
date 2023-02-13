import torch
import pytorch_lightning as pl
import torch.nn as nn

from models.bert import BERT
from models.bert_cnn import BERT_CNN
from models.bert_lstm import BERT_LSTM
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from torchmetrics.classification import MulticlassAccuracy

class Flat_Tuning(pl.LightningModule):
    def __init__(self, lr, num_classes, model_path, log_loss):
        super(Flat_Tuning, self).__init__()                 
        if model_path == 'bert':
            self.model = BERT(num_classes=num_classes)

        elif model_path == 'bert-cnn':
            self.model = BERT_CNN(num_classes=num_classes) 

        elif model_path == 'bert-bilstm':
            self.model = BERT_LSTM(num_classes=num_classes, bidirectional=True)

        elif model_path == 'bert-lstm':
            self.model = BERT_LSTM(num_classes=num_classes, bidirectional=False)

        if log_loss == 'binary':
            self.criterion = nn.BCEWithLogitsLoss()
        
        elif log_loss == 'categorical':
            self.criterion == nn.CrossEntropyLoss()
        
        self.lr = lr
        self.accuracy_metric = MulticlassAccuracy(num_classes=num_classes)
        self.log_loss = log_loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

        return optimizer

    def training_step(self, train_batch, batch_idx):
        input_ids, binary_target, categorical_target = train_batch

        preds = self.model(input_ids=input_ids)
        max_pred_idx = preds.argmax(1)

        if self.log_loss == 'binary':
            loss = self.criterion(preds, binary_target.float())
            target = binary_target.argmax(1)

        elif self.log_loss == 'categorical':
            loss = self.criterion(preds, categorical_target)
            target = categorical_target

        accuracy = self.accuracy_metric(max_pred_idx, target)
        self.log_dict({'train_loss': loss, 'train_accuracy': accuracy}, prog_bar=True, on_epoch=True)

        return loss

    def validation_step(self, valid_batch, batch_idx):
        input_ids, binary_target, categorical_target = valid_batch

        preds = self.model(input_ids=input_ids)
        max_pred_idx = preds.argmax(1)

        if self.log_loss == 'binary':
            loss = self.criterion(preds, binary_target.float())
            target = binary_target.argmax(1)

        elif self.log_loss == 'categorical':
            loss = self.criterion(preds, categorical_target)
            target = categorical_target

        accuracy = self.accuracy_metric(max_pred_idx, target)
        self.log_dict({'val_loss': loss, 'val_accuracy': accuracy}, prog_bar=True, on_epoch=True)

        return loss

    def test_step(self, test_batch, batch_idx):
        input_ids, binary_target, categorical_target = test_batch

        preds = self.model(input_ids=input_ids)
        max_pred_idx = preds.argmax(1)

        if self.log_loss == 'binary':
            loss = self.criterion(preds, binary_target.float())
            target = binary_target.argmax(1)

        elif self.log_loss == 'categorical':
            loss = self.criterion(preds, categorical_target)
            target = categorical_target

        accuracy = self.accuracy_metric(max_pred_idx, target)
        self.log_dict({'test_loss': loss, 'test_accuracy': accuracy}, prog_bar=True, on_epoch=True)

        return loss

class Trainer(object):
    def __init__(self, module, model_path, method, loss):
        if method == 'flat':
            pl.seed_everything(42, workers=True)
            
            model = Flat_Tuning(lr=2e-5, num_classes=module.count_flat_classes(), model_path=model_path, log_loss=loss)
            checkpoint_callback = ModelCheckpoint(dirpath=f'./checkpoints/flat_{model_path}_{loss}_result', monitor='val_loss')
            logger = TensorBoardLogger('logs', name=f'flat_{model_path}_{loss}_result')
            early_stop_callback = EarlyStopping(monitor='val_loss', min_delta=0.00, check_on_train_epoch_end=1, patience=3)

            trainer = pl.Trainer(accelerator='gpu',
                                max_epochs=50,
                                default_root_dir=f'./checkpoints/flat_{model_path}_{loss}_result',
                                callbacks = [checkpoint_callback, early_stop_callback],
                                deterministic=True,
                                logger=logger)

            trainer.fit(model=model, datamodule=module)
            trainer.test(model=model, datamodule=module, ckpt_path='best')
        
        elif method == 'level':
            _, level_on_nodes_indexed = module.generate_hierarchy()
            level_size = len(level_on_nodes_indexed)

        elif method == 'section':
            pass
