import pytorch_lightning as pl
import torch
import torch.nn as nn

from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import classification_report
from models.bert import BERT
from models.bert_cnn import BERT_CNN
from models.bert_lstm import BERT_LSTM

class Flat_Trainer(pl.LightningModule):
    def __init__(self, lr, model_path, num_classes):
        super(Flat_Trainer, self).__init__()     
        if model_path == 'bert-cnn':
            self.criterion = nn.BCELoss()
        else:
            self.criterion = nn.BCEWithLogitsLoss()
        
        self.lr = lr
        
        if model_path == 'bert':
            self.model = BERT(num_classes=num_classes)
        elif model_path == 'bert-cnn':
            self.model = BERT_CNN(num_classes=num_classes) 
        elif model_path == 'bert-bilstm':
            self.model = BERT_LSTM(num_classes=num_classes, bidirectional=True)
        elif model_path == 'bert-lstm':
            self.model = BERT_LSTM(num_classes=num_classes, bidirectional=False)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

        return optimizer

    def training_step(self, train_batch, batch_idx):
        flat_input_ids, flat_target = train_batch

        output = self.model(input_ids=flat_input_ids)
        loss = self.criterion(output.cpu(), target=flat_target.float().cpu())

        preds = output.argmax(1).cpu()
        flat_target = flat_target.argmax(1).cpu()
        report = classification_report(flat_target, preds, output_dict=True, zero_division=0)

        self.log_dict({'train_loss': loss, 'train_accuracy': report["accuracy"]}, prog_bar=True, on_epoch=True)

        return loss

    def validation_step(self, valid_batch, batch_idx):
        flat_input_ids, flat_target = valid_batch

        output = self.model(input_ids=flat_input_ids)
        loss = self.criterion(output.cpu(), target=flat_target.float().cpu())

        preds = output.argmax(1).cpu()
        flat_target = flat_target.argmax(1).cpu()
        report = classification_report(flat_target, preds, output_dict=True, zero_division=0)

        self.log_dict({'val_loss': loss, 'val_accuracy': report["accuracy"]}, prog_bar=True, on_epoch=True)

        return loss

    def test_step(self, test_batch, batch_idx):
        flat_input_ids, flat_target = test_batch

        output = self.model(input_ids=flat_input_ids)
        loss = self.criterion(output.cpu(), target=flat_target.float().cpu())

        preds = output.argmax(1).cpu()
        flat_target = flat_target.argmax(1).cpu()
        report = classification_report(flat_target, preds, output_dict=True, zero_division=0)

        self.log_dict({'test_loss': loss, 'test_accuracy': report["accuracy"]}, prog_bar=True, on_epoch=True)

        return loss

class Hierarchical_Trainer:
    pass

class Trainer:
    def __init__(self, model_path, module, num_classes, method):
        if method == 'flat':
            model = Flat_Trainer(lr=2e-5, num_classes=num_classes, model_path=model_path)
            self.flat_fine_tuning(model=model, module=module, model_path=model_path)
        elif method == 'hierarchy':
            self.hierarchical_fine_tuning()

    def flat_fine_tuning(self, model, module, model_path):
        pl.seed_everything(42, workers=True)
        
        checkpoint_callback = ModelCheckpoint(dirpath=f'./checkpoints/flat_{model_path}_result', monitor='val_loss')
        logger = TensorBoardLogger('logs', name=f'flat_{model_path}_result')
        early_stop_callback = EarlyStopping(monitor='val_loss', min_delta=0.00, check_on_train_epoch_end=1, patience=3)

        trainer = pl.Trainer(
            accelerator='gpu',
            max_epochs=30,
            default_root_dir=f'./checkpoints/flat_{model_path}_result',
            callbacks = [checkpoint_callback, early_stop_callback],
            deterministic=True,
            logger=logger)

        trainer.fit(model=model, datamodule=module)
        trainer.test(model=model, datamodule=module, ckpt_path='best')

    def hierarchical_fine_tuning(self):
        pass
