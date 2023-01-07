import pytorch_lightning as pl
import torch
import torch.nn as nn

from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import accuracy_score
from sklearn.metrics import matthews_corrcoef
from models.bert import BERT
from models.bert_cnn import BERT_CNN
from models.bert_lstm import BERT_LSTM

class Flat_FineTuning(pl.LightningModule):
    def __init__(self, lr, model_path, num_classes):
        super(Flat_FineTuning, self).__init__()                 
        if model_path == 'bert':
            self.model = BERT(num_classes=num_classes)
        elif model_path == 'bert-cnn':
            self.model = BERT_CNN(num_classes=num_classes) 
        elif model_path == 'bert-bilstm':
            self.model = BERT_LSTM(num_classes=num_classes, bidirectional=True)
        elif model_path == 'bert-lstm':
            self.model = BERT_LSTM(num_classes=num_classes, bidirectional=False)
        
        self.lr = lr
        self.criterion = nn.BCELoss()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

        return optimizer

    def training_step(self, train_batch, batch_idx):
        input_ids, target = train_batch

        probabilities = self.model(input_ids=input_ids)
        loss = self.criterion(probabilities.cpu(), target=target.float().cpu())

        max_probabilities_idx = probabilities.argmax(1).cpu()
        max_target_idx = target.argmax(1).cpu()
        accuracy = accuracy_score(max_target_idx, max_probabilities_idx)
        mcc = matthews_corrcoef(max_target_idx, max_probabilities_idx)

        self.log_dict({'train_loss': loss, 'train_accuracy': accuracy, 'train_mcc': mcc}, prog_bar=True, on_epoch=True)

        return loss

    def validation_step(self, valid_batch, batch_idx):
        input_ids, target = valid_batch

        probabilities = self.model(input_ids=input_ids)
        loss = self.criterion(probabilities.cpu(), target=target.float().cpu())

        max_probabilities_idx = probabilities.argmax(1).cpu()
        max_target_idx = target.argmax(1).cpu()
        accuracy = accuracy_score(max_target_idx, max_probabilities_idx)
        mcc = matthews_corrcoef(max_target_idx, max_probabilities_idx)

        self.log_dict({'val_loss': loss, 'val_accuracy': accuracy, 'val_mcc': mcc}, prog_bar=True, on_epoch=True)

        return loss

    def test_step(self, test_batch, batch_idx):
        input_ids, target = test_batch

        probabilities = self.model(input_ids=input_ids)
        loss = self.criterion(probabilities.cpu(), target=target.float().cpu())

        max_probabilities_idx = probabilities.argmax(1).cpu()
        max_target_idx = target.argmax(1).cpu()
        accuracy = accuracy_score(max_target_idx, max_probabilities_idx)
        mcc = matthews_corrcoef(max_target_idx, max_probabilities_idx)

        self.log_dict({'test_loss': loss, 'test_accuracy': accuracy, 'test_mcc': mcc}, prog_bar=True, on_epoch=True)

        return loss

class Hierarchical_FineTuning:
    pass

class Trainer:
    def __init__(self, model_path, module, num_classes, method):
        if method == 'flat':
            pl.seed_everything(42, workers=True)
            
            model = Flat_FineTuning(lr=2e-5, num_classes=num_classes, model_path=model_path)
            checkpoint_callback = ModelCheckpoint(dirpath=f'./checkpoints/flat_{model_path}_result', monitor='val_loss')
            logger = TensorBoardLogger('logs', name=f'flat_{model_path}_result')
            early_stop_callback = EarlyStopping(monitor='val_loss', min_delta=0.00, check_on_train_epoch_end=1, patience=3)

            trainer = pl.Trainer(accelerator='gpu',
                                max_epochs=30,
                                default_root_dir=f'./checkpoints/flat_{model_path}_result',
                                callbacks = [checkpoint_callback, early_stop_callback],
                                deterministic=True,
                                logger=logger)

            trainer.fit(model=model, datamodule=module)
            trainer.test(model=model, datamodule=module, ckpt_path='best')
         
        elif method == 'hierarchy':
            pass
