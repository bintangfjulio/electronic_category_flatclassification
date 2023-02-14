import torch
import pytorch_lightning as pl
import torch.nn as nn
import numpy as np

from statistics import mean
from tqdm import tqdm
from models.bert import BERT
from models.bert_cnn import BERT_CNN
from models.bert_lstm import BERT_LSTM
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from torchmetrics.classification import MulticlassAccuracy
from torchmetrics.classification import MulticlassF1Score

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
        self.f1_micro_metric = MulticlassF1Score(num_classes=num_classes, average='micro')
        self.f1_macro_metric = MulticlassF1Score(num_classes=num_classes, average='macro')
        self.log_loss = log_loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

        return optimizer

    def training_step(self, train_batch, batch_idx):
        input_ids, binary_target, categorical_target = train_batch

        preds = self.model(input_ids=input_ids)
        max_preds_idx = preds.argmax(1)

        if self.log_loss == 'binary':
            loss = self.criterion(preds, binary_target.float())
            target = binary_target.argmax(1)

        elif self.log_loss == 'categorical':
            loss = self.criterion(preds, categorical_target)
            target = categorical_target

        accuracy = self.accuracy_metric(max_preds_idx, target)
        f1_micro = self.f1_micro_metric(max_preds_idx, target)
        f1_macro = self.f1_macro_metric(max_preds_idx, target)
        self.log_dict({'train_loss': loss, 'train_accuracy': accuracy, 'train_f1_micro': f1_micro, 'train_f1_macro': f1_macro}, prog_bar=True, on_epoch=True)

        return loss

    def validation_step(self, valid_batch, batch_idx):
        input_ids, binary_target, categorical_target = valid_batch

        preds = self.model(input_ids=input_ids)
        max_preds_idx = preds.argmax(1)

        if self.log_loss == 'binary':
            loss = self.criterion(preds, binary_target.float())
            target = binary_target.argmax(1)

        elif self.log_loss == 'categorical':
            loss = self.criterion(preds, categorical_target)
            target = categorical_target

        accuracy = self.accuracy_metric(max_preds_idx, target)
        f1_micro = self.f1_micro_metric(max_preds_idx, target)
        f1_macro = self.f1_macro_metric(max_preds_idx, target)
        self.log_dict({'val_loss': loss, 'val_accuracy': accuracy, 'val_f1_micro': f1_micro, 'val_f1_macro': f1_macro}, prog_bar=True, on_epoch=True)

        return loss

    def test_step(self, test_batch, batch_idx):
        input_ids, binary_target, categorical_target = test_batch

        preds = self.model(input_ids=input_ids)
        max_preds_idx = preds.argmax(1)

        if self.log_loss == 'binary':
            loss = self.criterion(preds, binary_target.float())
            target = binary_target.argmax(1)

        elif self.log_loss == 'categorical':
            loss = self.criterion(preds, categorical_target)
            target = categorical_target

        accuracy = self.accuracy_metric(max_preds_idx, target)
        f1_micro = self.f1_micro_metric(max_preds_idx, target)
        f1_macro = self.f1_macro_metric(max_preds_idx, target)
        self.log_dict({'test_loss': loss, 'test_accuracy': accuracy, 'test_f1_micro': f1_micro, 'test_f1_macro': f1_macro}, prog_bar=True, on_epoch=True)

        return loss

class Level_Tuning(object):
    def __init__(self, module, seed, device, max_epochs, lr, model_path, log_loss, early_stop_patience, last_level_checkpoint=None):
        super(Level_Tuning, self).__init__() 
        np.random.seed(seed) 
        torch.manual_seed(seed)

        if device == 'cuda':
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True

        self.device = device
        self.module = module
        self.max_epochs = max_epochs
        self.lr = lr
        self.model_path = model_path
        self.early_stop_patience = early_stop_patience

        if log_loss == 'binary':
            self.criterion = nn.BCEWithLogitsLoss()
        
        elif log_loss == 'categorical':
            self.criterion == nn.CrossEntropyLoss()

        # self.last_level_checkpoint = last_level_checkpoint

    def initialize_model(self, num_classes):
        if self.model_path == 'bert':
            self.model = BERT(num_classes=num_classes)

        elif self.model_path == 'bert-cnn':
            self.model = BERT_CNN(num_classes=num_classes) 

        elif self.model_path == 'bert-bilstm':
            self.model = BERT_LSTM(num_classes=num_classes, bidirectional=True)

        elif self.model_path == 'bert-lstm':
            self.model = BERT_LSTM(num_classes=num_classes, bidirectional=False)

        self.model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.scheduler = torch.optim.lr_scheduler.LinearLR(self.optimizer, 
                                                        start_factor=0.5, 
                                                        total_iters=100)

        # if self.last_level_checkpoint:
        #     self.model.load_state_dict(torch.load(self.last_level_checkpoint))
        
    def fit(self):
        _, level_on_nodes_indexed = self.module.generate_hierarchy()
        level_size = len(level_on_nodes_indexed)

        patience = self.early_stop_patience
        minimum_loss = 0
        fail = 0

        train_graph = []
        val_graph = []

        for epoch in range(self.max_epochs):
            print("Epoch = ", (epoch + 1))

            for level in range(level_size):
                self.num_classes = len(level_on_nodes_indexed[level])
                self.initialize_model(num_classes=self.num_classes)

                self.train_set, self.valid_set = self.module.level_dataloader(stage='fit', level=level)

                print("Level ", level)
                print("=" * 50)

                print("Training Stage")
                train_loss, train_accuracy, train_f1_micro, train_f1_macro = self.training_step()

                print("Validation Stage")
                val_loss, val_accuracy, val_f1_micro, val_f1_macro = self.validation_step()
                print("=" * 50)

                if round(val_loss, 2) < round(minimum_loss, 2):
                    fail = 0
                    minimum_loss = val_loss
            
                else:
                    fail += 1

            if fail == patience:
                break

    def test(self):
        _, level_on_nodes_indexed = self.module.generate_hierarchy()
        level_size = len(level_on_nodes_indexed)

        test_graph = []

        for level in range(level_size):
            self.test_set = self.module.level_dataloader(stage='test', level=level)
            print("Level ", level)
            print("=" * 50)

            print("Test Stage")
            test_loss, test_accuracy, test_f1_micro, test_f1_macro = self.test_step()
            print("=" * 50)

    def training_step(self):
        self.model.train()
        self.model.zero_grad()

        train_step_loss = []
        train_step_accuracy = []
        train_step_f1_micro = []
        train_step_f1_macro = []

        training_progress = tqdm(self.train_set)

        for train_batch in training_progress:
            input_ids, binary_target, categorical_target = train_batch

            input_ids = input_ids.to(self.device)
            binary_target = binary_target.to(self.device)
            categorical_target = categorical_target.to(self.device)

            preds = self.model(input_ids=input_ids)
            max_preds_idx = preds.argmax(1)

            if self.log_loss == 'binary':
                loss = self.criterion(preds, binary_target.float())
                target = binary_target.argmax(1)

            elif self.log_loss == 'categorical':
                loss = self.criterion(preds, categorical_target)
                target = categorical_target

            accuracy, f1_micro, f1_macro = self.scoring_result(max_preds_idx=max_preds_idx, target=target)
            train_step_accuracy.append(accuracy)
            train_step_f1_micro.append(f1_micro)
            train_step_f1_macro.append(f1_macro)
            train_step_loss.append(loss.item())
            
            training_progress.set_description("Train Step Loss : " + str(round(loss.item(), 2)) + 
                                        " | Train Step Accuracy : " + str(round(accuracy, 2)) + 
                                        " | Train Step F1 Micro : " + str(round(f1_micro, 2)) +
                                        " | Train Step F1 Macro : " + str(round(f1_macro, 2)))

            loss.backward()
            self.optimizer.step()

        print("On Epoch Train Loss: ", round(mean(train_step_loss), 2))
        print("On Epoch Train Accuracy: ", round(mean(train_step_accuracy), 2))
        print("On Epoch Train F1 Micro: ", round(mean(train_step_f1_micro), 2))
        print("On Epoch Train F1 Macro: ", round(mean(train_step_f1_macro), 2))

        self.scheduler.step()

        return mean(train_step_loss), mean(train_step_accuracy), mean(train_step_f1_micro), mean(train_step_f1_macro)

    def validation_step(self):
        with torch.no_grad():
            val_step_loss = []
            val_step_accuracy = []
            val_step_f1_micro = []
            val_step_f1_macro = []

            self.model.eval()
            validation_progress = tqdm(self.valid_set)

            for valid_batch in validation_progress:
                input_ids, binary_target, categorical_target = valid_batch

                input_ids = input_ids.to(self.device)
                binary_target = binary_target.to(self.device)
                categorical_target = categorical_target.to(self.device)

                preds = self.model(input_ids=input_ids)
                max_preds_idx = preds.argmax(1)

                if self.log_loss == 'binary':
                    loss = self.criterion(preds, binary_target.float())
                    target = binary_target.argmax(1)

                elif self.log_loss == 'categorical':
                    loss = self.criterion(preds, categorical_target)
                    target = categorical_target

                accuracy, f1_micro, f1_macro = self.scoring_result(max_preds_idx=max_preds_idx, target=target)
                val_step_accuracy.append(accuracy)
                val_step_f1_micro.append(f1_micro)
                val_step_f1_macro.append(f1_macro)
                val_step_loss.append(loss.item())
                
                validation_progress.set_description("Validation Step Loss : " + str(round(loss.item(), 2)) + 
                                            " | Validation Step Accuracy : " + str(round(accuracy, 2)) + 
                                            " | Validation Step F1 Micro : " + str(round(f1_micro, 2)) +
                                            " | Validation Step F1 Macro : " + str(round(f1_macro, 2)))

        print("On Epoch Validation Loss: ", round(mean(val_step_loss), 2))
        print("On Epoch Validation Accuracy: ", round(mean(val_step_accuracy), 2))
        print("On Epoch Validation F1 Micro: ", round(mean(val_step_f1_micro), 2))
        print("On Epoch Validation F1 Macro: ", round(mean(val_step_f1_macro), 2))

        self.scheduler.step()

        return mean(val_step_loss), mean(val_step_accuracy), mean(val_step_f1_micro), mean(val_step_f1_macro)

    def test_step(self):
        with torch.no_grad():
            test_step_loss = []
            test_step_accuracy = []
            test_step_f1_micro = []
            test_step_f1_macro = []

            self.model.eval()
            test_progress = tqdm(self.test_set)

            for test_batch in test_progress:
                input_ids, binary_target, categorical_target = test_batch

                input_ids = input_ids.to(self.device)
                binary_target = binary_target.to(self.device)
                categorical_target = categorical_target.to(self.device)

                preds = self.model(input_ids=input_ids)
                max_preds_idx = preds.argmax(1)

                if self.log_loss == 'binary':
                    loss = self.criterion(preds, binary_target.float())
                    target = binary_target.argmax(1)

                elif self.log_loss == 'categorical':
                    loss = self.criterion(preds, categorical_target)
                    target = categorical_target

                accuracy, f1_micro, f1_macro = self.scoring_result(max_preds_idx=max_preds_idx, target=target)
                test_step_accuracy.append(accuracy)
                test_step_f1_micro.append(f1_micro)
                test_step_f1_macro.append(f1_macro)
                test_step_loss.append(loss.item())
                
                test_progress.set_description("Test Step Loss : " + str(round(loss.item(), 2)) + 
                                            " | Test Step Accuracy : " + str(round(accuracy, 2)) + 
                                            " | Test Step F1 Micro : " + str(round(f1_micro, 2)) +
                                            " | Test Step F1 Macro : " + str(round(f1_macro, 2)))

        print("On Epoch Test Loss: ", round(mean(test_step_loss), 2))
        print("On Epoch Test Accuracy: ", round(mean(test_step_accuracy), 2))
        print("On Epoch Test F1 Micro: ", round(mean(test_step_f1_micro), 2))
        print("On Epoch Test F1 Macro: ", round(mean(test_step_f1_macro), 2))

        self.scheduler.step()

        return mean(test_step_loss), mean(test_step_accuracy), mean(test_step_f1_micro), mean(test_step_f1_macro)

    def scoring_result(self, max_preds_idx, target):
        accuracy_metric = MulticlassAccuracy(num_classes=self.num_classes)
        f1_micro_metric = MulticlassF1Score(num_classes=self.num_classes, average='micro')
        f1_macro_metric = MulticlassF1Score(num_classes=self.num_classes, average='macro')

        accuracy = accuracy_metric(max_preds_idx, target)
        f1_micro = f1_micro_metric(max_preds_idx, target)
        f1_macro = f1_macro_metric(max_preds_idx, target)

        return accuracy, f1_micro, f1_macro

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
                                logger=logger,
                                deterministic=True)

            trainer.fit(model=model, datamodule=module)
            trainer.test(model=model, datamodule=module, ckpt_path='best')
        
        elif method == 'level':
            trainer = Level_Tuning(module=module,
                                seed=42, 
                                device='cuda', 
                                max_epochs=50,
                                lr=2e-5, 
                                model_path=model_path, 
                                log_loss=loss,
                                early_stop_patience=3)

            trainer.fit()
            trainer.test()

        elif method == 'section':
            pass
