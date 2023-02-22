import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd

from statistics import mean
from tqdm import tqdm
from models.bert import BERT
from models.bert_cnn import BERT_CNN
from models.bert_lstm import BERT_LSTM
from torchmetrics.classification import MulticlassAccuracy
from torchmetrics.classification import MulticlassF1Score

class Flat_FineTuning(object):
    def __init__(self, seed, tree, device, max_epochs, lr, early_stop_patience):
        super(Flat_FineTuning, self).__init__() 
        np.random.seed(seed) 
        torch.manual_seed(seed)

        if device == 'cuda':
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True

        self.tree = tree
        self.device = device
        self.max_epochs = max_epochs
        self.lr = lr
        self.early_stop_patience = early_stop_patience
        self.criterion = nn.BCEWithLogitsLoss()

    def initialize_model(self, model, num_classes):
        if model == 'bert':
            self.model = BERT(num_classes=num_classes)

        elif model == 'bert-cnn':
            self.model = BERT_CNN(num_classes=num_classes) 

        elif model == 'bert-bilstm':
            self.model = BERT_LSTM(num_classes=num_classes, bidirectional=True)

        elif model == 'bert-lstm':
            self.model = BERT_LSTM(num_classes=num_classes, bidirectional=False)

        self.model.to(self.device)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.scheduler = torch.optim.lr_scheduler.LinearLR(self.optimizer, start_factor=0.5, total_iters=100) 

        self.accuracy_metric = MulticlassAccuracy(num_classes=num_classes).to(self.device)
        self.f1_micro_metric = MulticlassF1Score(num_classes=num_classes, average='micro').to(self.device)
        self.f1_macro_metric = MulticlassF1Score(num_classes=num_classes, average='macro').to(self.device)

    def scoring_result(self, preds, target):
        accuracy = self.accuracy_metric(preds, target)
        f1_micro = self.f1_micro_metric(preds, target)
        f1_macro = self.f1_macro_metric(preds, target)

        return accuracy, f1_micro, f1_macro

    def training_step(self):
        self.model.train(True)

        train_step_loss = []
        train_step_accuracy = []
        train_step_f1_micro = []
        train_step_f1_macro = []

        training_progress = tqdm(self.train_set)

        for train_batch in training_progress:
            input_ids, target = train_batch

            input_ids = input_ids.to(self.device)
            target = target.to(self.device)

            preds = self.model(input_ids=input_ids)
            loss = self.criterion(preds, target.float())

            preds = preds.argmax(1)
            target = target.argmax(1)

            accuracy, f1_micro, f1_macro = self.scoring_result(preds=preds, target=target)

            train_step_loss.append(loss.item())
            train_step_accuracy.append(accuracy.item())
            train_step_f1_micro.append(f1_micro.item())
            train_step_f1_macro.append(f1_macro.item())
            
            training_progress.set_description("Train Step Loss : " + str(round(loss.item(), 2)) + 
                                        " | Train Step Accuracy : " + str(round(accuracy.item(), 2)) + 
                                        " | Train Step F1 Micro : " + str(round(f1_micro.item(), 2)) +
                                        " | Train Step F1 Macro : " + str(round(f1_macro.item(), 2)))

            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            self.model.zero_grad()

        print("On Epoch Train Loss: ", round(mean(train_step_loss), 2))
        print("On Epoch Train Accuracy: ", round(mean(train_step_accuracy), 2))
        print("On Epoch Train F1 Micro: ", round(mean(train_step_f1_micro), 2))
        print("On Epoch Train F1 Macro: ", round(mean(train_step_f1_macro), 2))

        return mean(train_step_loss), mean(train_step_accuracy), mean(train_step_f1_micro), mean(train_step_f1_macro)

    def validation_step(self):
        self.model.eval()

        val_step_loss = []
        val_step_accuracy = []
        val_step_f1_micro = []
        val_step_f1_macro = []

        with torch.no_grad():
            validation_progress = tqdm(self.valid_set)

            for valid_batch in validation_progress:
                input_ids, target = valid_batch

                input_ids = input_ids.to(self.device)
                target = target.to(self.device)

                preds = self.model(input_ids=input_ids)
                loss = self.criterion(preds, target.float())

                preds = preds.argmax(1)
                target = target.argmax(1)

                accuracy, f1_micro, f1_macro = self.scoring_result(preds=preds, target=target)

                val_step_loss.append(loss.item())
                val_step_accuracy.append(accuracy.item())
                val_step_f1_micro.append(f1_micro.item())
                val_step_f1_macro.append(f1_macro.item())
                
                validation_progress.set_description("Validation Step Loss : " + str(round(loss.item(), 2)) + 
                                            " | Validation Step Accuracy : " + str(round(accuracy.item(), 2)) + 
                                            " | Validation Step F1 Micro : " + str(round(f1_micro.item(), 2)) +
                                            " | Validation Step F1 Macro : " + str(round(f1_macro.item(), 2)))

                self.model.zero_grad()

        print("On Epoch Validation Loss: ", round(mean(val_step_loss), 2))
        print("On Epoch Validation Accuracy: ", round(mean(val_step_accuracy), 2))
        print("On Epoch Validation F1 Micro: ", round(mean(val_step_f1_micro), 2))
        print("On Epoch Validation F1 Macro: ", round(mean(val_step_f1_macro), 2))

        return mean(val_step_loss), mean(val_step_accuracy), mean(val_step_f1_micro), mean(val_step_f1_macro)

    def test_step(self):
        self.model.eval()

        test_step_loss = []
        test_step_accuracy = []
        test_step_f1_micro = []
        test_step_f1_macro = []

        with torch.no_grad():
            test_progress = tqdm(self.test_set)

            for test_batch in test_progress:
                input_ids, target = test_batch

                input_ids = input_ids.to(self.device)
                target = target.to(self.device)

                preds = self.model(input_ids=input_ids)
                loss = self.criterion(preds, target.float())

                preds = preds.argmax(1)
                target = target.argmax(1)

                accuracy, f1_micro, f1_macro = self.scoring_result(preds=preds, target=target)

                test_step_loss.append(loss.item())
                test_step_accuracy.append(accuracy.item())
                test_step_f1_micro.append(f1_micro.item())
                test_step_f1_macro.append(f1_macro.item())
                
                test_progress.set_description("Test Step Loss : " + str(round(loss.item(), 2)) + 
                                            " | Test Step Accuracy : " + str(round(accuracy.item(), 2)) + 
                                            " | Test Step F1 Micro : " + str(round(f1_micro.item(), 2)) +
                                            " | Test Step F1 Macro : " + str(round(f1_macro.item(), 2)))

        print("On Epoch Test Loss: ", round(mean(test_step_loss), 2))
        print("On Epoch Test Accuracy: ", round(mean(test_step_accuracy), 2))
        print("On Epoch Test F1 Micro: ", round(mean(test_step_f1_micro), 2))
        print("On Epoch Test F1 Macro: ", round(mean(test_step_f1_macro), 2))

        return mean(test_step_loss), mean(test_step_accuracy), mean(test_step_f1_micro), mean(test_step_f1_macro)
        
    def fit(self, model, datamodule):
        level_on_nodes_indexed, _, _ = self.tree.generate_hierarchy()        

        fail = 0
        minimum_loss = 1.00

        train_accuracy_epoch = []
        train_loss_epoch = []
        train_f1_micro_epoch = []
        train_f1_macro_epoch = []
        train_epoch = []

        val_accuracy_epoch = []
        val_loss_epoch = []
        val_f1_micro_epoch = []
        val_f1_macro_epoch = []
        val_epoch = []

        self.initialize_model(model=model, num_classes=len(level_on_nodes_indexed[-1]))
        self.model.zero_grad()

        self.train_set, self.valid_set = datamodule.flat_dataloader(stage='fit')

        for epoch in range(self.max_epochs):
            print("Training Stage...")
            print("Epoch ", epoch)
            print("=" * 50)

            train_loss, train_accuracy, train_f1_micro, train_f1_macro = self.training_step()
            
            train_loss_epoch.append(train_loss)
            train_accuracy_epoch.append(train_accuracy)
            train_f1_micro_epoch.append(train_f1_micro)
            train_f1_macro_epoch.append(train_f1_macro)
            train_epoch.append(epoch)
            
            print("Validation Stage...")
            print("=" * 50)

            val_loss, val_accuracy, val_f1_micro, val_f1_macro = self.validation_step()

            val_loss_epoch.append(val_loss)
            val_accuracy_epoch.append(val_accuracy)
            val_f1_micro_epoch.append(val_f1_micro)
            val_f1_macro_epoch.append(val_f1_macro)
            val_epoch.append(epoch)

            if round(val_loss, 2) < round(minimum_loss, 2):
                fail = 0
                minimum_loss = val_loss

                if not os.path.exists(f'checkpoints/flat_{model}_result'):
                    os.makedirs(f'checkpoints/flat_{model}_result')

                if os.path.exists(f'checkpoints/flat_{model}_result/flat_model.pt'):
                    os.remove(f'checkpoints/flat_{model}_result/flat_model.pt')

                torch.save(self.model.state_dict(), f'checkpoints/flat_{model}_result/flat_model.pt')

            else:
                fail += 1

            if fail == self.early_stop_patience:
                break

        if not os.path.exists(f'logs/flat_{model}_result'):
            os.makedirs(f'logs/flat_{model}_result')
            
        train_result = pd.DataFrame({'epoch': train_epoch, 'accuracy': train_accuracy_epoch, 'loss': train_loss_epoch, 'f1_micro': train_f1_micro_epoch, 'f1_macro': train_f1_macro_epoch})
        valid_result = pd.DataFrame({'epoch': val_epoch, 'accuracy': val_accuracy_epoch, 'loss': val_loss_epoch, 'f1_micro': val_f1_micro_epoch, 'f1_macro': val_f1_macro_epoch})
        
        train_result.to_csv(f'logs/flat_{model}_result/train_epoch.csv', index=False, encoding='utf-8')
        valid_result.to_csv(f'logs/flat_{model}_result/valid_epoch.csv', index=False, encoding='utf-8')

    def test(self, model, datamodule):
        test_accuracy_epoch = []
        test_loss_epoch = []
        test_f1_micro_epoch = []
        test_f1_macro_epoch = []

        self.model.load_state_dict(torch.load(f'checkpoints/flat_{model}_result/flat_model.pt'))
        self.model.to(self.device)

        self.test_set = datamodule.flat_dataloader(stage='test')

        print("Test Stage...")
        print("=" * 50)
        
        test_loss, test_accuracy, test_f1_micro, test_f1_macro = self.test_step()
        
        test_loss_epoch.append(test_loss)
        test_accuracy_epoch.append(test_accuracy)
        test_f1_micro_epoch.append(test_f1_micro)
        test_f1_macro_epoch.append(test_f1_macro)
                        
        if not os.path.exists(f'logs/flat_{model}_result'):
            os.makedirs(f'logs/flat_{model}_result')
                        
        test_result = pd.DataFrame({'accuracy': test_accuracy_epoch, 'loss': test_loss_epoch, 'f1_micro': test_f1_micro_epoch, 'f1_macro': test_f1_macro_epoch})
        test_result.to_csv(f'logs/flat_{model}_result/test_epoch.csv', index=False, encoding='utf-8')