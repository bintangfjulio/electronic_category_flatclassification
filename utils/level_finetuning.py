import os
import torch
import shutil
import torch.nn as nn
import numpy as np
import pandas as pd

from helpers.graph_helper import Graph_Helper
from statistics import mean
from tqdm import tqdm
from models.bert import BERT
from models.bert_cnn import BERT_CNN
from models.bert_lstm import BERT_LSTM
from torchmetrics.classification import MulticlassAccuracy
from torchmetrics.classification import MulticlassF1Score

class Level_FineTuning(object):
    def __init__(self, seed, tree, device, max_epochs, lr, early_stop_patience, epoch_checkpoint=None):
        super(Level_FineTuning, self).__init__() 
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
        self.epoch_checkpoint = epoch_checkpoint
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

        if self.epoch_checkpoint is not None:
            self.model.load_state_dict(torch.load(self.epoch_checkpoint))

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.scheduler = torch.optim.lr_scheduler.LinearLR(self.optimizer, start_factor=0.5, total_iters=100) 

        self.accuracy_metric = MulticlassAccuracy(num_classes=num_classes)
        self.f1_micro_metric = MulticlassF1Score(num_classes=num_classes, average='micro')
        self.f1_macro_metric = MulticlassF1Score(num_classes=num_classes, average='macro')

    def scoring_result(self, preds, target):
        accuracy = self.accuracy_metric(preds, target)
        f1_micro = self.f1_micro_metric(preds, target)
        f1_macro = self.f1_macro_metric(preds, target)

        return accuracy, f1_micro, f1_macro
                        
    def training_step(self):
        self.model.train()
        self.model.zero_grad()

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
            train_step_accuracy.append(accuracy)
            train_step_f1_micro.append(f1_micro)
            train_step_f1_macro.append(f1_macro)
            
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
                input_ids, target = valid_batch

                input_ids = input_ids.to(self.device)
                target = target.to(self.device)

                preds = self.model(input_ids=input_ids)
                loss = self.criterion(preds, target.float())

                preds = preds.argmax(1)
                target = target.argmax(1)

                accuracy, f1_micro, f1_macro = self.scoring_result(preds=preds, target=target)

                val_step_loss.append(loss.item())
                val_step_accuracy.append(accuracy)
                val_step_f1_micro.append(f1_micro)
                val_step_f1_macro.append(f1_macro)
                
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
                input_ids, target = test_batch

                input_ids = input_ids.to(self.device)
                target = target.to(self.device)

                preds = self.model(input_ids=input_ids)
                loss = self.criterion(preds, target.float())

                preds = preds.argmax(1)
                target = target.argmax(1)

                accuracy, f1_micro, f1_macro = self.scoring_result(preds=preds, target=target)

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

    def fit(self, model, datamodule):
        level_on_nodes_indexed, _, _ = self.tree.generate_hierarchy()        
        level_size = len(level_on_nodes_indexed)

        fail = 0
        minimum_loss = 1.00

        train_accuracy_graph = []
        train_loss_graph = []
        train_f1_micro_graph = []
        train_f1_macro_graph = []
        train_epoch = []
        train_level = []
        
        val_accuracy_graph = []
        val_loss_graph = []
        val_f1_micro_graph = []
        val_f1_macro_graph = []
        val_epoch = []
        val_level = []

        for epoch in range(self.max_epochs):
            for level in range(level_size):
                if epoch > 0:
                    self.epoch_checkpoint = f'checkpoints/level_{model}_result/level{str(level)}_epoch{str(epoch - 1)}_temp.pt'

                self.initialize_model(model=model, num_classes=len(level_on_nodes_indexed[level]))
                self.train_set, self.valid_set = datamodule.level_dataloader(stage='fit', level=level)

                print("Training Stage...")
                print("Epoch ", epoch)
                print("Level ", level)
                print("=" * 50)

                train_loss, train_accuracy, train_f1_micro, train_f1_macro = self.training_step()
                
                train_loss_graph.append(train_loss)
                train_accuracy_graph.append(train_accuracy)
                train_f1_micro_graph.append(train_f1_micro)
                train_f1_macro_graph.append(train_f1_macro)
                train_epoch.append(epoch)
                train_level.append(level)

                if not os.path.exists(f'checkpoints/level_{model}_result'):
                    os.makedirs(f'checkpoints/level_{model}_result')

                torch.save(self.model.state_dict(), f'checkpoints/level_{model}_result/level{str(level)}_epoch{str(epoch)}_temp.pt')

                if epoch > 0:
                    os.remove(f'checkpoints/level_{model}_result/level{str(level)}_epoch{str(epoch - 1)}_temp.pt')

                print("Validation Stage...")
                print("=" * 50)

                val_loss, val_accuracy, val_f1_micro, val_f1_macro = self.validation_step()
                
                val_loss_graph.append(val_loss)
                val_accuracy_graph.append(val_accuracy)
                val_f1_micro_graph.append(val_f1_micro)
                val_f1_macro_graph.append(val_f1_macro)
                val_epoch.append(epoch)
                val_level.append(level)

                if(level == level_size - 1):
                    if round(val_loss, 2) < round(minimum_loss, 2):
                        fail = 0
                        minimum_loss = val_loss                    
                    else:
                        fail += 1

                    for level in range(level_size):
                        if not os.path.exists(f'checkpoints/level_{model}_result/best_model'):
                            os.makedirs(f'checkpoints/level_{model}_result/best_model')

                        if os.path.exists(f'checkpoints/level_{model}_result/best_model/level{str(level)}_model.pt'):
                            os.remove(f'checkpoints/level_{model}_result/best_model/level{str(level)}_model.pt')

                        shutil.copy(f'checkpoints/level_{model}_result/level{str(level)}_epoch{str(epoch)}_temp.pt', f'checkpoints/level_{model}_result/best_model/level{str(level)}_model.pt')

            if fail == self.early_stop_patience:
                break
        
        if not os.path.exists(f'logs/level_{model}_result'):
            os.makedirs(f'logs/level_{model}_result')
        
        train_graph = pd.DataFrame({'epoch': train_epoch, 'level': train_level, 'accuracy': train_accuracy_graph, 'loss': train_loss_graph, 'f1_micro': train_f1_micro_graph, 'f1_macro': train_f1_macro_graph})
        valid_graph = pd.DataFrame({'epoch': val_epoch, 'level': val_level, 'accuracy': val_accuracy_graph, 'loss': val_loss_graph, 'f1_micro': val_f1_micro_graph, 'f1_macro': val_f1_macro_graph})
        
        train_graph.to_csv(f'logs/level_{model}_result/train_graph.csv', index=False, encoding='utf-8')
        valid_graph.to_csv(f'logs/level_{model}_result/valid_graph.csv', index=False, encoding='utf-8')

        # for data in [train_graph, valid_graph]:
        #     graph_generator = Graph_Helper(data, stage='fit')
        #     graph_generator.save_graph()

    def test(self, model, datamodule):
        level_on_nodes_indexed, _, _ = self.tree.generate_hierarchy()        
        level_size = len(level_on_nodes_indexed)

        test_accuracy_graph = []
        test_loss_graph = []
        test_f1_micro_graph = []
        test_f1_macro_graph = []
        test_level = []

        for level in range(level_size):
            self.epoch_checkpoint = f'checkpoints/level_{model}_result/best_model/level{str(level)}_model.pt'
            self.initialize_model(model=model, num_classes=len(level_on_nodes_indexed[level]))
            self.test_set = datamodule.level_dataloader(stage='test', level=level)

            print("Test Stage...")
            print("Level ", level)
            print("=" * 50)
            
            test_loss, test_accuracy, test_f1_micro, test_f1_macro = self.test_step()
            
            test_loss_graph.append(test_loss)
            test_accuracy_graph.append(test_accuracy)
            test_f1_micro_graph.append(test_f1_micro)
            test_f1_macro_graph.append(test_f1_macro)
            test_level.append(level)
                        
        if not os.path.exists(f'logs/level_{model}_result'):
            os.makedirs(f'logs/level_{model}_result')
                        
        test_graph = pd.DataFrame({'level': test_level, 'accuracy': test_accuracy_graph, 'loss': test_loss_graph, 'f1_micro': test_f1_micro_graph, 'f1_macro': test_f1_macro_graph})
        test_graph.to_csv(f'logs/level_{model}_result/test_graph.csv', index=False, encoding='utf-8')
        
        # graph_generator = Graph_Helper(test_graph, stage='test')
        # graph_generator.save_graph()
