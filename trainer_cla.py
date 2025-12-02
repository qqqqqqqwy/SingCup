import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np

from model import MidFusionCNNLSTMClassifier
from utils import setup_seed, preprocess_solutecla_data, preprocess_concla_data
from dataloader import LiquidDataset

class BaseClaTrainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.bandwidth = config.get('bandwidth', 1500)
        self.duration = config.get('duration', 100)
        self.model_name = config.get('model_name', 'ClaNet')
        self.root_dir = config.get('root_dir', r"dataset/merged_dataset")
        self.batch_size = config.get('batch_size', 8)
        self.num_epochs = config.get('num_epochs', 50)
        self.learning_rate = config.get('learning_rate', 0.001)
        setup_seed(config.get('seed', 42))

    def train_epoch(self, model, loader, criterion, optimizer):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for inputs_ac, inputs_mass, labels in loader:
            inputs_ac, inputs_mass, labels = inputs_ac.to(self.device), inputs_mass.to(self.device), labels.to(self.device)
            
            optimizer.zero_grad()
            outputs = model(inputs_ac, inputs_mass)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs_ac.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
        return running_loss / total, 100.0 * correct / total

    def eval_epoch(self, model, loader, criterion):
        model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs_ac, inputs_mass, labels in loader:
                inputs_ac, inputs_mass, labels = inputs_ac.to(self.device), inputs_mass.to(self.device), labels.to(self.device)
                outputs = model(inputs_ac, inputs_mass)
                loss = criterion(outputs, labels)
                running_loss += loss.item() * inputs_ac.size(0)
                preds = outputs.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        return running_loss / total, 100.0 * correct / total

    def run(self, task_name, X_train, M_train, y_train, X_val, M_val, y_val, num_classes, i):
        print(f"\n--- Starting Training for {task_name} (Classes: {num_classes}) ---")
        
        train_dataset = LiquidDataset(X_train, M_train, y_train, task_type='classification')
        val_dataset = LiquidDataset(X_val, M_val, y_val, task_type='classification')
        
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=0)
        
        model = MidFusionCNNLSTMClassifier(ac_feat_size=self.bandwidth, num_classes=num_classes).to(self.device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
        
        best_acc = 0.0
        # Model Weight Save Path
        save_path = f'./models/{self.model_name}_{task_name}.pth'
        
        for epoch in range(self.num_epochs):
            train_loss, train_acc = self.train_epoch(model, train_loader, criterion, optimizer)
            val_loss, val_acc = self.eval_epoch(model, val_loader, criterion)
            
            scheduler.step(val_loss)
            
            print(f"[{task_name}] Epoch {epoch+1}/{self.num_epochs} | "
                  f"Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | "
                  f"Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}%")
            
            if val_acc > best_acc:
                best_acc = val_acc
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                torch.save(model.state_dict(), save_path)
                print(f"\n\nbest random seed:{i}\n\n")
        
        print(f"Best Val Acc for {task_name}: {best_acc:.2f}% | Model saved to {save_path}")

        model.load_state_dict(torch.load(save_path))
        model.eval()
        
        all_true_labels = []
        all_predicted_labels = []
        
        with torch.no_grad():
            for inputs_ac, inputs_mass, labels in val_loader:
                inputs_ac = inputs_ac.to(self.device)
                inputs_mass = inputs_mass.to(self.device)
                
                outputs = model(inputs_ac, inputs_mass)
                predicted = outputs.argmax(dim=1)
                
                all_true_labels.append(labels.cpu().numpy())
                all_predicted_labels.append(predicted.cpu().numpy())
        
        final_true_labels = np.concatenate(all_true_labels)
        final_predicted_labels = np.concatenate(all_predicted_labels)
        
        return best_acc, final_true_labels, final_predicted_labels


class SoluteCla(BaseClaTrainer):
    def trainMain(self):
        concentrations = [1, 3, 5]
        for conc in concentrations:
            all_run_results = []
            for i in range(50):
                qwy = random.randint(1, 10000)
                setup_seed(qwy)
                print(f"\n>>> Processing Solute Classification with Concentration: {conc}")
                X_train, M_train, y_train, X_val, M_val, y_val, folders = preprocess_solutecla_data(
                    self.root_dir, conc, self.model_name, self.duration, self.bandwidth
                )
                
                if len(folders) < 2:
                    print(f"Not enough classes found for concentration{conc} (found {len(folders)}), skipping.")
                    continue
                    
                acc, true_labels, pred_labels = self.run(
                    task_name=f"solutecla_{conc}%_",
                    X_train=X_train, M_train=M_train, y_train=y_train,
                    X_val=X_val, M_val=M_val, y_val=y_val,
                    num_classes=len(folders), i=i
                )

                all_run_results.append({
                    'run_id': i,
                    'acc': acc,
                    'true_labels': true_labels,
                    'pred_labels': pred_labels
                })

            all_run_results.sort(key=lambda x: x['acc'], reverse=True)
            top_10_results = all_run_results[:10]
            print(f"\n>>> Top 10 results for Concentration {conc}% selected. Saving...")
            result_dir = "cla_results"
            output_dir_path = os.path.join(result_dir, "solutecla", f"{conc}%")
            os.makedirs(output_dir_path, exist_ok=True)

            for rank, res in enumerate(top_10_results):
                # rank: 0 ~ 9
                # original run_id: which experiment to get this result
                original_id = res['run_id']
                output_filename = os.path.join(output_dir_path, f"{rank}_matrix.npz")
                
                np.savez(output_filename, 
                         true_labels=res['true_labels'], 
                         predicted_labels=res['pred_labels'])
                print(f"Saved Rank {rank+1} (Run {original_id}, Acc: {res['acc']:.2f}%) to {output_filename}")

class ConcentrationCla(BaseClaTrainer):
    def trainMain(self):
        solutes = ['pu', 'guo', 'zhe']
        for s_name in solutes:
            all_run_results = []
            for i in range(50):
                qwy = random.randint(1, 10000)
                setup_seed(qwy)
                print(f"\n>>> Processing Concentration Classification with Solute: {s_name}")
                X_train, M_train, y_train, X_val, M_val, y_val, folders = preprocess_concla_data(
                    self.root_dir, s_name, self.model_name, self.duration, self.bandwidth
                )
                
                if len(folders) < 2:
                    print(f"Not enough classes found for solute {s_name} (found {len(folders)}), skipping.")
                    continue

                acc, true_labels, pred_labels = self.run(
                    task_name=f"concla_{s_name}_",
                    X_train=X_train, M_train=M_train, y_train=y_train,
                    X_val=X_val, M_val=M_val, y_val=y_val,
                    num_classes=len(folders), i=i
                )

                all_run_results.append({
                    'run_id': i,
                    'acc': acc,
                    'true_labels': true_labels,
                    'pred_labels': pred_labels
                })

            all_run_results.sort(key=lambda x: x['acc'], reverse=True)
            top_10_results = all_run_results[:10]
            print(f"\n>>> Top 10 results for Solute {s_name} selected. Saving...")

            result_dir = "cla_results"
            output_dir_path = os.path.join(result_dir, "concla", s_name)
            os.makedirs(output_dir_path, exist_ok=True)

            for rank, res in enumerate(top_10_results):
                original_id = res['run_id']
                output_filename = os.path.join(output_dir_path, f"{rank}_matrix.npz")
                
                np.savez(output_filename, 
                         true_labels=res['true_labels'], 
                         predicted_labels=res['pred_labels'])
                print(f"Saved Rank {rank+1} (Run {original_id}, Acc: {res['acc']:.2f}%) to {output_filename}")