import os
import csv
import torch
import joblib
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from model import *
from utils import *
from dataloader import LiquidDataset
from torch.utils.data import DataLoader
from collections import defaultdict

class LiquidTrainer:
    def __init__(self, config): # param config: Dictionary containing hyperparameters and path configurations
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.bandwidth = config.get('bandwidth', 1500)
        self.duration = config.get('duration', 100)
        self.model_name = config.get('model_name', 'CoLANet')
        self.root_dir = config.get('root_dir', r"dataset/merged_dataset_final")
        self.batch_size = config.get('batch_size', 32)
        self.num_epochs = config.get('num_epochs', 70)
        self.learning_rate = config.get('learning_rate', 0.001)
        self.solute_quantity = config.get('solute_quantity', None)

        if self.solute_quantity != None:
            self.model_name += f"_quan{self.solute_quantity}"
        if self.bandwidth != 1500:
            self.model_name += f"_band{self.bandwidth}"
        if self.duration != 100:
            self.model_name += f"_dura{self.duration}"

        self.model_save_path = f'./models/{self.model_name}.pth'
        self.ac_scaler_path = f'./models/normalization/{self.model_name}.pkl'
        self.mass_scaler_path = f'./models/normalization/{self.model_name}_mass.pkl'
        
        self.MAE_csv_path = f'./MAE/{self.model_name}_loss.csv'
        self.table_csv_path = f'./table/{self.model_name}_table.csv'
        
        setup_seed(config.get('seed', 42))
        print(f"Trainer initialized on device: {self.device} | Model: {self.model_name}")

    def build_model(self):
        if self.model_name == "CoLANet_wo_mass":
            model = CNNLSTMNet_Fusion_wo_mass(ac_size=self.bandwidth, num_classes=SOLUTE_TYPE, input_channels=1).to(self.device)
        elif self.model_name == "CoLANet_wo_lstm":
            model = CNNLSTMNet_Fusion_wo_lstm(ac_size=self.bandwidth, num_classes=SOLUTE_TYPE, input_channels=1).to(self.device)
        elif self.model_name == "CoLANet_wo_attn":
            model = CNNLSTMNet_Fusion_wo_attn(ac_size=self.bandwidth, num_classes=SOLUTE_TYPE, input_channels=1).to(self.device)
        elif self.model_name == "ResNet18_pt":
            model = ResNet18_pt(ac_size=self.bandwidth, num_classes=SOLUTE_TYPE, input_channels=1).to(self.device)
        elif self.model_name == "ResNet18":
            model = ResNet18(ac_size=self.bandwidth, num_classes=SOLUTE_TYPE, input_channels=1).to(self.device)
        elif self.model_name == "TCN":
            model = TCN(ac_size=self.bandwidth, num_classes=SOLUTE_TYPE, input_channels=1).to(self.device)
        elif self.model_name == "static_channel":
            model = static_channel(ac_size=self.bandwidth, num_classes=SOLUTE_TYPE, input_channels=1).to(self.device)
        elif self.model_name == "MVUE":
            model = MVUE(ac_size=self.bandwidth, num_classes=SOLUTE_TYPE, input_channels=1).to(self.device)
        else:
            model = CNNLSTMNet_Fusion(ac_size=self.bandwidth, num_classes=SOLUTE_TYPE, input_channels=1).to(self.device)
        criterion = nn.SmoothL1Loss()
        optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=3
        )
        return model, criterion, optimizer, scheduler

    def trainMain(self):
        print(f"Running Training Mode...")
        # Data Preprocessing
        if self.solute_quantity == None:
            X_train, M_train, y_train, X_val, M_val, y_val, folders = preprocess_data(
                self.root_dir, model_name=self.model_name, bandwidth=self.bandwidth, duration=self.duration
            )
        else:
            X_train, M_train, y_train, X_val, M_val, y_val, folders = preprocess_data_sq(
                self.root_dir, solute_quantity=self.solute_quantity, model_name=self.model_name, bandwidth=self.bandwidth, duration=self.duration
            )
        
        train_dataset = LiquidDataset(X_train, M_train, y_train)
        val_dataset   = LiquidDataset(X_val, M_val, y_val)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=0)
        val_loader   = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=0)
        
        model, criterion, optimizer, scheduler = self.build_model()
        
        print("Starting training loop...")
        
        os.makedirs(os.path.dirname(self.MAE_csv_path), exist_ok=True)
        solute_names = ['Glucose', 'Fructose', 'Sucrose']
        header = ['epoch'] + [f'train_{s}_mae' for s in solute_names] + [f'val_{s}_mae' for s in solute_names]

        with open(self.MAE_csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            
        total_steps = self.num_epochs * len(train_loader)
        
        with tqdm(total=total_steps, desc=f"Training {self.model_name}", unit="step") as pbar:
            for epoch in range(self.num_epochs):
                model.train()
                running_loss = 0.0
                
                total_output_sums = torch.zeros(SOLUTE_TYPE).to(self.device)
                total_density_error = torch.zeros(1).to(self.device)
                total = 0
                
                for inputs_ac, inputs_mass, labels in train_loader:
                    inputs_ac = inputs_ac.to(self.device)
                    inputs_mass = inputs_mass.to(self.device)
                    labels = labels.to(self.device)
                    
                    optimizer.zero_grad()
                    outputs = model(inputs_ac, inputs_mass)

                    if self.model_name == "static_channel":
                        # 1. Expand labels for Loss calculation: (B, C) -> (B, 100, C) -> (B*100, C)
                        # train on every single frame as an independent sample
                        batch_size = labels.size(0)
                        labels_expanded = labels.unsqueeze(1).repeat(1, 100, 1).view(-1, labels.shape[-1])
                        loss = criterion(outputs, labels_expanded)

                        # 2. Reshape outputs for Metric calculation: (B*100, C) -> (B, C)
                        # aggregate (mean) the 100 frames to get the sample-level prediction for MAE
                        outputs_for_metric = outputs.view(batch_size, 100, -1).mean(dim=1)
                    else:
                        loss = criterion(outputs, labels)
                        outputs_for_metric = outputs

                    loss.backward()
                    optimizer.step()
                    
                    running_loss += loss.item() * inputs_ac.size(0)
                    total_output_sums += torch.sum(torch.abs(outputs_for_metric - labels), dim=0)
                    total_density_error += torch.sum(torch.abs(torch.sum(outputs_for_metric - labels, dim=1)), dim=0)
                    total += labels.size(0)
                    
                    pbar.update(1)
                    pbar.set_postfix({'epoch': f"{epoch+1}/{self.num_epochs}", 'loss': f"{loss.item():.4f}"})
                
                epoch_loss = running_loss / total
                epoch_train_mae_per_solute = total_output_sums / total
                train_mae_list = epoch_train_mae_per_solute.cpu().tolist()

                if self.model_name == "static_channel":
                    epoch_val_loss, epoch_val_mae, epoch_val_density_mae = evaluate_static_model(
                        model, val_loader, criterion, self.device
                    )
                else:
                    epoch_val_loss, epoch_val_mae, epoch_val_density_mae = evaluate_model(
                        model, val_loader, criterion, self.device
                    )

                val_mae_list = epoch_val_mae.cpu().tolist()
                
                with open(self.MAE_csv_path, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([epoch + 1] + train_mae_list + val_mae_list)
                
                scheduler.step(epoch_loss)
                train_avg = sum(train_mae_list) / len(train_mae_list)
                val_avg = sum(val_mae_list) / len(val_mae_list)
                log_msg = (
                    f"Epoch {epoch+1} | "
                    f"Train Loss: {epoch_loss:.4f} | Train Avg MAE: {train_avg:.4f} | "
                    f"Val Avg MAE: {val_avg:.4f}"
                )
                pbar.write(log_msg)

        print("Training complete.")
        
        os.makedirs(os.path.dirname(self.model_save_path), exist_ok=True)
        torch.save(model.state_dict(), self.model_save_path)
        print(f"Save model to: {self.model_save_path}")

        os.makedirs(os.path.dirname(self.table_csv_path), exist_ok=True)
        print(f"Saving metrics table to {self.table_csv_path}...")
        with open(self.table_csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Metric', 'Glucose_MAE', 'Fructose_MAE', 'Sucrose_MAE', 'Average_MAE', 'Density_MAE'])
            
            val_mae_list = epoch_val_mae.cpu().tolist()
            val_average_mea_item = epoch_val_mae.mean().item()
            val_density_item = epoch_val_density_mae.item()
            
            row = ['Value'] + [f"{x:.4f}" for x in val_mae_list] + [f"{val_average_mea_item:.4f}"] + [f"{val_density_item:.4f}"]
            writer.writerow(row)

    def valMain(self):
        print(f"Running Inference Mode (Recursive) | Model: {self.model_name}")
        
        if not os.path.exists(self.model_save_path):
            print(f"Error: Model file not found at {self.model_save_path}")
            return

        if not os.path.exists(self.ac_scaler_path) or not os.path.exists(self.mass_scaler_path):
            print("Error: Scaler files not found. Please run training first.")
            return

        try:
            ac_scaler = joblib.load(self.ac_scaler_path)
            mass_scaler = joblib.load(self.mass_scaler_path)
            print(f"Scalers loaded from {os.path.dirname(self.ac_scaler_path)}")
        except Exception as e:
            print(f"Error loading scalers: {e}")
            return

        model = CNNLSTMNet_Fusion(ac_size=self.bandwidth, num_classes=SOLUTE_TYPE, input_channels=1).to(self.device)
        
        try:
            model.load_state_dict(torch.load(self.model_save_path, map_location=self.device, weights_only=True))
            model.eval()
            print(f"Model weights loaded successfully from {self.model_save_path}")
        except Exception as e:
            print(f"Error loading model weights: {e}")
            return

        root_dir = self.config.get('robustness_path', 'dataset/robustness/')
        print(f"Scanning root directory: {root_dir}")

        if not os.path.isdir(root_dir):
            print(f"Error: Root directory '{root_dir}' does not exist.")
            return

        processed_folders = 0
        
        for dirpath, dirnames, filenames in os.walk(root_dir):
            has_records = any(f"record{i}.npy" in filenames for i in range(4))
            
            if has_records:
                print(f"\nProcessing directory: {dirpath}")
                processed_folders += 1
                
                for k in range(10): # record0 ~ record9
                    acoustic_filename = f'record{k}.npy'
                    mass_filename = f'new_weight{k}.npy'
                    output_filename = f'out_put{k}.npy'
                    
                    acoustic_path = os.path.join(dirpath, acoustic_filename)
                    mass_path = os.path.join(dirpath, mass_filename)
                    output_path = os.path.join(dirpath, output_filename)
                    
                    if os.path.exists(acoustic_path) and os.path.exists(mass_path):
                        try:
                            data_ac = np.load(acoustic_path)
                            data_mass = np.load(mass_path)

                            if data_mass.ndim == 1:
                                data_mass = data_mass.reshape(1, -1) # (100,) -> (1, 100)
                            
                            if data_ac.shape[1] != self.bandwidth:
                                print(f"  [Skip] Shape mismatch in {acoustic_filename}: {data_ac.shape}")
                                continue
                            
                            data_ac_flat = data_ac.reshape(1, -1)
                            data_ac_scaled = ac_scaler.transform(data_ac_flat)
                            data_mass_scaled = mass_scaler.transform(data_mass)
                            data_ac_scaled = data_ac_scaled.reshape(data_ac.shape)
                            data_ac_tensor = torch.tensor(data_ac_scaled, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(self.device)
                            data_mass_tensor = torch.tensor(data_mass_scaled, dtype=torch.float32).unsqueeze(-1).to(self.device)

                            with torch.no_grad():
                                output = model(data_ac_tensor, data_mass_tensor)
                            
                            output_numpy = output.cpu().numpy() # (1, SOLUTE_TYPE)
                            
                            if output_numpy.shape[0] == 1:
                                output_numpy = output_numpy.squeeze(0)
                                
                            np.save(output_path, output_numpy)
                            print(f"  Saved: {output_filename}")
                            
                        except Exception as e:
                            print(f"  Error processing {acoustic_filename}: {e}")
                    
        print(f"\nInference complete. Processed {processed_folders} directories.")
