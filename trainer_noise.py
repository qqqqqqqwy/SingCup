import os
import re
import random
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from model import UNet2D
import torch.optim as optim
from utils import setup_seed
from dataloader import DenoiseDataset
from torch.utils.data import DataLoader

class DenoisedTrainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.root_dir = config.get('root_dir', r"dataset/merged_dataset_final")
        self.noise_dir = r"dataset/noise_npy" 

        self.batch_size = config.get('batch_size', 16)
        self.num_epochs = config.get('num_epochs', 50)
        self.learning_rate = config.get('learning_rate', 1e-3)
        self.model_save_path = "models/Unet_denoise.pth"
        self.robustness_path = config.get('robustness_path', 'dataset/robustness/')
        
        setup_seed(config.get('seed', 42))
        print(f"DenoisedTrainer initialized. Device: {self.device}")

    def trainMain(self):
        print("Running Denoise Training Mode...")
        train_set = DenoiseDataset(self.root_dir, self.noise_dir, mode='train')
        val_set = DenoiseDataset(self.root_dir, self.noise_dir, mode='val')
        
        train_loader = DataLoader(train_set, batch_size=self.batch_size, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_set, batch_size=self.batch_size, shuffle=False, num_workers=4)
        
        model = UNet2D().to(self.device)
        optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)
        criterion = nn.MSELoss()
        
        os.makedirs(os.path.dirname(self.model_save_path), exist_ok=True)
        best_val_loss = float('inf')
        
        for epoch in range(self.num_epochs):
            model.train()
            total_loss = 0.0
            
            with tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.num_epochs} [Train]", leave=False) as pbar:
                for noisy, clean in pbar:
                    noisy, clean = noisy.to(self.device), clean.to(self.device)
                    output = model(noisy)
                    loss = criterion(output, clean)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
                    pbar.set_postfix({'loss': f"{loss.item():.6f}"})
            
            avg_train_loss = total_loss / len(train_loader)
            
            # Validation
            model.eval()
            val_loss_sum = 0.0
            with torch.no_grad():
                for noisy, clean in val_loader:
                    noisy, clean = noisy.to(self.device), clean.to(self.device)
                    output = model(noisy)
                    loss = criterion(output, clean)
                    val_loss_sum += loss.item()
            
            avg_val_loss = val_loss_sum / len(val_loader)
            print(f"[{epoch+1}/{self.num_epochs}] Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")
            
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(model.state_dict(), self.model_save_path)
                print(f"Model saved (Best Val Loss: {best_val_loss:.6f})")

        print("Denoise training complete.")

    def valMain(self):
        print("Running Denoise Generation & Inference Mode...")
        
        if not os.path.exists(self.model_save_path):
            print(f"Error: Model not found at {self.model_save_path}. Please train first.")
            return

        model = UNet2D().to(self.device)
        model.load_state_dict(torch.load(self.model_save_path, map_location=self.device))
        model.eval()
        
        # Label storage path: dataset/robustness/noise
        base_noise_path = os.path.join(self.robustness_path, "noise")
        # Path for storing noisy data: dataset/robustness/noise/noUnet
        noisy_save_path = os.path.join(base_noise_path, "noUnet")
        # Path for storing noise reduction data: dataset/robustness/noise/Denoised
        denoised_save_path = os.path.join(base_noise_path, "Denoised")

        os.makedirs(base_noise_path, exist_ok=True)
        os.makedirs(noisy_save_path, exist_ok=True)
        os.makedirs(denoised_save_path, exist_ok=True)
        
        noise_list = os.listdir(self.noise_dir)
        subfolders = [f for f in os.listdir(self.root_dir) if os.path.isdir(os.path.join(self.root_dir, f))]
        
        if not noise_list or not subfolders:
            print("Error: Missing noise files or data folders.")
            return

        num_samples_to_generate = 10
        random_indices = np.random.choice(len(noise_list), size=len(noise_list), replace=False)
        tag = 0
        
        print(f"Generating {num_samples_to_generate} samples...")
        print(f"  - Labels -> {base_noise_path}")
        print(f"  - Noisy -> {noisy_save_path}")
        print(f"  - Denoised -> {denoised_save_path}")

        for i in range(num_samples_to_generate):
            select_subfolder = random.choice(subfolders)
            
            labels = np.zeros(3, dtype=np.float32) 
            match = re.search(r"pu_(\d+)_guo_(\d+)_zhe_(\d+)", select_subfolder)
            if match:
                labels[0] = float(match.group(1)) # pu
                labels[1] = float(match.group(2)) # guo
                labels[2] = float(match.group(3)) # zhe
            
            select_id = random.randint(0, 9)
            clean_filename = f"record{select_id}.npy"
            weight_filename = f"new_weight{select_id}.npy"
            
            clean_path = os.path.join(self.root_dir, select_subfolder, clean_filename)
            weight_path = os.path.join(self.root_dir, select_subfolder, weight_filename)
            noise_path = os.path.join(self.noise_dir, noise_list[random_indices[tag % len(noise_list)]])
            tag += 1

            if os.path.exists(noise_path) and os.path.exists(clean_path) and os.path.exists(weight_path):
                clean = np.load(clean_path)
                noise = np.load(noise_path)
                weight = np.load(weight_path) # (100,)
                
                if clean.shape != (100, 1500) or noise.shape != (100, 1500):
                    continue
                
                # Mixed noise
                snr_db = np.random.uniform(5, 20)
                signal_power = np.mean(clean ** 2)
                noise_power = np.mean(noise ** 2)
                target_noise_power = signal_power / (10 ** (snr_db / 10))
                scaling_factor = np.sqrt(target_noise_power / (noise_power + 1e-8))                
                noisy_data = clean + scaling_factor * noise
                                
                # 1. Save Label to dataset/robustness/noise/real{i}.npy
                np.save(os.path.join(base_noise_path, f"real{i}.npy"), labels)
                # 2. Save audio containing noise to dataset/robustness/noise/noUnet/record{i}.npy
                np.save(os.path.join(noisy_save_path, f"record{i}.npy"), noisy_data)                
                # 3. Save quality data to dataset/robustness/noise/noUnet/new_weight{i}.npy
                np.save(os.path.join(noisy_save_path, f"new_weight{i}.npy"), weight)

                inp_tensor = torch.tensor(noisy_data, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(self.device)
                
                with torch.no_grad():
                    denoised_output = model(inp_tensor)
                
                denoised_np = denoised_output.cpu().numpy().squeeze() # (100, 1500)
                
                # 4. Save noise-reduced audio to dataset/robustness/noise/Denoised/record{i}.npy
                np.save(os.path.join(denoised_save_path, f"record{i}.npy"), denoised_np)                
                # 5. Save quality data to dataset/robustness/noise/Denoised/new_weight{i}.npy
                np.save(os.path.join(denoised_save_path, f"new_weight{i}.npy"), weight)
                print(f"Sample {i} processed. SNR: {snr_db:.2f}dB")

        print("Validation process completed successfully.")