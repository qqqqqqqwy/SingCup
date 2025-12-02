import os
import torch
import random
import numpy as np
from torch.utils.data import Dataset

class LiquidDataset(Dataset):
    def __init__(self, X, M, y, transform=None, task_type='regression'):
        self.samples = [(X[i], M[i], y[i]) for i in range(len(y))]
        self.transform = transform
        self.task_type = task_type
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sequence_data, mass_data, label = self.samples[idx]

        if self.transform:
            sequence_data = self.transform(sequence_data)

        # Convert to tensor
        sequence_data = torch.tensor(sequence_data, dtype=torch.float32).unsqueeze(0)  # (1, 100, 1500)
        mass_data     = torch.tensor(mass_data, dtype=torch.float32).unsqueeze(-1)       # (100, 1)
        
        if self.task_type == 'classification':
            label = torch.tensor(label, dtype=torch.long)
        else:
            label = torch.tensor(label, dtype=torch.float32) * 100

        return sequence_data, mass_data, label
    
class DenoiseDataset(Dataset):
    def __init__(self, root_dir, noise_dir, mode='train'):
        all_pairs = []
        
        if not os.path.exists(noise_dir):
            raise ValueError(f"Noise directory not found: {noise_dir}")
            
        noise_list = [f for f in os.listdir(noise_dir) if f.endswith('.npy')]
        if len(noise_list) == 0:
            raise ValueError("No .npy noise files found in noise_dir")

        folders = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
        
        print(f"Scanning {len(folders)} folders for denoising dataset...")

        tag = 0
        num_noise = len(noise_list)
        random_indices = np.random.choice(num_noise, size=max(num_noise, 50000), replace=True)

        valid_files_count = 0
        
        for folder in folders:
            folder_path = os.path.join(root_dir, folder)
            for fname in os.listdir(folder_path):
                if fname.startswith("record") and fname.endswith(".npy") and "aug" not in fname and "dp" not in fname:
                    clean_path = os.path.join(folder_path, fname)
                    
                    noise_idx = random_indices[tag % len(random_indices)]
                    noise_path = os.path.join(noise_dir, noise_list[noise_idx])
                    
                    if os.path.exists(clean_path) and os.path.exists(noise_path):
                        all_pairs.append({
                            'clean_path': clean_path,
                            'noise_path': noise_path,
                            'snr_db': np.random.uniform(5, 20) # Random select SNR
                        })
                        tag += 1
                        valid_files_count += 1

        print(f"Total valid pairs found: {valid_files_count}")
        
        random.shuffle(all_pairs)
        split = int(0.8 * len(all_pairs))
        
        if mode == 'train':
            self.pairs = all_pairs[:split]
        elif mode == 'val':
            self.pairs = all_pairs[split:]
        else:
            raise ValueError("mode must be 'train' or 'val'")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        item = self.pairs[idx]
        
        clean = np.load(item['clean_path'])
        noise = np.load(item['noise_path'])
        snr_db = item['snr_db']
        
        if clean.shape != (100, 1500):
            clean = clean[:100, :1500] 
        
        if noise.shape != (100, 1500):
             pass

        signal_power = np.mean(clean ** 2)
        noise_power = np.mean(noise ** 2)
        target_noise_power = signal_power / (10 ** (snr_db / 10))
        scaling_factor = np.sqrt(target_noise_power / (noise_power + 1e-8))
        
        noisy = clean + scaling_factor * noise
        
        # Add Channel Dimension [1, 100, 1500]
        noisy = noisy[None, :, :]
        clean = clean[None, :, :]
        
        return (
            torch.tensor(noisy, dtype=torch.float32),
            torch.tensor(clean, dtype=torch.float32)
        )