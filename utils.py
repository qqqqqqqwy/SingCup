import os
import re
import torch
import random
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler

SOLUTE_TYPE = 3
FG_list = ['pu_1_guo_1_zhe_3', 'pu_3_guo_4_zhe_4', 'pu_3_guo_5_zhe_1', 'pu_4_guo_4_zhe_1', 'pu_4_guo_6_zhe_0']

def setup_seed(seed=42):
    random.seed(seed) 
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"The random seed has been fixed to: {seed}")

def generate_labels(folder_names):
    categories = ["pu", "guo", "zhe"]
    num_categories = len(categories)
    labels = np.zeros((len(folder_names), num_categories), dtype=np.float32)
    for i, folder in enumerate(folder_names):
        matches = re.findall(r"(pu|guo|zhe)_(\d+)", folder)
        for cat, value in matches:
            value = float(value) / 100.0
            idx = categories.index(cat)
            labels[i, idx] = value

    return labels

def evaluate_model(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    total_output_sums = torch.zeros(SOLUTE_TYPE).to(device)
    total_density_error = torch.zeros(1).to(device)
    total = 0
    with torch.no_grad():
        for inputs_ac, inputs_mass, labels in dataloader:
            inputs_ac = inputs_ac.to(device)     # (batch, 1, 100, 1500)
            inputs_mass = inputs_mass.to(device) # (batch, 100)
            labels = labels.to(device)
            outputs = model(inputs_ac, inputs_mass)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs_ac.size(0)
            total_output_sums += torch.sum(torch.abs(outputs - labels), dim=0) # 计算绝对误差
            total_density_error += torch.sum(torch.abs(torch.sum(outputs - labels, dim=1)), dim=0) # Sum of solutes error
            total += labels.size(0)
            
    val_loss = running_loss / total
    val_mae = total_output_sums / total
    val_density_mae = total_density_error / total
    return val_loss, val_mae, val_density_mae

def preprocess_data(root_dir, model_name='CoLANet', duration=100, bandwidth=1500):
    folders = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
    y_labels = generate_labels(folders)

    train_data, val_data = [], []

    for folder, label in zip(folders, y_labels):
        folder_path = os.path.join(root_dir, folder)

        # Find all original record files (record0.npy ... record9.npy)
        record_files = sorted([f for f in os.listdir(folder_path) if f.startswith("record") and f.endswith(".npy")])
        record_ids = [int(f.replace("record", "").replace(".npy", "")) for f in record_files]

        # Randomly select 2 IDs as the validation set
        val_ids = np.random.choice(record_ids, size=2, replace=False)

        for rid in record_ids:
            record_file = f"record{rid}.npy"
            mass_file = f"new_weight{rid}.npy"  # Original Mass File
            record_path = os.path.join(folder_path, record_file)
            mass_path   = os.path.join(folder_path, mass_file)

            try:
                data = np.load(record_path)  # (100,1500)
                mass = np.load(mass_path)    # (100,)
                if data.shape != (100, 1500) or mass.shape != (100,):
                    raise ValueError("shape mismatch")

                # Place it in the validation set (containing only the original data).
                if rid in val_ids and not folder in FG_list:
                    val_data.append((data, mass, label))
                else:
                    # The training set contains the raw data.
                    train_data.append((data, mass, label))
                    if "wo_aug" in model_name:
                        continue
                    # The training set should also include augmented data. (aug_xxx_record{rid}.npy)
                    aug_files = sorted([f for f in os.listdir(folder_path) if f.endswith(f"record{rid}.npy") and f.startswith("aug")])
                    for aug_f in aug_files:
                        if ("aug_1" in aug_f or "aug_2" in aug_f) and "wo_GTS" in model_name:
                            continue
                        if ("aug_3" in aug_f or "aug_4" in aug_f) and "wo_TPS" in model_name:
                            continue
                        if ("aug_5" in aug_f) and "wo_NI" in model_name:
                            continue
                        aug_path = os.path.join(folder_path, aug_f)
                        aug_mass_path = aug_f.replace(".npy", "_mass.npy")
                        aug_mass_path = os.path.join(folder_path, aug_mass_path)

                        if not os.path.exists(aug_mass_path):
                            print(f"Lack of enhanced mass file: {aug_mass_path}")
                            continue
                        try:
                            aug_data = np.load(aug_path)
                            aug_mass = np.load(aug_mass_path)
                            if aug_data.shape != (100, 1500) or aug_mass.shape != (100,):
                                raise ValueError("shape mismatch")
                            train_data.append((aug_data, aug_mass, label))
                        except Exception as e:
                            print(f"Processing enhanced file {aug_path} with error: {e}")
                            continue

            except Exception as e:
                print(f"Processing file {record_path} with error: {e}")
                continue

    print(f"Training set {len(train_data)} Validation set {len(val_data)} samples")

    # convert numpy
    X_train = np.array([x[0] for x in train_data])
    M_train = np.array([x[1] for x in train_data])
    y_train = np.array([x[2] for x in train_data])

    X_val   = np.array([x[0] for x in val_data])
    M_val   = np.array([x[1] for x in val_data])
    y_val   = np.array([x[2] for x in val_data])

    X_train = X_train[:,:duration,:bandwidth]
    M_train = M_train[:,:duration]
    X_val = X_val[:,:duration,:bandwidth]
    M_val = M_val[:,:duration]
    # Standardization
    scaler = StandardScaler()
    print('X_val shape:', X_val.shape)
    
    # flatten, then fit, and reshape back to the original state
    X_train = scaler.fit_transform(X_train.reshape(X_train.shape[0], -1)).reshape(X_train.shape)
    X_val   = scaler.transform(X_val.reshape(X_val.shape[0], -1)).reshape(X_val.shape)

    mass_scaler = StandardScaler()
    M_train = mass_scaler.fit_transform(M_train)
    M_val   = mass_scaler.transform(M_val)

    ### Save scaler
    # Use model_name as the filename prefix, consistent with the main module.
    scaler_path = f'./models/normalization/{model_name}.pkl'
    print(f"Saving acoustic scaler to: {scaler_path}")
    os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
    joblib.dump(scaler, scaler_path)
    
    scaler_path_mass = f'./models/normalization/{model_name}_mass.pkl'
    os.makedirs(os.path.dirname(scaler_path_mass), exist_ok=True)
    joblib.dump(mass_scaler, scaler_path_mass)

    return X_train, M_train, y_train, X_val, M_val, y_val, folders


def load_and_process_data_cla(root_dir, target_folders, duration, bandwidth, model_name):
    label_map = {folder: i for i, folder in enumerate(target_folders)}
    num_classes = len(target_folders)
    print(f"Categories ({num_classes}): {target_folders}")

    train_data, val_data = [], []

    for folder in target_folders:
        folder_path = os.path.join(root_dir, folder)
        class_idx = label_map[folder] # Label is an integer index

        record_files = sorted([f for f in os.listdir(folder_path) if f.startswith("record") and f.endswith(".npy")])
        record_ids = [int(f.replace("record", "").replace(".npy", "")) for f in record_files]
        
        if len(record_ids) >= 2:
            val_ids = np.random.choice(record_ids, size=2, replace=False)
        else:
            val_ids = []

        for rid in record_ids:
            record_file = f"record{rid}.npy"
            mass_file = f"new_weight{rid}.npy"
            record_path = os.path.join(folder_path, record_file)
            mass_path   = os.path.join(folder_path, mass_file)

            try:
                data = np.load(record_path)
                mass = np.load(mass_path)
                if data.shape != (100, 1500) or mass.shape != (100,):
                    continue

                if rid in val_ids:
                    val_data.append((data, mass, class_idx))
                else:
                    train_data.append((data, mass, class_idx))
                    
                    if "wo_aug" in model_name: continue

                    aug_files = sorted([f for f in os.listdir(folder_path) if f.endswith(f"record{rid}.npy") and f.startswith("aug")])
                    for aug_f in aug_files:
                        if ("aug_1" in aug_f or "aug_2" in aug_f) and "wo_GTS" in model_name: continue
                        if ("aug_3" in aug_f or "aug_4" in aug_f) and "wo_TPS" in model_name: continue
                        if ("aug_5" in aug_f) and "wo_NI" in model_name: continue
                        
                        aug_path = os.path.join(folder_path, aug_f)
                        aug_mass_path = os.path.join(folder_path, aug_f.replace(".npy", "_mass.npy"))
                        
                        if os.path.exists(aug_mass_path):
                            try:
                                ad = np.load(aug_path)
                                am = np.load(aug_mass_path)
                                if ad.shape == (100, 1500) and am.shape == (100,):
                                    train_data.append((ad, am, class_idx))
                            except: pass
            except: pass

    def process_split(dataset):
        if not dataset: return None, None, None
        X = np.array([x[0] for x in dataset])
        M = np.array([x[1] for x in dataset])
        y = np.array([x[2] for x in dataset])
        X = X[:, :duration, :bandwidth]
        M = M[:, :duration]
        return X, M, y

    X_train, M_train, y_train = process_split(train_data)
    X_val, M_val, y_val = process_split(val_data)
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train.reshape(X_train.shape[0], -1)).reshape(X_train.shape)
    X_val   = scaler.transform(X_val.reshape(X_val.shape[0], -1)).reshape(X_val.shape)

    mass_scaler = StandardScaler()
    M_train = mass_scaler.fit_transform(M_train)
    M_val   = mass_scaler.transform(M_val)
    # Do not save over the initialization file here.
    
    return X_train, M_train, y_train, X_val, M_val, y_val, target_folders

def preprocess_concla_data(root_dir, solute_name, model_name='Solute_cla', duration=100, bandwidth=1500):
    # Filter folders where the solute_name concentration is not zero and all other solutes are zero
    all_folders = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
    target_folders = []
    
    for folder in all_folders:
        # Analyze folder names pu_1_guo_0_zhe_0
        matches = re.search(r"pu_(\d+)_guo_(\d+)_zhe_(\d+)", folder)
        if matches:
            p_val, g_val, z_val = int(matches.group(1)), int(matches.group(2)), int(matches.group(3))
            
            if solute_name == 'pu':
                if p_val != 0 and g_val == 0 and z_val == 0: target_folders.append(folder)
            elif solute_name == 'guo':
                if p_val == 0 and g_val != 0 and z_val == 0: target_folders.append(folder)
            elif solute_name == 'zhe':
                if p_val == 0 and g_val == 0 and z_val != 0: target_folders.append(folder)
    
    target_folders.sort() # Sorting guarantees consistent label order
    return load_and_process_data_cla(root_dir, target_folders, duration, bandwidth, model_name)

def preprocess_solutecla_data(root_dir, concentration, model_name='Concentrations_cla', duration=100, bandwidth=1500):
    # Filter folders where the concentration of a single solute equals the specified concentration
    all_folders = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
    target_folders = []
    
    for folder in all_folders:
        matches = re.search(r"pu_(\d+)_guo_(\d+)_zhe_(\d+)", folder)
        if matches:
            p_val, g_val, z_val = int(matches.group(1)), int(matches.group(2)), int(matches.group(3))
            
            if p_val == concentration and g_val == 0 and z_val == 0:
                target_folders.append(folder)
            elif p_val == 0 and g_val == concentration and z_val ==0:
                target_folders.append(folder)
            elif p_val == 0 and g_val == 0 and z_val == concentration:
                target_folders.append(folder)
                
    target_folders.sort()
    return load_and_process_data_cla(root_dir, target_folders, duration, bandwidth, model_name)

def preprocess_data_sq(root_dir, solute_quantity, model_name='CoLANet', duration=100, bandwidth=1500):
    folders = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
    y_labels = generate_labels(folders)

    train_data, val_data = [], []
    
    # Used to count how many eligible folders were actually loaded.
    filtered_folders_count = 0

    for folder, label in zip(folders, y_labels):
        # label shape is (3,), for example [0.03, 0.00, 0.00]
        # Count the number of solutes with concentrations greater than 0 in the current label.
        current_quantity = np.sum(label > 0)
        
        # If the number of solutes in the current folder does not match the target quantity, skip.
        if current_quantity != solute_quantity:
            continue
        
        filtered_folders_count += 1

        folder_path = os.path.join(root_dir, folder)

        # The following logic is fully consistent with the original preprocess_data.
        record_files = sorted([f for f in os.listdir(folder_path) if f.startswith("record") and f.endswith(".npy")])
        record_ids = [int(f.replace("record", "").replace(".npy", "")) for f in record_files]

        # Randomly select 2 IDs as the validation set.
        val_ids = np.random.choice(record_ids, size=2, replace=False)

        for rid in record_ids:
            record_file = f"record{rid}.npy"
            mass_file = f"new_weight{rid}.npy"
            record_path = os.path.join(folder_path, record_file)
            mass_path   = os.path.join(folder_path, mass_file)

            try:
                data = np.load(record_path)
                mass = np.load(mass_path)
                if data.shape != (100, 1500) or mass.shape != (100,):
                    raise ValueError("shape mismatch")

                # Place it in the validation set (which contains only original data and is not included in FG_list).
                if rid in val_ids and not folder in FG_list:
                    val_data.append((data, mass, label))
                else:
                    train_data.append((data, mass, label))
                    if "wo_aug" in model_name:
                        continue
                    aug_files = sorted([f for f in os.listdir(folder_path) if f.endswith(f"record{rid}.npy") and f.startswith("aug")])
                    for aug_f in aug_files:
                        if ("aug_1" in aug_f or "aug_2" in aug_f) and "wo_GTS" in model_name:
                            continue
                        if ("aug_3" in aug_f or "aug_4" in aug_f) and "wo_TPS" in model_name:
                            continue
                        if ("aug_5" in aug_f) and "wo_NI" in model_name:
                            continue
                        aug_path = os.path.join(folder_path, aug_f)
                        aug_mass_path = aug_f.replace(".npy", "_mass.npy")
                        aug_mass_path = os.path.join(folder_path, aug_mass_path)

                        if not os.path.exists(aug_mass_path):
                            continue
                        try:
                            aug_data = np.load(aug_path)
                            aug_mass = np.load(aug_mass_path)
                            if aug_data.shape != (100, 1500) or aug_mass.shape != (100,):
                                continue
                            train_data.append((aug_data, aug_mass, label))
                        except Exception:
                            continue

            except Exception as e:
                print(f"Processing file {record_path} with error: {e}")
                continue

    print(f"Solute Quantity Filter: {solute_quantity}")
    print(f"Number of eligible folders: {filtered_folders_count}")
    print(f"Number of training set samples: {len(train_data)}, Number of samples in the validation set: {len(val_data)}")

    if len(train_data) == 0:
        raise ValueError(f"No data was found for {solute_quantity} solute. Please check your dataset or parameter settings!")

    # convert numpy
    X_train = np.array([x[0] for x in train_data])
    M_train = np.array([x[1] for x in train_data])
    y_train = np.array([x[2] for x in train_data])

    X_val   = np.array([x[0] for x in val_data])
    M_val   = np.array([x[1] for x in val_data])
    y_val   = np.array([x[2] for x in val_data])

    X_train = X_train[:,:duration,:bandwidth]
    M_train = M_train[:,:duration]
    X_val = X_val[:,:duration,:bandwidth]
    M_val = M_val[:,:duration]
    
    scaler = StandardScaler()
    
    X_train = scaler.fit_transform(X_train.reshape(X_train.shape[0], -1)).reshape(X_train.shape)
    X_val   = scaler.transform(X_val.reshape(X_val.shape[0], -1)).reshape(X_val.shape)

    mass_scaler = StandardScaler()
    M_train = mass_scaler.fit_transform(M_train)
    M_val   = mass_scaler.transform(M_val)

    # scaler_path = f'./models/normalization/{model_name}.pkl'
    # os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
    # joblib.dump(scaler, scaler_path)
    
    # scaler_path_mass = f'./models/normalization/{model_name}_mass.pkl'
    # os.makedirs(os.path.dirname(scaler_path_mass), exist_ok=True)
    # joblib.dump(mass_scaler, scaler_path_mass)

    return X_train, M_train, y_train, X_val, M_val, y_val, folders

def evaluate_static_model(model, dataloader, criterion, device):
        model.eval()
        running_loss = 0.0
        total_output_sums = torch.zeros(SOLUTE_TYPE).to(device)
        total_density_error = torch.zeros(1).to(device)
        total = 0
        
        with torch.no_grad():
            for inputs_ac, inputs_mass, labels in dataloader:
                inputs_ac = inputs_ac.to(device)     # (batch, 1, 100, 1500)
                inputs_mass = inputs_mass.to(device) # (batch, 100)
                labels = labels.to(device)
                
                outputs = model(inputs_ac, inputs_mass)
                
                # 1. Expand labels for Loss calculation: (B, C) -> (B*100, C)
                batch_size = labels.size(0)
                labels_expanded = labels.unsqueeze(1).repeat(1, 100, 1).view(-1, labels.shape[-1])
                loss = criterion(outputs, labels_expanded)
                
                # 2. Average outputs for Metric calculation: (B*100, C) -> (B, C)
                outputs_for_metric = outputs.view(batch_size, 100, -1).mean(dim=1)

                running_loss += loss.item() * inputs_ac.size(0)
                total_output_sums += torch.sum(torch.abs(outputs_for_metric - labels), dim=0) 
                total_density_error += torch.sum(torch.abs(torch.sum(outputs_for_metric - labels, dim=1)), dim=0)
                total += labels.size(0)
                
        val_loss = running_loss / total
        val_mae = total_output_sums / total
        val_density_mae = total_density_error / total
        return val_loss, val_mae, val_density_mae
