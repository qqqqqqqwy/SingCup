import os
import pandas as pd
import matplotlib.pyplot as plt

def plot_curves_from_csv(csv_path='./MAE/CoLANet_loss.csv'):
    """
    Reads the CSV containing detailed MAE for each solute and plots
    Training and Validation curves separately, following the style of plot_ours.py.
    """
    if not os.path.exists(csv_path):
        print(f"Error: Data file not found at {csv_path}")
        return

    save_dir = './MAE_images'
    os.makedirs(save_dir, exist_ok=True)
    print(f"Loading data from {csv_path}...")
    
    df = pd.read_csv(csv_path)

    plt.rcParams.update({'font.size': 36})
    try:
        plt.style.use('seaborn-v0_8-whitegrid')
    except OSError:
        plt.style.use('seaborn-whitegrid')

    solute_names = ['Glucose', 'Fructose', 'Sucrose'] 
    
    epochs = df['epoch']

    print("Plotting Training Curves...")
    plt.figure(figsize=(12, 8))
    
    for name in solute_names:
        col_name = f'train_{name}_mae'
        if col_name in df.columns:
            plt.plot(epochs, df[col_name], 
                     marker='o', linestyle='-', linewidth=3, markersize=10, 
                     label=f'{name}')
        else:
            print(f"Warning: Column {col_name} not found in CSV.")

    plt.xlabel('Epoch')
    plt.ylabel('MAE %')
    plt.legend(fontsize=30)
    plt.grid(True)
    plt.tight_layout()
    
    save_path_train = os.path.join(save_dir, 'reg_ours_training_loss_curves.eps')
    plt.savefig(save_path_train, format='eps', bbox_inches='tight')
    plt.show()
    print(f"Training curves saved to: {save_path_train}")

    print("Plotting Validation Curves...")
    plt.figure(figsize=(12, 8))
    
    for name in solute_names:
        col_name = f'val_{name}_mae'
        if col_name in df.columns:
            plt.plot(epochs, df[col_name], 
                     marker='o', linestyle='-', linewidth=3, markersize=10, 
                     label=f'{name}')
        else:
            print(f"Warning: Column {col_name} not found in CSV.")
    
    plt.xlabel('Epoch')
    plt.ylabel('MAE %')
    plt.legend(fontsize=30)
    plt.grid(True)
    plt.tight_layout()
    
    save_path_val = os.path.join(save_dir, 'reg_ours_validation_loss_curves.eps')
    plt.savefig(save_path_val, format='eps', bbox_inches='tight')
    plt.show()
    print(f"Validation curves saved to: {save_path_val}")

if __name__ == "__main__":
    plot_curves_from_csv('./MAE/CoLANet_loss.csv')