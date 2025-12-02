import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import os

def plot_avg_confusion_matrix(folder="cla_results/solutecla/1%", qwy=1):
    class_labels = [f'{qwy}%A',f'{qwy}%B',f'{qwy}%C']
    num_classes = len(class_labels)
    label_order_for_cm = [0, 1, 2]
    
    cm_total = np.zeros((num_classes, num_classes), dtype=np.float64)
    file_count = 0
    
    # traverse 0~9
    for i in range(10):
        filepath = os.path.join(folder, f"{i}_matrix.npz")
        if not os.path.exists(filepath):
            print(f"WARNING: file {filepath} not exists, skip.")
            continue
        try:
            data = np.load(filepath)
            true_labels = data['true_labels']
            predicted_labels = data['predicted_labels']
            cm = confusion_matrix(true_labels, predicted_labels, labels=label_order_for_cm)
            cm_total += cm
            file_count += 1
        except Exception as e:
            print(f"loading {filepath} fails: {e}")
    
    if file_count == 0:
        print("Error: No valid confusion matrix file found.")
        return
    
    # Calculate the average confusion matrix
    cm_avg = cm_total / file_count

    # Normalize by row to percentage
    cm_sum = cm_avg.sum(axis=1)[:, np.newaxis]
    epsilon = 1e-8
    cm_normalized = cm_avg.astype('float') / (cm_sum + epsilon) * 100
    
    # Threshold annotation
    threshold = 1.0
    annot_labels = np.full_like(cm_normalized, "", dtype=object)
    for i in range(cm_normalized.shape[0]):
        for j in range(cm_normalized.shape[1]):
            if cm_normalized[i, j] > threshold:
                annot_labels[i, j] = f"{cm_normalized[i, j]:.2f}%"

    # Draw heat map
    plt.style.use('default')
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    
    fig = plt.figure(figsize=(30, 24))
    ax = sns.heatmap(
        cm_normalized,
        annot=annot_labels,
        fmt='s',
        cmap='Blues',
        xticklabels=class_labels,
        yticklabels=class_labels,
        linewidths=.8,
        annot_kws={"size": 70}
    )
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=100) 
    ax.set_xlabel('Predicted Label', labelpad=15, fontsize=140, fontweight=100)
    ax.set_ylabel('True Label', labelpad=15, fontsize=140, fontweight=100)
    plt.xticks(rotation=45, ha='right', fontsize=120, fontweight=100)
    plt.yticks(rotation=0, fontsize=120, fontweight=100)

    output_filename = f'confusion_matrix.eps'
    output_path = os.path.join(real_folder, output_filename)
    plt.savefig(output_path, format='eps', bbox_inches='tight')
    print(f"The average confusion matrix image has been saved as a vector image '{output_path}' (base on {file_count} files)")
    plt.show()

if __name__ == '__main__':
    root_dir = "cla_results/solutecla"
    i=1
    for folder in os.listdir(root_dir):
        real_folder = os.path.join(root_dir, folder)
        plot_avg_confusion_matrix(folder=real_folder, qwy=i)
        i+=2