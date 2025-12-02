import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

def summarize_and_plot_results(root_dir='dataset/robustness'):
    # Traverse the subfolders i in the specified root directory,
    # and then group them according to the subfolders j of i, generating a grouped boxplot for each i.
    true_output = np.array([0., 0., 0.])
    print(f"Start analyzing root directory: '{root_dir}'")
    
    if not os.path.isdir(root_dir):
        print(f"Error: The root directory '{root_dir}' does not exist.")
        return

    # Get all home folders 'i'
    main_subdirectories = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
    
    for i in main_subdirectories:
        dir_i_path = os.path.join(root_dir, i)
        print(f"Processing main folder: {dir_i_path}")

        # Retrieve all subfolders j under i and sort them to ensure consistent order
        subdirs_j = sorted([d for d in os.listdir(dir_i_path) if os.path.isdir(os.path.join(dir_i_path, d))])
        if i=="temporal_stability":
            subdirs_j=["7days","14days","21days"]

        if i=="different_people":
            subdirs_j=["M1", "M2", "M3", "F1", "F2"]

        if i=="different_temperature":
            subdirs_j=["5°C", "40°C", "60°C"]

        if i == "different_drinks":
            subdirs_j = [
                "cola_no_gas",    # 1
                "qipaoshui",      # 2
                "binghongcha",    # 3
                "chengzhi_kuer",  # 4
                "dongfangshuye",  # 5
                "jiadele"         # 6
            ]
        
        if not subdirs_j:
            print(f"No subfolders were found in '{dir_i_path}', skip.")
            continue
    
        # Use a dictionary to store error data by sub folder name
        errors_by_subdir = {}
        
        for j_name in subdirs_j:
            dir_j_path = os.path.join(dir_i_path, j_name)
            errors_for_this_j = []
            
            # Traverse all files inside the current subfolders j
            for subdir, _, files in os.walk(dir_j_path):
                for file in files:
                    if 'out_put' in file and file.endswith('.npy'):
                        true_output = np.array([0.0, 0.0, 0.0])
                        if i == "noise":
                            tmp_id = int(file.replace("out_put","").replace(".npy",""))
                            real_path = os.path.join(dir_i_path, f"real{tmp_id}.npy")
                            true_output = np.load(real_path)
                            # print(true_output)
                        
                        if i == "different_drinks":
                            if j_name == "cola_no_gas":      # 1.
                                true_output = np.array([3.6, 5.4, 0.0])
                            elif j_name == "qipaoshui":      # 2.
                                true_output = np.array([0.0, 0.0, 0.0])
                            elif j_name == "binghongcha":    # 3.
                                true_output = np.array([3.3, 4.0, 0.78])
                            elif j_name == "chengzhi_kuer":  # 4.
                                true_output = np.array([3.1, 4.5, 1.1])
                            elif j_name == "dongfangshuye":  # 5.
                                true_output = np.array([0.0, 0.0, 0.0])
                            elif j_name == "jiadele":        # 6.
                                true_output = np.array([1.0, 0.6, 2.6])

                        file_path = os.path.join(subdir, file)
                        try:
                            prediction = np.load(file_path)
                            error = prediction.flatten() - true_output
                            if error.shape == (3,):
                                errors_for_this_j.append(error)
                        except Exception as e:
                            print(f"There was an error when processing file {file_path}: {e}")
            
            if errors_for_this_j:
                errors_by_subdir[j_name] = np.array(errors_for_this_j)
                solute_labels_original = ['Glucose', 'Fructose', 'Sucrose']
                all_errors_for_j = np.array(errors_for_this_j)
                mae_per_solute = np.mean(np.abs(all_errors_for_j), axis=0)
                print(f"MAE for subdirectory '{j_name}':")
                for k in range(len(mae_per_solute)):
                    print(f"{solute_labels_original[k]}: {mae_per_solute[k]:.4f}")
                all_errors_for_j = np.array(errors_for_this_j)
                mae = np.mean(np.abs(all_errors_for_j))
                print(f"MAE for '{j_name}': {mae:.2f}")

        if not errors_by_subdir:
            print(f" No 'out_put *. npy' files were found in the subfolders of '{dir_i_path}', skip drawing.")
            continue

        try:
            fig, ax = plt.subplots(figsize=(20, 9))
            
            # Define the color and label of solutes, glucose, fructose, sucrose
            solute_labels = ['A', 'B', 'C']
            colors = ['#FF9999', '#66B2FF', '#99FF99'] # RGB
            
            # Define the original column index order of A, B and C.
            reorder_indices = [0, 1, 2]
            
            all_plot_data = []
            positions = []
            
            # Calculate the position of each group on the X-axis
            num_subgroups = len(solute_labels)
            num_groups = len(subdirs_j)
            
            for group_idx, j_name in enumerate(subdirs_j):
                # Retrieve the data of the current group
                group_data = errors_by_subdir.get(j_name)
                if group_data is None:
                    continue
                
                # Reorder data columns to match A,B,C
                group_data_reordered = group_data[:, reorder_indices]
                
                # Calculate the positions of the three boxes within the current group
                # (num_subgroups + 2) is to leave more gaps between groups
                base_pos = group_idx * (num_subgroups + 2)
                pos_for_group = [base_pos + k for k in range(num_subgroups)]
                
                all_plot_data.extend([group_data_reordered[:, k] for k in range(num_subgroups)])
                positions.extend(pos_for_group)

            if not all_plot_data:
                print(f"There is no plotting data available for'{i}'")
                continue

            # Draw all boxplots
            bp = ax.boxplot(all_plot_data, positions=positions, patch_artist=True, widths=0.8, showfliers=False)

            # Color each box diagram
            for k in range(len(bp['boxes'])):
                bp['boxes'][k].set_facecolor(colors[k % num_subgroups])

            # Set Chart Aesthetics
            ax.axhline(0, color='red', linestyle='--', linewidth=0.8)
            ax.tick_params(axis='y', labelsize=60)
            ax.tick_params(axis='x', labelsize=60)
            ax.set_ylabel('Error (g/100ml)', fontsize=60)
            if "drinks" in i:
                ax.set_xlabel('Beverage Index', fontsize=60)

            if i=="different_people":
                ax.set_xlabel('User', fontsize=60)
            
            if i=="different_temperature":
                ax.set_xlabel('Temperature', fontsize=60)

            # 设置X轴刻度和标签
            tick_positions = [idx * (num_subgroups + 2) + (num_subgroups - 1) / 2 for idx in range(num_groups)]
            ax.set_xticks(tick_positions)
            ax.set_xticklabels(subdirs_j)

            if i == "noise":
                ax.set_xticklabels(["Denoised","w/o. denoising"])

            if i == "different_drinks":
                ax.set_xticklabels(["1", "2", "3", "4", "5", "6"])
            
            # Create and add legend
            solute_labels = ['Glucose', 'Fructose', 'Sucrose']
            legend_elements = [Patch(facecolor=colors[k], label=solute_labels[k]) for k in range(num_subgroups)]
            ax.legend(handles=legend_elements, fontsize=35, loc='upper center', ncol=3, frameon=False)
            
            ax.grid(axis='y', linestyle='--', alpha=1)
            plt.tight_layout() # Automatically adjust layout to prevent label overflow

            # Save image
            output_filename = f'{i}_box.eps'
            output_path = os.path.join(dir_i_path, output_filename)
            plt.savefig(output_path, format='eps', bbox_inches='tight')
            plt.show()
            plt.close(fig)            
            print(f"The grouped box diagram has been saved to: {output_path}")

        except Exception as e:
            print(f"Failed to generate or save image for {dir_i_path}: {e}")

if __name__ == '__main__':
    summarize_and_plot_results()