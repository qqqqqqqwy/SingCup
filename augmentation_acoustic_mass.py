import numpy as np
import os
import random
from myPlot import spectrogramPlot, massPlot

def randomCompressWhole(S, mass):
    S_t = S.T
    compress_ratio_min = 0.6; compress_ratio_max = 1.2
    D = S_t.shape[1]
    compress_ratio = np.random.uniform(compress_ratio_min, compress_ratio_max)
    # compress_ratio = 1.5
    # compress_ratio = 0.5; 
    C_idx = np.floor(np.arange(0, D, compress_ratio)).astype(int)
    S_t = S_t[:, C_idx]
    mass_t = mass[C_idx]
    
    period_diff = S.shape[0] - S_t.shape[1]; 
    # print(period_diff)
    if compress_ratio <= 1:
        S_t = S_t[:, :S.shape[0]]  # Trim off the excess portion
        mass_t = mass_t[:mass.shape[0]]  # Trim off the excess portion
    else:
        # padding = np.full((S_t.shape[0], period_diff), min_value)
        # S_t = np.hstack((S_t, padpading))
        # S_t = np.hstack((S_t, np.zeros((S_t.shape[0], period_diff))))
        S_t = np.hstack((S_t, np.tile(S_t[:, -1:], (1, period_diff))))
        mass_t = np.hstack((mass_t, np.tile(mass_t[-1], period_diff)))
    return S_t.T, mass_t

######### random accelarate of decelaration
def randomDisturbance(S, mass, T):
    S = S.T
    duration = S.shape[1]*T; t = np.arange(S.shape[1])*T
    compress_ratio_min = 0.5; compress_ratio_max = 1.5

    segment_duration_options = np.arange(1, 2.1, 0.1)  # [1, 1.1, 1.2, ..., 2]
    segment_duration = np.random.choice(segment_duration_options)
    # segment_duration = 7
    segment_frames = int(segment_duration / T)

    anchor = np.random.uniform(0, duration - segment_duration)
    # print(f"Selected anchor point: {anchor:.2f} seconds")

    anchor_idx = np.argmin(np.abs(t - anchor))
    
    start_idx = anchor_idx
    end_idx = anchor_idx + segment_frames
    
    compress_ratio = np.random.uniform(compress_ratio_min, compress_ratio_max)
    # print(f"Anchor {anchor:.2f}s: Compression ratio = {compress_ratio:.2f}")
    
    C_idx = np.floor(np.arange(0, segment_frames, compress_ratio)).astype(int)
    
    S_segment = S[:, start_idx:end_idx]; S_resampled = S_segment[:, C_idx]
    mass_segment = mass[start_idx:end_idx]; mass_resampled = mass_segment[C_idx]
    
    S_r = np.hstack((S[:, :start_idx], S_resampled, S[:, end_idx:]))
    mass_t = np.hstack((mass[:start_idx], mass_resampled, mass[end_idx:]))
    
    period_diff = S.shape[1] - S_r.shape[1]; 
    # print(period_diff)
    if period_diff < 0:
        S_r = S_r[:, :S.shape[1]]  # Trim off the excess portion
        mass_t = mass_t[:mass.shape[0]]  # Trim off the excess portion
    else:
        S_r = np.hstack((S_r, np.tile(S_r[:, -1:], (1, period_diff))))
        mass_t = np.hstack((mass_t, np.tile(mass_t[-1], period_diff)))

    return S_r.T, mass_t

def addSnrNoise(matrix, mass, snr_db=np.random.uniform(15, 20)):
    signal_power = np.mean(np.square(matrix))  # Signal Power
    snr_linear = 10 ** (snr_db / 10)  # Convert to linear scale
    noise_power = signal_power / snr_linear  # Noise power
    noise_std = np.sqrt(noise_power)  # Standard deviation of noise
    noise = np.random.normal(0, noise_std, matrix.shape)  # Generate noise
    matrix_s = matrix + noise
    
    return  matrix_s, mass

def chooseAugmentate(S, mass, T, foldernam, filename):
    """
    Apply random data augmentation methods and save results.
    
    Parameters:
    S: Input signal matrix
    T: Time step
    foldernam: Output directory path
    filename: Base filename for saving
    times: Number of augmentations to generate
    """
    min_value = 1e-4
    # List of available augmentation functions
    aug_functions = [
        ('randomCompressWhole', lambda x1, x2: randomCompressWhole(x1, x2), 2),
        ('randomDisturbance', lambda x1, x2: randomDisturbance(x1, x2, T), 2),
        ('addSnrNoise', lambda x1, x2: addSnrNoise(x1, x2), 1)
    ]
    
    # Ensure output directory exists
    os.makedirs(foldernam, exist_ok=True)
    
    # Remove extension from filename if present
    base_filename = os.path.splitext(filename)[0]
    
    tag = 1
    for aug_name, aug_function, count in aug_functions:
        for _ in range(count):
            S_r, mass_r = aug_function(S, mass)
            S_r = np.where(S_r <= min_value, min_value, S_r)
            # Generate output filenames
            aug_filename = f"aug_{tag}_{aug_name}_{base_filename}.npy"
            audio_output_path = os.path.join(foldernam, aug_filename)
            
            mass_aug_filename = f"aug_{tag}_{aug_name}_{base_filename}_mass.npy"
            mass_output_path = os.path.join(foldernam, mass_aug_filename)
            
            # Save results
            np.save(audio_output_path, S_r)
            np.save(mass_output_path, mass_r)
            tag += 1
        
    return True

if __name__ == '__main__':
    T = 0.1
    filename = r'G:\SoundCup final\largeScaleDataset_jianxi_train\1_chi\record1.npy'
    filename_mass = r'G:\SoundCup final\largeScaleDataset_jianxi_train\1_chi\new_weight1.npy'
    S = np.load(filename)
    mass = np.load(filename_mass)
    # S_n, mass_n = randomCompressWhole(S, mass)
    S_n, mass_n = randomDisturbance(S, mass, T)
    spectrogramPlot(S_n)
    massPlot(mass_n)
