import os
import math
import random
import numpy as np
import soundfile as sf
from pathlib import Path
import matplotlib.pyplot as plt
from augmentation_acoustic_mass import chooseAugmentate
from scipy.ndimage import median_filter, uniform_filter1d

min_value = 1e-4
def process_audio(audio_path, output_audio_path, process_duration=10, frame_rate=10, f0=1000, B=15000):
    fe = f0 + B
    audio_data, sample_rate = sf.read(audio_path)

    total_duration = len(audio_data) / sample_rate 
    frame_duration = 1.0 / frame_rate
    samples_per_frame = round(sample_rate * frame_duration)
    # Actual frames processed
    process_frames = round(process_duration / frame_duration)
    # print(process_frames)
    total_frames = math.floor(total_duration / frame_duration)
    
    # Calculate the starting frame (beginning from the last processed frame)
    # start_frame = max(0, total_frames - process_frames)
    start_frame = 0
    audio_features = []
    for i in range(start_frame, total_frames):
        start = i * samples_per_frame
        end = start + samples_per_frame
        # Ensure that end does not exceed the length of the audio data.
        if end > len(audio_data):
            break
        segment = audio_data[start:end]
        # Compute the Fourier transform and take the magnitude as the feature.
        fft_feat = np.fft.fft(segment)
        idx_f0 = round(f0 * frame_duration)
        idx_fe = round(fe * frame_duration)
        fft_magnitude = np.abs(fft_feat[idx_f0:idx_fe])
        audio_features.append(fft_magnitude)

    while len(audio_features) < process_frames:
        audio_features.append(np.zeros(fft_magnitude.shape) + min_value) # Zero-fill

    audio_features = np.array(audio_features)
    audio_features = audio_features[0:process_frames]
    # print(audio_features.shape)
    np.save(output_audio_path, audio_features) # [20:100]
    # print(f"Audio processing completed. Features was saved to {output_audio_path}")
    return audio_features

def process_weight(filepath, process_duration=10, median_kernel=5, mean_kernel=5):
    Fs_m = 10
    x = np.load(filepath)
    x_med = median_filter(x, size=median_kernel, mode='reflect')
    x_smooth = uniform_filter1d(x_med, size=mean_kernel, mode='reflect')
    
    x = 11 / 10 * np.arange(len(x_smooth))/len(x_smooth)
    x_interp = np.arange(round(process_duration*Fs_m))/(process_duration*Fs_m)
    y_interp = np.interp(x_interp, x, x_smooth)
    np.save(str(filepath).replace('weight', 'new_weight'), y_interp)
    return y_interp

if __name__ == "__main__":
    root = "dataset/robustness"
    for subfolder in os.listdir(root):
        root_dir = os.path.join(root, subfolder)
        for folder_name in os.listdir(root_dir):
            folder_path = os.path.join(root_dir, folder_name)
            if not os.path.isdir(folder_path):
                continue

            folder = Path(folder_path)
            wav_files = list(folder.glob("*.wav"))
            len_wav_files = len(wav_files)
            if len_wav_files > 0:
                aug_npy_files = list(folder.glob("aug_*.npy"))
                for file_path in aug_npy_files:
                    file_path.unlink()

                for m in range(len_wav_files):
                    wav = wav_files[m]
                    record_path = wav
                    weight_path = Path(str(record_path).replace("record", "weight").replace("wav", "npy"))
                    path = Path(wav); i = path.stem[len("record"):]

                    output_audio_filename = f"record{i}.npy"
                    output_audio_path = os.path.join(root_dir, folder_name, output_audio_filename)
                    output_weight_path = Path(str(output_audio_path).replace("record", "weight"))
                    # print(output_audio_path)
                    audio_features = process_audio(record_path, output_audio_path)
                    mass_data = process_weight(weight_path)

    random.seed(42)
    T = 0.1; aug_times = 5
    root_dir = "dataset/merged_dataset_final"
    for folder_name in os.listdir(root_dir):
            folder_path = os.path.join(root_dir, folder_name)
            if not os.path.isdir(folder_path):
                continue

            folder = Path(folder_path)
            wav_files = list(folder.glob("*.wav"))
            len_wav_files = len(wav_files)
            if len_wav_files > 0:
                aug_npy_files = list(folder.glob("aug_*.npy"))
                for file_path in aug_npy_files:
                    file_path.unlink()

                for m in range(len_wav_files):
                    wav = wav_files[m]
                    record_path = wav
                    weight_path = Path(str(record_path).replace("record", "weight").replace("wav", "npy"))
                    path = Path(wav); i = path.stem[len("record"):]

                    output_audio_filename = f"record{i}.npy"
                    output_audio_path = os.path.join(root_dir, folder_name, output_audio_filename)
                    output_weight_path = Path(str(output_audio_path).replace("record", "weight"))
                    # print(output_audio_path)
                    audio_features = process_audio(record_path, output_audio_path)
                    mass_data = process_weight(weight_path)

                    # data augmentation
                    chooseAugmentate(audio_features, mass_data, T, folder_path, output_audio_filename)
