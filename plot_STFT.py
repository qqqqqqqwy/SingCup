import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft
from scipy.io import wavfile

# Basic Parameters
Fs = 48000  # sampling rate
T = 0.1     # Time per frame (seconds)
N = round(Fs * T)  # Number of sampling points per frame
minterp = 1 # No interpolation
Duration = 10 # Length of each recording (seconds)

data_root = "all_data"
subfolders = ["1_zhe","3_zhe",'5_zhe','7_zhe','12_zhe']

# Traverse all subfolders
for folder_name in subfolders:
    folder_path = os.path.join(data_root, folder_name)
    if not os.path.isdir(folder_path):
        continue

    # Traverse record{i}.wav and new_weight{i}.npy
    for i in range(19):
        record_path = os.path.join(folder_path, f'record{i}.wav')
        weight_path = os.path.join(folder_path, f'new_weight{i}.npy')

        if not (os.path.exists(record_path) and os.path.exists(weight_path)):
            continue

        # Load audio
        try:
            fs, y = wavfile.read(record_path)
            if y.ndim > 1:
                y = y[:, 0]  # Only take the first channel
            y = y.astype(np.float32)
            if fs != Fs:
                print(f"{record_path} sampling rate {fs} compared to the preset {Fs} inconsistent, skip")
                continue
        except Exception as e:
            print(f"loading {record_path} fails: {e}")
            continue

        m_data = np.load(weight_path)

        # Check audio length
        if len(y) < N * 2:
            print(f" {record_path} audio too short, skip")
            continue

        # Draw time-frequency plot (STFT style)
        D = int(len(y) / N) - 1
        S = []

        for m in range(D):
            y_temp = y[m * N:(m + 1) * N]
            fft_temp = fft(y_temp, minterp * N)
            S.append(np.abs(fft_temp[:round(N * minterp / 2)]))

        S = np.array(S).T
        S = 20 * np.log10(S + 1e-12)

        t_interval = np.arange(S.shape[1]) * T / 2
        f_range = np.arange(S.shape[0]) / T / minterp / 1e3

        plt.figure()
        plt.imshow(S, extent=[t_interval[0], t_interval[-1], f_range[0], f_range[-1]], 
                   aspect='auto', origin='lower', cmap='jet', vmin=-80, vmax=30)
        # plt.ylim([0, 20])
        plt.yticks(fontsize=12)
        plt.xticks([])
        plt.ylabel('Frequency (kHz)', fontsize=12)
        plt.gcf().set_size_inches(10/2.54, 7/2.54)
        plt.tight_layout()

        stft_basename = f'STFT_record{i}'
        # plt.savefig(os.path.join(folder_path, stft_basename + '.eps'), format='eps')
        plt.savefig(os.path.join(folder_path, stft_basename + '.png'), format='png')
        plt.close()

        # Draw a quality change curve
        len_m = len(m_data)
        t_m_interval = np.arange(len_m) / len_m * Duration

        plt.figure()
        plt.plot(t_m_interval, m_data, color='r', linewidth=2, marker='o', markersize=3)
        plt.grid(True)
        plt.ylabel('Mass (g)', fontsize=12)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.gcf().set_size_inches(10/2.54, 3.5/2.54)
        plt.tight_layout()

        mass_basename = f'new_weight{i}'
        # plt.savefig(os.path.join(folder_path, mass_basename + '.eps'), format='eps')
        plt.savefig(os.path.join(folder_path, mass_basename + '.png'), format='png')
        plt.close()