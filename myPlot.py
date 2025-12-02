import numpy as np
import matplotlib.pyplot as plt

def spectrogramPlot(data, filename=''):
    # Basic Parameters
    T = 0.1     # Time per frame (seconds)
    # Process the spectrogram
    S = data.T
    S = 20 * np.log10(S + 1e-8)

    # Time and frequency axes
    t_interval = np.arange(S.shape[1]) * T
    f_range = np.arange(S.shape[0]) / T / 1e3 + 1

    # Create and save the plot
    plt.figure()
    plt.imshow(S, extent=[t_interval[0], t_interval[-1], f_range[0], f_range[-1]], 
                aspect='auto', origin='lower', cmap='jet', vmin=-80, vmax=30)
    plt.ylim([1, 16])
    plt.yticks(fontsize=12, fontname='Arial')
    plt.ylabel('Frequency (kHz)', fontsize=12, fontname='Arial')
    plt.gcf().set_size_inches(10/2.54, 7/2.54)
    plt.tight_layout()

    # Save the plot
    if len(filename) > 0:
        output_path = filename.replace("wav", "png")
        plt.savefig(output_path, format='png')
        plt.close()
    else:
        plt.show()

def massPlot(data, filename=''):
    plt.figure()
    plt.plot(np.arange(len(data))/10, data)
    
    plt.ylim([0, 300])
    plt.ylabel('Mass (g)', fontsize=12, fontname='Arial')
    plt.xlabel('Time (s)', fontsize=12, fontname='Arial')
    plt.gcf().set_size_inches(10/2.54, 7/2.54)
    
    if len(filename) > 0:
        output_path = filename.replace("npy", "png"); 
        plt.savefig(output_path, format='png')
        plt.close()
    else:
        plt.show()