import os
import librosa
import numpy as np

from featureExtractionConfig import *

def calculate_hfd(audio_signal, k_max):
    """
    Calculate the Higuchi Fractal Dimension of an audio signal.
    :param audio_signal: The audio signal from which to calculate the HFD.
    :param k_max: The maximum scale of analysis.
    :return: The HFD of the audio signal.
    """
    print("Calculating Higuchi Fractal Dimension...")
    L = []
    x = np.asarray(audio_signal)
    N = x.size

    for k in range(1, k_max):
        Lk = []
        for m in range(k):
            Lmk = 0
            for i in range(1, int(np.floor((N - m) / k))):
                Lmk += abs(x[m + i * k] - x[m + (i - 1) * k])
            Lmk = Lmk * (N - 1) / np.floor((N - m) / k) / k
            Lk.append(Lmk)
        L.append(np.log(np.mean(Lk)))
    L = np.asarray(L)

    k = np.log(np.arange(1, k_max))
    HFD, _ = np.polyfit(k, L, 1)
    return HFD

def process_directory(input_dir, output_dir, k_max=10):
    """
    Process all WAV files in a directory to extract their Higuchi Fractal Dimension.
    :param input_dir: Directory containing WAV files.
    :param output_dir: Directory to save HFD values.
    :param k_max: Maximum scale of analysis for HFD calculation.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    file_count = 0

    for file_name in os.listdir(input_dir):
        if file_name.endswith('.wav'):
            file_count += 1
            print(f"Opening {file_name}...")
            
            file_path = os.path.join(input_dir, file_name)
            y, sr = librosa.load(file_path, sr=None)
            
            hfd = calculate_hfd(y, k_max)
            
            file_id = file_name.split('.')[0]
            output_file_path = os.path.join(output_dir, f"{file_id}_HFD.npy")
            np.save(output_file_path, np.array([hfd]))
            
            print(f"Processed {file_name} to: {output_file_path}")
    print(f"Processed {file_count} files... Feature extraction complete.")

input_directory = RAW_WAV_PATH
output_directory = HFD_FEATURE_OUTPUT

process_directory(input_directory, output_directory)
