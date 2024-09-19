import os
import soundfile as sf
import noisereduce as nr
import numpy as np
import warnings
from featureExtractionConfig import *

input_directory = RAW_WAV_PATH
output_directory = NOISE_REDUCED_OUTPUT

def convert_to_mono(data):
    if len(data.shape) == 2:  # Check if the file is stereo
        return np.mean(data, axis=1)  # Averaging the two channels
    else:
        return data  # The file is already mono

def process_file(file_path, output_path):
    try:
        print(f"Starting processing: {file_path}")
        data, rate = sf.read(file_path)
        data = convert_to_mono(data)  # Convert to mono
        print(f"File read successfully. Data shape: {data.shape}, Rate: {rate}")

        if np.any(data) and not np.all((data == 0)):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                reduced_noise = nr.reduce_noise(y=data, sr=rate,
                                                n_fft=1024, win_length=512, 
                                                hop_length=256, prop_decrease=0.7)
                print(f"Noise reduction completed.")

            sf.write(output_path, reduced_noise, rate)
            print(f"File written to output: {output_path}")
        else:
            print(f"Skipping silent or near-silent file: {file_path}")
    except Exception as e:
        print(f"Error processing {file_path}: {e}")

if not os.path.exists(output_directory):
    os.makedirs(output_directory)

problematic_file = '5e28af994576770678834ae0-2023-09-06T14-17-04Z.wav'
for filename in os.listdir(input_directory):
    if filename.endswith(".wav") and filename != problematic_file:
        file_path = os.path.join(input_directory, filename)
        output_path = os.path.join(output_directory, filename)
        process_file(file_path, output_path)

print("All files processed.")



import os
import soundfile as sf
import noisereduce as nr
import numpy as np
from featureExtractionConfig import *

input_directory = RAW_WAV_PATH
output_directory = NOISE_REDUCED_OUTPUT

def convert_to_mono(data):
    if len(data.shape) == 2:  # Check if the file is stereo
        return np.mean(data, axis=1)  # Averaging the two channels
    else:
        return data  # The file is already mono

def process_file(file_path, output_path):
    try:
        data, rate = sf.read(file_path)
        data = convert_to_mono(data)  # Convert to mono

        if np.any(data) and not np.all((data == 0)):
            reduced_noise = nr.reduce_noise(y=data, sr=rate,
                                            n_fft=1024, win_length=512, 
                                            hop_length=256, prop_decrease=0.7)

            sf.write(output_path, reduced_noise, rate)
        else:
            print(f"Skipping silent or near-silent file: {file_path}")
    except Exception as e:
        print(f"Error processing {file_path}: {e}")

if not os.path.exists(output_directory):
    os.makedirs(output_directory)

for filename in os.listdir(input_directory):
    if filename.endswith(".wav"):
        file_path = os.path.join(input_directory, filename)
        output_path = os.path.join(output_directory, filename)
        process_file(file_path, output_path)

print("All files processed.")
