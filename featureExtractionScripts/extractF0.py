import os
import librosa
import numpy as np

from featureExtractionConfig import *

def extract_f0(file_path, sr=SAMPLE_RATE, frame_length=2048, hop_length=512):
    print("Extracting F0...")
    y, sr = librosa.load(file_path, sr=sr)
    f0, _, _ = librosa.pyin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'), sr=sr, frame_length=frame_length, hop_length=hop_length)
    return f0

def process_directory(input_dir, output_dir, sr=SAMPLE_RATE, frame_length=2048, hop_length=512):
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    file_count = 0    

    # Iterate over all WAV files in the input directory
    for file_name in os.listdir(input_dir):
        if file_name.endswith('.wav'):
            print(f"Opening {file_name}...")
            
            file_count += 1

            # Extract ID from the file name
            file_id = file_name.split('-')[0]
            print(f"ID: {file_id}")

            # Construct full file path
            file_path = os.path.join(input_dir, file_name)

            # Extract F0
            f0 = extract_f0(file_path, sr, frame_length, hop_length)

            # Replace NaN values with zeros
            f0_cleaned = np.nan_to_num(f0)

            # Save the F0 array as a .npy file
            output_file_path = os.path.join(output_dir, f"{file_id}.npy")
            np.save(output_file_path, f0_cleaned)

            print(f"Processed {file_name} to: {output_file_path}")
            print(f"Processed {file_count} files...")
    print("F0 Extraction complete.")

# Set your input and output directories here
input_directory = RAW_WAV_PATH
output_directory = F0_OUTPUT

process_directory(input_directory, output_directory)