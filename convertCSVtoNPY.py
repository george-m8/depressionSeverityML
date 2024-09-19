import os
import numpy as np
import pandas as pd
from config import *

def process_csv_to_npy(main_directory, subdirectories):
    for subdirectory in subdirectories:
        subdirectory_path = os.path.join(main_directory, subdirectory)

        # Check if the subdirectory exists
        if not os.path.isdir(subdirectory_path):
            print(f"Subdirectory {subdirectory} does not exist.")
            continue

        # List of CSV files in the subdirectory
        csv_files = [f for f in os.listdir(subdirectory_path) if f.endswith('.csv')]

        # Proceed if there are CSV files
        if csv_files:
            npy_directory = f"{subdirectory_path}_npy"
            os.makedirs(npy_directory, exist_ok=True)

            for csv_file in csv_files:
                csv_path = os.path.join(subdirectory_path, csv_file)
                data = pd.read_csv(csv_path)

                npy_path = os.path.join(npy_directory, os.path.splitext(csv_file)[0] + '.npy')
                np.save(npy_path, data.to_numpy())

                print(f"Saved {npy_path}")

# Example usage
main_directory = FEATURE_DIR
subdirectories = [LBP_DIR, ALEXNETFC7_DIR, C3D_DIR, HOG_DIR, HSVHIST_DIR, VGG_DIR]  # Replace with your actual subdirectories
process_csv_to_npy(main_directory, subdirectories)


