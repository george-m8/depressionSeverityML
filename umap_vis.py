import pandas as pd
import numpy as np
import umap
import matplotlib.pyplot as plt
import seaborn as sns
import os
import re
import logging
from datetime import datetime
import config

from config import *
from compile_data import compile_data  

# Configuration for logging
# Get the current date and time
current_time = datetime.now()
timestamp = current_time.strftime("%Y%m%d_%H%M%S")

# Configure logging level and file path
log_file_base_name = f"{os.path.basename(__file__).replace('.py', '')}_{timestamp}.log"
log_file_path = os.path.join(LOG_DIR, log_file_base_name)
level = getattr(logging, config.LOGGING_LEVEL)

# Create a logger
logger = logging.getLogger()
logger.setLevel(level)

# Create handlers
file_handler = logging.FileHandler(log_file_path, mode='a')
console_handler = logging.StreamHandler()

# Create formatters and add it to handlers
formatter = logging.Formatter('%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s', datefmt='%H:%M:%S')
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

# Add handlers to the logger
logger.addHandler(file_handler)
logger.addHandler(console_handler)

logging.info("Starting script...")

# UMAP Dimensionality Reduction and Visualization

# Example usage
# def umap_reduction_and_visualization(features, targets, plot_dimensions=3, umap_params={'n_neighbors': 15, 'metric': 'cosine'})

def umap_reduction_and_visualization(features, targets, plot_dimensions=2, umap_params=None):
    # Default UMAP parameters
    default_params = {
        'n_neighbors': 30,
        'min_dist': 0.1,
        'n_components': plot_dimensions,  # Set based on plot dimensions
        'metric': 'euclidean',
        'n_epochs': 500,
        'learning_rate': 1.0,
        'spread': 1.0,
        'random_state': 42,
        'local_connectivity': 1,
        'repulsion_strength': 1
    }

    # Update default parameters with any user-provided parameters
    if umap_params:
        default_params.update(umap_params)

    # Initialize UMAP with the updated parameters
    reducer = umap.UMAP(**default_params)

    # Perform UMAP reduction
    embedding = reducer.fit_transform(features)

    # Plotting
    plt.figure(figsize=(12, 10))
    if plot_dimensions == 3:
        ax = plt.subplot(111, projection='3d')
        ax.scatter(embedding[:, 0], embedding[:, 1], embedding[:, 2], c=targets, cmap='Spectral')
        ax.set_xlabel('UMAP Dimension 1')
        ax.set_ylabel('UMAP Dimension 2')
        ax.set_zlabel('UMAP Dimension 3')
        plt.title('3D UMAP Projection of Features')
    else:  # Default to 2D if not 3D
        sns.scatterplot(x=embedding[:, 0], y=embedding[:, 1], hue=targets, palette="Spectral", legend='full')
        plt.xlabel('UMAP Dimension 1')
        plt.ylabel('UMAP Dimension 2')
        plt.title('2D UMAP Projection of Features')

    plt.show()


#feature_dirs = [RESNET50_DIR]
feature_dirs = [HFD_DIR]

#ground_truth_path = GROUND_TRUTH_PATH
#feature_dir = FEATURE_DIR
#id_column = VIDEO_ID
#score_column = SCORE

ground_truth_path = GROUND_TRUTH_PATH
feature_dir = FEATURE_DIR
id_column = VIDEO_ID
score_column = SCORE

for feature_dir in feature_dirs:
    compiled_data = compile_data(GROUND_TRUTH_PATH, os.path.join(FEATURE_DIR, feature_dir), VIDEO_ID, SCORE)
    logging.debug(f"Number of NaNs in compiled data ({feature_dir}): {compiled_data.isna().sum()}")

    #data_df = compile_data(ground_truth_path, feature_dir, id_column, score_column)
    data_df = compiled_data

    # Separate features and target values
    if SCORE in data_df.columns:
        features = data_df.drop(columns=[score_column])
    else:
        features = data_df

    targets = data_df[score_column]

    # Apply UMAP and Visualize
    umap_reduction_and_visualization(features, targets)

'''# Main Execution
if __name__ == "__main__":
    # Define paths and columns
    ground_truth_path = 'path/to/ground_truth.csv'  # Update this path
    feature_dir = 'path/to/feature_directory'      # Update this path
    id_column = 'ID_COLUMN_NAME'                   # Update the ID column name
    score_column = 'SCORE_COLUMN_NAME'             # Update the score column name

    # Compile Data
    data_df = compile_data(ground_truth_path, feature_dir, id_column, score_column)

    # Separate features and target values
    features = data_df.drop(columns=[id_column, score_column])
    targets = data_df[score_column]

    # Apply UMAP and Visualize
    umap_reduction_and_visualization(features, targets)
'''
