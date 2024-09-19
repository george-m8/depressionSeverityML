import pandas as pd
import numpy as np
import os
import re

from config import *
import logging

def compile_data(ground_truth_path, feature_dir, id_column, score_column):
    logging.info(f"Compiling data... Current feature directory path: {feature_dir}")

    cache_file_name = f"{os.path.basename(feature_dir)}_cached.csv"
    cache_file_path = os.path.join(CACHE_DIR, cache_file_name)

    if USE_CACHE and os.path.exists(cache_file_path) and not OVERWRITE_CACHE:
        logging.info(f"Cached data exists. Loading cached data from: {cache_file_path}")
        return pd.read_csv(cache_file_path)
    
    else: logging.info("No cached data available. Compiling data.")

    ground_truth_df = pd.read_csv(ground_truth_path)[[id_column, score_column]]
    #ground_truth_df[id_column] = ground_truth_df[id_column].astype(str).apply(lambda x: x.zfill(5))
    logging.debug(f"Number of NaNs in ground_truth_df: {ground_truth_df.isna().sum()}")
    
    file_count = 0
    feature_data = []
    feature_length = None
    max_feature_length = 0  # Track the maximum feature length

    
    for file in os.listdir(feature_dir):
        logging.debug(f"Processing feature file: {file}")
        file_path = os.path.join(feature_dir, file)
        file_extension = file.split('.')[-1]
        
        if file_extension in ['csv', 'npy']:
            video_id = file.split('-', 1)[0]
            video_id = video_id.rsplit('.', 1)[0]

            logging.debug(f"Video ID: {video_id}")

            features = pd.read_csv(file_path, header=None).values.flatten() if file_extension == 'csv' else np.load(file_path).flatten()
            logging.debug(f"Feature length for file {file}: {len(features)}")

            # Update max_feature_length if current features length is greater
            if len(features) > max_feature_length:
                max_feature_length = len(features)
                logging.debug(f"Updated max_feature_length: {max_feature_length}")
            
            feature_data.append([video_id] + list(features))

            
            file_count += 1
            logging.debug(f"Read {file_count} files...")
    
    logging.debug(f"Final max_feature_length: {max_feature_length}")
   
    # Pad shorter rows with zeros to match max_feature_length
    for i in range(len(feature_data)):
        feature_length = len(feature_data[i]) - 1  # Subtract 1 for video_id
        if feature_length < max_feature_length:
            feature_data[i].extend([0] * (max_feature_length - feature_length))
            
    # Ensure that the DataFrame is created only if there's consistent feature data
    if feature_length is not None:
            feature_df = pd.DataFrame(feature_data, columns=[VIDEO_ID] + [f'feature_{i}' for i in range(max_feature_length)])

            # Merge feature DataFrame with ground truth DataFrame
            final_df = pd.merge(feature_df, ground_truth_df, left_on=VIDEO_ID, right_on=id_column, how='left')

            # Reorder columns so that ID and Score are the first two columns
            column_order = [id_column, score_column] + [col for col in final_df.columns if col not in [id_column, score_column]]
            final_df = final_df[column_order]
    else:
        logging.error("No valid feature files processed.")
        return None

    # Check for any NaN values in the score column after the merge
    if final_df[score_column].isna().any():
        logging.warning("There are missing scores in the merged data.")
    
    logging.debug(f"Dropping ID columns: {VIDEO_ID}")
    final_df.drop(columns=[VIDEO_ID], inplace=True)
    
    logging.debug(f"Number of NaNs in compiled data: {final_df.isna().sum()}")

    # Calculate total number of rows
    total_rows = len(final_df)

    # Calculate the number of rows with at least one NaN
    rows_with_nan = len(final_df[final_df.isna().any(axis=1)])

    # Check if less than 10% of rows have NaNs
    if rows_with_nan / total_rows < 0.10:
        print(f"Rows with NaN values: {rows_with_nan}")
        print("Total rows with NaN values is less than 10%. Dropping rows...")
        # Remove rows with NaNs
        final_df = final_df.dropna()

        # Optional: Reset index if needed
        # final_df = final_df.reset_index(drop=True)

    logging.info("Compile complete.")
    if USE_CACHE:
        # Extract the directory path from the cache file path
        cache_dir = os.path.dirname(cache_file_path)

        # Check if the directory exists, and create it if it doesn't
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir, exist_ok=True)
            logging.info(f"Created cache directory: {cache_dir}")
        logging.info(f"Saving compiled data to cache: {cache_file_path}")
        final_df.to_csv(cache_file_path, index=False)

    return final_df


# Working code below, can't handle features with different lengths:
'''
import pandas as pd
import numpy as np
import os
import re

from config import *
import logging

def compile_data(ground_truth_path, feature_dir, id_column, score_column):
    logging.info(f"Compiling data... Current feature directory path: {feature_dir}")

    cache_file_name = f"{os.path.basename(feature_dir)}_cached.csv"
    cache_file_path = os.path.join(CACHE_DIR, cache_file_name)

    if USE_CACHE and os.path.exists(cache_file_path) and not OVERWRITE_CACHE:
        logging.info(f"Cached data exists. Loading cached data from: {cache_file_path}")
        return pd.read_csv(cache_file_path)
    
    else: logging.info("No cached data available. Compiling data.")

    ground_truth_df = pd.read_csv(ground_truth_path)[[id_column, score_column]]
    #ground_truth_df[id_column] = ground_truth_df[id_column].astype(str).apply(lambda x: x.zfill(5))
    logging.debug(f"Number of NaNs in ground_truth_df: {ground_truth_df.isna().sum()}")
    
    file_count = 0
    feature_data = []
    feature_length = None
    
    for file in os.listdir(feature_dir):
        logging.debug(f"Processing feature file: {file}")
        file_path = os.path.join(feature_dir, file)
        file_extension = file.split('.')[-1]
        
        if file_extension in ['csv', 'npy']:
            video_id = file.rsplit('.', 1)[0]

            logging.debug(f"Video ID: {video_id}")

            features = pd.read_csv(file_path, header=None).values.flatten() if file_extension == 'csv' else np.load(file_path).flatten()

            # Check and set the feature length for the first file
            if feature_length is None:
                feature_length = len(features)
            # If current file's feature length is different, log an error
            elif len(features) != feature_length:
                logging.error(f"Feature length mismatch in file {file}. Expected {feature_length}, got {len(features)}")
                continue  # Skip this file
            
            feature_data.append([video_id] + list(features))

            
            file_count += 1
            logging.debug(f"Read {file_count} files...")
            
    # Ensure that the DataFrame is created only if there's consistent feature data
    if feature_length is not None:
        feature_df = pd.DataFrame(feature_data, columns=["video_id"] + [f'feature_{i}' for i in range(feature_length)])
        # ... [rest of your DataFrame processing code] ...
    else:
        logging.error("No valid feature files processed.")
        return None

    feature_df = pd.DataFrame(feature_data, columns=[VIDEO_ID] + [f'feature_{i}' for i in range(len(features))])
    
    # Merge feature DataFrame with ground truth DataFrame
    final_df = pd.merge(feature_df, ground_truth_df, left_on=VIDEO_ID, right_on=id_column, how='left')

    # Reorder columns so that ID and Score are the first two columns
    column_order = [id_column, score_column] + [col for col in final_df.columns if col not in [id_column, score_column]]
    final_df = final_df[column_order]

    # Drop duplicate ID column after merge
    #final_df.drop(columns=[id_column], inplace=True)

    # Check for any NaN values in the score column after the merge
    if final_df[score_column].isna().any():
        logging.warning("There are missing scores in the merged data.")
    
    logging.debug(f"Dropping ID columns: {VIDEO_ID}")
    final_df.drop(columns=[VIDEO_ID], inplace=True)
    
    logging.debug(f"Number of NaNs in compiled data: {final_df.isna().sum()}")

    logging.info("Compile complete.")
    if USE_CACHE:
        # Extract the directory path from the cache file path
        cache_dir = os.path.dirname(cache_file_path)

        # Check if the directory exists, and create it if it doesn't
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir, exist_ok=True)
            logging.info(f"Created cache directory: {cache_dir}")
        logging.info(f"Saving compiled data to cache: {cache_file_path}")
        final_df.to_csv(cache_file_path, index=False)

    return final_df
'''