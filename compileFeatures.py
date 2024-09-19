import pandas as pd
import os
import re

# Adjust these paths according to your file locations
main_csv_path = 'VideoMem/training_set/train_scores.csv'  # Path to the main CSV with video IDs and scores
features_folder_path = 'VideoMem/training_set/Features/LBP'  # Path to the folder containing feature CSVs

# Read the main CSV file
main_df = pd.read_csv(main_csv_path)

# Function to extract video_id from the filename
def extract_video_id(filename):
    match = re.search(r'video(\d+)-', filename)
    print(f'Extracting ID: {match}')
    return int(match.group(1)) if match else None

# Initialize a list to store the feature data
features_data = []

# Iterate over files in the features folder
for filename in os.listdir(features_folder_path):
    video_id = extract_video_id(filename)
    if video_id is not None:
        # Read the feature file
        print(f'Getting features from {filename}')
        feature_path = os.path.join(features_folder_path, filename)
        feature_df = pd.read_csv(feature_path, header=None)

        # Assuming all feature values are in one row
        features = feature_df.iloc[0].tolist()
        
        # Combine the data
        features_data.append([video_id] + features)

# Create a DataFrame for the features
features_df = pd.DataFrame(features_data, columns=['video_id'] + [f'feature_{i+1}' for i in range(len(features_data[0]) - 1)])

# Merge the main DataFrame with the features DataFrame
print("Combining dataframes...")
combined_df = pd.merge(main_df, features_df, on='video_id')

# Save the combined DataFrame to a new CSV file
print("Saving combined...")
combined_df.to_csv('combined_data.csv', index=False)
