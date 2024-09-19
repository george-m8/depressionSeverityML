import os
import librosa
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image
from skimage.transform import resize

from featureExtractionConfig import *

def audio_to_spectrogram(file_path, sr=SAMPLE_RATE):
    print("Converting audio to spectrogram...")
    y, sr = librosa.load(file_path, sr=sr)
    spectrogram = librosa.feature.melspectrogram(y=y, sr=sr)
    spectrogram = librosa.power_to_db(spectrogram, ref=np.max)

    # Convert to PIL Image format
    spectrogram = resize(spectrogram, (224, 224), mode='constant')
    spectrogram = np.stack((spectrogram,) * 3, -1)  # Convert to 3 channels
    spectrogram = image.img_to_array(spectrogram)
    spectrogram = np.expand_dims(spectrogram, axis=0)
    spectrogram = preprocess_input(spectrogram)

    return spectrogram

def extract_resnet50_features(spectrogram):
    print("Extracting features using ResNet50...")
    model = ResNet50(weights='imagenet', include_top=False)
    features = model.predict(spectrogram)
    return features

def process_directory(input_dir, output_dir):
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    file_count = 0

    for file_name in os.listdir(input_dir):
        if file_name.endswith('.wav'):
            print(f"Opening {file_name}...")
            
            file_count += 1
            file_id = file_name.split('-')[0]
            print(f"ID: {file_id}")

            file_path = os.path.join(input_dir, file_name)
            
            spectrogram = audio_to_spectrogram(file_path)
            features = extract_resnet50_features(spectrogram)

            # Save the features as a .npy file
            output_file_path = os.path.join(output_dir, f"{file_id}.npy")
            np.save(output_file_path, features)

            print(f"Processed {file_name} to: {output_file_path}")
            print(f"Processed {file_count} files...")
    print("Feature extraction complete.")

input_directory = RAW_WAV_PATH
output_directory = RESNET50_FEATURE_OUTPUT

process_directory(input_directory, output_directory)
