import os
import soundfile as sf
import numpy as np
import librosa
from pydub import AudioSegment, silence
from featureExtractionConfig import *

input_directory = RAW_WAV_PATH
output_directory = START_END_TRIM_OUTPUT
sample_rate = SAMPLE_RATE

def convert_to_mono(data):
    if len(data.shape) == 2:
        return np.mean(data, axis=1)
    else:
        return data

def resample_audio(data, original_rate, target_rate):
    if original_rate != target_rate:
        return librosa.resample(data, orig_sr=original_rate, target_sr=target_rate)
    return data

def process_file(file_path, output_path):
    try:
        data, rate = sf.read(file_path, dtype='int16')
        data = convert_to_mono(data)
        data = resample_audio(data, rate, sample_rate)

        sample_width = 2  # Since we're reading as 16-bit
        audio_segment = AudioSegment(
            data.tobytes(), 
            frame_rate=sample_rate, 
            sample_width=sample_width, 
            channels=1
        )

        # Detect and trim silence
        trimmed_audio = silence.detect_nonsilent(
            audio_segment, 
            min_silence_len=1000, 
            silence_thresh=-50
        )
        if trimmed_audio:
            start_trim = trimmed_audio[0][0]
            end_trim = trimmed_audio[0][1]
            print(f"Start trim: {start_trim}")
            print(f"End trim: {end_trim}")
            trimmed_audio_segment = audio_segment[start_trim:end_trim]

            # Save the trimmed audio
            trimmed_audio_segment.export(output_path, format="wav", parameters=["-ar", str(sample_rate)])
            print(f"Processed and saved at 48kHz: {output_path}")
        else:
            print(f"No non-silent sections found in {file_path}")
            audio_segment.export(output_path, format="wav", parameters=["-ar", str(sample_rate)])
            print(f"Saved at 48kHz: {output_path}")
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