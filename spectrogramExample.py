import os
import random
import glob
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import spectrogram

def create_and_save_spectrogram(wav_directory, output_directory):
    # Find all WAV files in the specified directory
    wav_files = glob.glob(os.path.join(wav_directory, '*.wav'))
    
    # Pick a random WAV file
    random_wav_file = random.choice(wav_files)
    print(f"Selected file: {random_wav_file}")
    
    # Load the WAV file
    sample_rate, samples = wavfile.read(random_wav_file)
    
    # Generate the spectrogram
    frequencies, times, Sxx = spectrogram(samples, sample_rate)
    
    # Plot and save the spectrogram
    plt.pcolormesh(times, frequencies, 10 * np.log10(Sxx), shading='gouraud')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.colorbar(label='Intensity [dB]')
    
    # Create output file name
    output_file_name = os.path.basename(random_wav_file).replace('.wav', '_spectrogram.png')
    output_file_path = os.path.join(output_directory, output_file_name)
    
    # Save the figure
    plt.savefig(output_file_path)
    plt.close()  # Close the figure to free memory
    print(f"Spectrogram saved to {output_file_path}")

# Example usage
wav_directory = 'wavs'
output_directory = 'spectrograms'
create_and_save_spectrogram(wav_directory, output_directory)
