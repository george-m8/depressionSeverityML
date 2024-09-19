import os
from pydub import AudioSegment
from pocketsphinx import Pocketsphinx, get_model_path
from featureExtractionConfig import *

model_path = get_model_path()

def convert_to_48k_16bit(file_path):
    """Converts audio to 48kHz 16-bit format if not already in that format."""
    audio = AudioSegment.from_wav(file_path)
    if audio.frame_rate != 48000 or audio.sample_width != 2:
        audio = audio.set_frame_rate(48000).set_sample_width(2)
        audio.export(file_path, format='wav')  # Overwrite the file with the new format

def get_vowel_phonemes(file_path):
    """Extracts vowel phonemes from an audio file using PocketSphinx."""
    config = {
        'hmm': os.path.join(model_path, 'en-us'),
        'lm': False,
        'dict': os.path.join(model_path, 'cmudict-en-us.dict'),
        'allphone': os.path.join(model_path, 'en-us-phone.lm.bin'),
        'lw': 2.0,
        'pip': 0.3,
        'beam': 1e-20,
        'pbeam': 1e-20
    }

    ps = Pocketsphinx(**config)
    ps.decode(
        audio_file=file_path,
        buffer_size=2048,
        no_search=False,
        full_utt=False
    )

    vowel_phonemes = ["AA", "AE", "AH", "AO", "AW", "AY", "EH", "ER", "EY", "IH", "IY", "OW", "OY", "UH", "UW"]
    extracted_vowels = []

    for segment in ps.segments():
        phoneme = segment[2]
        if phoneme in vowel_phonemes:
            start_time = segment[0] / 1000.0  # Convert to seconds
            end_time = segment[1] / 1000.0    # Convert to seconds
            extracted_vowels.append((phoneme, start_time, end_time))

    return extracted_vowels

for file_name in os.listdir(RAW_WAV_PATH):
    if file_name.endswith('.wav'):
        file_path = os.path.join(RAW_WAV_PATH, file_name)
        convert_to_48k_16bit(file_path)
        vowels = get_vowel_phonemes(file_path)
        print(f"Vowel phonemes in {file_name}: {vowels}")
