import os
import sys

# Add the higher-level directory to sys.path
higher_level_directory = os.path.abspath("../")  # Adjust the path as necessary
sys.path.append(higher_level_directory)

from config import MAIN_PROJECT_DIR,FEATURE_DIR

SAMPLE_RATE = 48000 # in hz

RAW_WAV_DIR = "wavs"
RAW_WAV_PATH = os.path.join(MAIN_PROJECT_DIR, RAW_WAV_DIR)

START_END_TRIM_OUTPUT = "wavs_startEndTrim"
NOISE_REDUCED_OUTPUT = "wavs_noiseReduced"
F0_OUTPUT = "f0"
RESNET50_FEATURE_OUTPUT = "resnet50"
RESNET50_BLACKMAN_OUTPUT = "resenet50_blackman"
HFD_FEATURE_OUTPUT = "hfd"
EXTRACT_VOWELS_PATH = "extractVowels"

SAVE_TO_DIR = FEATURE_DIR