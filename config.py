# Logging config
LOGGING_LEVEL = "DEBUG" #"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL" - Listed in hierarchical order. Logging prints level and higher.
LOG_DIR = "logs"

# File Locations
MAIN_PROJECT_DIR = "/Users/george/DepressionSeverity/"
GROUND_TRUTH_PATH = "/Users/george/DepressionSeverity/lookup.csv"

# Ground Truth columns
VIDEO_ID = "ID"
SCORE = "PHQ"
TARGET = SCORE

# Features main directory
FEATURE_DIR = "/Users/george/DepressionSeverity/features"

# Feature directory names
#LBP_DIR = "LBP" # Example
F0_DIR = "f0"
RESNET50_DIR = "resnet50"
RESNET50_B_DIR = "resnet50_blackman"
HFD_DIR = "hfd"
#Below are already compliled, using just for cache.
MFCC1000_DIR = "mfcc1000"
MFCCADV_DIR = "mfccadv"
MFCCNRADV_DIR = "mfccNRadv"

# Feature dimensions
#LBP_DIMS = 13275 # Unused

# Data directory
EVALUATION_DIR = 'evaluation'

# Results file name
RESULTS_FILE_NAME = 'evaluation_results.csv'

# complile_data module config
CACHE_DIR = 'compiledFeaturesCache'
USE_CACHE = True
OVERWRITE_CACHE = False

