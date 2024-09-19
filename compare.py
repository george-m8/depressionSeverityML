import os
from datetime import datetime
import csv
import config
import logging
from sklearn.model_selection import train_test_split
from sklearn.linear_model import BayesianRidge, LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR 
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor

from config import *
from compile_data import compile_data
from train import train_and_evaluate_model

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

# Define models
models = {
    'Bayesian Ridge': BayesianRidge(
        max_iter=300, tol=1e-3, alpha_1=1e-6, alpha_2=1e-6, lambda_1=1e-6, lambda_2=1e-6, alpha_init=None, lambda_init=None, compute_score=False, fit_intercept=True, copy_X=True, verbose=False
    ),
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(),
    'Support Vector Regression': SVR(),
    'Decision Tree': DecisionTreeRegressor(),
    'Gradient Boosting': GradientBoostingRegressor()
    #'More models': ModelFuction(),
}

# Log each model
for model_name, model_instance in models.items():
    logging.info('%s: %s', model_name, model_instance)

# Configure logging level
log_file_base_name = os.path.basename(__file__).replace('.py', '.log')
log_file_path = os.path.join(LOG_DIR, log_file_base_name)
level = getattr(logging, config.LOGGING_LEVEL)
logging.basicConfig(filename=log_file_path,
                    filemode='a',
                    format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                    datefmt='%H:%M:%S',
                    level=level)

#feature_dirs = [LBP_DIR, ALEXNETFC7_DIR, C3D_DIR, DENSENET121_DIR, EFFICIENTNET_B3_DIR, HOG_DIR, HSVHIST_DIR, RESNET50_DIR, VGG_DIR]
feature_dirs = [MFCC1000_DIR,MFCCADV_DIR,MFCCNRADV_DIR]

for feature_dir in feature_dirs:
    compiled_data = compile_data(GROUND_TRUTH_PATH, os.path.join(FEATURE_DIR, feature_dir), VIDEO_ID, SCORE)
    logging.debug(f"Number of NaNs in compiled data ({feature_dir}): {compiled_data.isna().sum()}")
    
    train_data, test_data = train_test_split(compiled_data, test_size=0.2, random_state=42)
    print(train_data.columns)
    X_train = train_data.drop([VIDEO_ID, SCORE], axis=1)
    logging.debug(f"Number of NaNs in X_train ({feature_dir}): {X_train.isna().sum()}")

    y_train = train_data[SCORE]
    logging.debug(f"Number of NaNs in y_train ({feature_dir}): {y_train.isna().sum()}")
    # Check and print NaNs in the compiled data
    nan_rows = compiled_data[compiled_data[SCORE].isna()]
    if not nan_rows.empty:
        logging.debug(f"NaN rows in compiled data ({feature_dir}):\n{nan_rows}")


    X_test = test_data.drop([VIDEO_ID, SCORE], axis=1)
    logging.debug(f"Number of NaNs in X_test ({feature_dir}): {X_test.isna().sum()}")

    y_test = test_data[SCORE]
    logging.debug(f"Number of NaNs in y_test ({feature_dir}): {y_test.isna().sum()}")

    for model_name, model in models.items():
        mse, r2, pearson_corr = train_and_evaluate_model(X_train, y_train, X_test, y_test, model)
        # File path for the CSV to store the results
        results_file_path = os.path.join(EVALUATION_DIR, RESULTS_FILE_NAME)

        model_str = str(model_name)

        # Results to be written to the file
        results = [
            datetime.now(),
            model_str,
            feature_dir,
            mse,
            r2,
            pearson_corr
        ]

        # Check if the file exists, if not, write headers
        if not os.path.exists(results_file_path):
            with open(results_file_path, 'w', newline='') as file:
                writer = csv.writer(file)
                # Define your headers here
                headers = ['Timestamp', 'Model', 'Feature', 'MSE', 'R-Squared', 'Pearson Correlation']
                writer.writerow(headers)

        # Write results to the CSV file
        with open(results_file_path, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(results)

        logging.info(f"Results have been saved to: {results_file_path}")