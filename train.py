import logging
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import pearsonr

def train_and_evaluate_model(X_train, y_train, X_test, y_test, model):
    logging.info(f'Training model: {model}')
    logging.info(f'Model parameters: {model.get_params()}')
    model.fit(X_train, y_train)
    test_predictions = model.predict(X_test)

    # Calculate evaluation metrics
    mse = mean_squared_error(y_test, test_predictions)
    r2 = r2_score(y_test, test_predictions)
    pearson_corr, _ = pearsonr(y_test, test_predictions)

    return mse, r2, pearson_corr