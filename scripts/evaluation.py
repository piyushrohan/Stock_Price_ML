import os
from datetime import datetime

from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
import logging
import pandas as pd


# Evaluate the model performance
def evaluate_model(ticker, y_test, predictions):
    # Create a directory for the ticker if it doesn't exist
    ticker_folder = f'predictions/{ticker}'
    if not os.path.exists(ticker_folder):
        os.makedirs(ticker_folder)
    # Generate a timestamp for the filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    predictions_filename = f'{ticker_folder}/{timestamp}.csv'

    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    mae = mean_absolute_error(y_test, predictions)
    mape = np.mean(np.abs((y_test - predictions) / y_test)) * 100

    logging.info(f"Model Evaluation Metrics:")
    logging.info(f"RMSE: {rmse:.4f}")
    logging.info(f"MAE: {mae:.4f}")
    logging.info(f"MAPE: {mape:.2f}%")

    # Save predictions to CSV
    results = pd.DataFrame({'Actual': y_test, 'Predicted': predictions.flatten()})
    results.to_csv(predictions_filename, index=False)
    logging.info(f"Predictions saved to '{predictions_filename}'")
