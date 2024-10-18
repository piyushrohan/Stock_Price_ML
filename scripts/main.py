import logging
import os
import sys

import numpy as np
import pandas as pd
from tensorflow.keras.optimizers import Adam

from scripts.config import load_config  # Import config loader
from scripts.evaluation import evaluate_model

# Ensure the project directory is in sys.path so that 'scripts' is recognized
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.preprocessing import fetch_stock_data, prepare_data
from scripts.model import build_model, save_model, train_model, load_model_from_path  # Correct function names
from scripts.prediction import prepare_test_data, make_prediction
from scripts.plotting import save_prediction_plot  # Import the new plotting function


def main():
    # Load the config file
    config = load_config()

    # Ensure the logs directory exists
    if not os.path.exists('logs'):
        os.makedirs('logs')

    logging.basicConfig(filename='logs/training.log', level=logging.INFO)

    # Create a unique model file path based on the ticker
    model_file_path = f'models/{config["ticker"]}_lstm_model.h5'

    # Fetch stock data
    logging.info(f"Fetching stock data for {config['ticker']}")
    df = fetch_stock_data(config['ticker'], config['period'])
    if df is None:
        logging.error("Failed to fetch data.")
        return

    # Split data into train and validation sets
    train_data_len = int(len(df) * 0.8)
    df_train = df[:train_data_len]  # First 80% for training
    df_valid = df[train_data_len:]  # Last 20% for validation

    # Display the range of training and validation data
    print("Training data range for 'Close':", df_train['Close'].min(), "-", df_train['Close'].max())
    print("Validation data range for 'Close':", df_valid['Close'].min(), "-", df_valid['Close'].max())

    # Prepare data
    logging.info("Preparing data for training")
    x_train, y_train, scaler_close, scaler_ma50, scaled_data, training_data_len = prepare_data(df, config)

    # Build and Train the model for the specific ticker
    logging.info(f"Training model for {config['ticker']}")
    if os.path.exists(model_file_path):
        print(f"Loading model from {model_file_path}...")
        model = load_model_from_path(model_file_path)
        model.compile(optimizer=Adam(learning_rate=config['learning_rate']), loss='mean_squared_error')
    else:
        print("Building a new model...")
        model = build_model(x_train, config=config)

    train_model(model, x_train, y_train, config)

    # Save model with a unique name based on the ticker
    if config['save_model']:
        logging.info(f"Saving model to {model_file_path}")
        save_model(model, model_file_path)

    # Prepare test data and make predictions
    X_test, Y_test = prepare_test_data(df, scaled_data, training_data_len,config)

    # Predict for the validation period first
    logging.info(f"Predicting validation data for {config['ticker']}")
    predicted_prices_valid = make_prediction(model, X_test, scaler_close)

    # Adjust predicted prices if there's a mismatch in length
    if len(predicted_prices_valid) < len(df_valid):
        # Pad the predicted prices with NaN or repeat the last value
        padding = len(df_valid) - len(predicted_prices_valid)
        predicted_prices_valid = np.pad(predicted_prices_valid, (0, padding), mode='edge')  # Repeats last value

    elif len(predicted_prices_valid) > len(df_valid):
        # Trim the predicted prices if they are longer
        predicted_prices_valid = predicted_prices_valid[:len(df_valid)]

    # Assuming predicted_prices_valid is a multivariate output, select the first column
    predicted_prices_valid = predicted_prices_valid[:, 0]  # Select the first column (or whichever is relevant)

    df_predicted_valid = pd.DataFrame(predicted_prices_valid, index=df_valid.index, columns=['Predicted'])

    # Call the save_prediction_plot function to generate and save the plot
    print("Validation Data (last few rows):")
    print(df_valid.tail())

    print("Predicted Data:")
    print(df_predicted_valid)
    save_prediction_plot(df_train, df_valid, df_predicted_valid, config['ticker'])

    # Trim predicted prices and Evaluate the model using the trimmed predictions
    evaluate_model(config['ticker'], y_test=Y_test, predictions=predicted_prices_valid[:len(Y_test)])




if __name__ == "__main__":
    main()
