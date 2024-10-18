import matplotlib.pyplot as plt
import os
import pandas as pd
from datetime import datetime


def save_prediction_plot(df_train, df_valid, df_predicted, ticker):
    """
    Saves a stock price prediction plot for a given ticker.
    Args:
        df_train: DataFrame containing the training data
        df_valid: DataFrame containing the validation data
        df_predicted: DataFrame containing the predicted values
        ticker: Stock ticker symbol
    """
    # Create a directory for the ticker if it doesn't exist
    ticker_folder = f'plots/{ticker}'
    if not os.path.exists(ticker_folder):
        os.makedirs(ticker_folder)

    # Generate a timestamp for the filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_filename = f'{ticker_folder}/{timestamp}.png'

    # Plot the data (Train, Validation, and Predicted)
    plt.figure(figsize=(16, 8))
    plt.title(f'Stock Price Prediction Model for {ticker}')
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Close Price USD ($)', fontsize=12)

    # Plot the train and validation data
    plt.plot(df_train.index, df_train['Close'], label='Train', color='blue')
    plt.plot(df_valid.index, df_valid['Close'], label='Validation', color='orange')

    # Plot the predicted data (both validation and future predictions)
    plt.plot(df_predicted.index, df_predicted['Predicted'], label='Predicted', color='green', linestyle='--')

    # Add a vertical line and annotation to mark the prediction start point
    plt.axvline(x=df_valid.index[0], color='black', label='Prediction Start', linestyle='--')
    plt.text(df_valid.index[0], df_valid['Close'].max(), 'Prediction Start', fontsize=9, color='black')

    # Customize the legend and grid
    plt.legend(loc='lower right')
    plt.grid(True)

    # Save the plot
    plt.savefig(plot_filename)
    plt.close()

    print(f"Plot saved to: {plot_filename}")
