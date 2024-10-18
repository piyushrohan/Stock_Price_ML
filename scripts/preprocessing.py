import logging

import numpy as np
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler


def fetch_stock_data(ticker, period='5y'):
    """
    Fetch historical stock data for a given ticker symbol.
    Args:
        ticker: The stock ticker symbol
        period: The time period to fetch data (default is '5y')
    Returns:
        A DataFrame with stock data or None if an error occurs
    """
    logging.info(f"Fetching data for {ticker} over period {period}")
    try:
        stock = yf.Ticker(ticker)
        data = stock.history(period=period)[['Close', 'Volume']]
        if data.empty:
            raise ValueError(f"No data found for {ticker}")
        return data
    except Exception as e:
        logging.error(f"Error fetching data: {e}")
        return None


def prepare_data(df, config):
    """
    Prepare the stock data for training by generating features like moving averages and scaling the data.
    Args:
        df: The stock data DataFrame
        config: The configuration dictionary containing settings like window size
    Returns:
        x_train: Training input data
        y_train: Training target data (close prices)
        scaler_close: Scaler for the 'Close' price
        scaler_ma50: Scaler for the 'MA50' moving average
        scaler_ma100: Scaler for the 'MA100' moving average
        scaler_volume: Scaler for the 'Volume'
        scaled_data: Scaled complete dataset
        training_data_len: Length of training data
    """
    df['MA50'] = df['Close'].rolling(window=50).mean()
    df.dropna(inplace=True)

    # Create separate scalers for each feature
    scaler_close = MinMaxScaler(feature_range=(0, 1))
    scaler_ma50 = MinMaxScaler(feature_range=(0, 1))

    # Transform the entire dataset using the fitted scalers
    scaled_close = scaler_close.fit_transform(df[['Close']])
    scaled_ma50 = scaler_ma50.fit_transform(df[['MA50']])

    # Combine the scaled features into one dataset
    scaled_data = np.hstack((scaled_close, scaled_ma50))

    # Create the training data
    training_data_len = int(np.ceil(len(scaled_data) * 0.8))  # 80% for training
    train_data = scaled_data[0:training_data_len, :]

    window_size = config['window_size']  # Get window size from config

    x_train, y_train = [], []
    for i in range(window_size, len(train_data)):
        x_train.append(train_data[i - window_size:i])
        y_train.append(train_data[i, 0])  # Only predicting 'Close' price (index 0)

    x_train, y_train = np.array(x_train), np.array(y_train)

    # Reshape the data to be 3D [samples, time steps, features]
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], x_train.shape[2]))

    # Return scalers and other necessary components
    return x_train, y_train, scaler_close, scaler_ma50, scaled_data, training_data_len
