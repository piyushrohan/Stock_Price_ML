
# Stock Price Prediction Using LSTM

## Overview

This project implements a Long Short-Term Memory (LSTM) neural network model for stock price prediction. The model predicts future stock prices based on historical stock data, using features such as Moving Averages and Volume. It incorporates advanced techniques like Dropout layers, learning rate schedulers, and EarlyStopping to enhance model performance. The model is flexible, allowing for configuration changes via a JSON file, making it easy to experiment with different model parameters.

## Table of Contents

- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Configuration File](#configuration-file)
- [Key Features](#key-features)
- [Technologies and Libraries Used](#technologies-and-libraries-used)
- [Model Explanation](#model-explanation)
- [Metrics and Reasoning](#metrics-and-reasoning)
- [How to Run the Project](#how-to-run-the-project)
- [Future Improvements](#future-improvements)

---

## Project Structure

```
stock_prediction_project/
├── config/
│   └── config.json          # Configuration file
├── data/
│   └── <fetched_data.csv>   # Stock data files stored here
├── logs/
│   └── training.log         # Training logs
├── models/
│   └── <ticker_lstm_model.h5>  # Saved models for each ticker
├── plots/
│   └── <ticker>/<timestamp.png>  # Prediction plots saved with timestamp
├── scripts/
│   ├── config.py            # Handles loading of configuration file
│   ├── data_preprocessing.py # Preprocesses stock data (e.g., scaling, feature generation)
│   ├── model_training.py    # Trains the LSTM model
│   ├── prediction.py        # Handles predictions using the trained model
│   ├── plotting.py          # Plots the predicted and actual stock prices
│   └── main.py              # Main entry point of the project
└── README.md                # Project documentation
```

---

## Getting Started

To get started with this project:

1. **Clone the repository**:
   ```bash
   git clone <repository_url>
   cd stock_prediction_project
   ```

2. **Install the required dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure the `config.json` file** with your desired model parameters (more on this below).

4. **Run the script** to predict stock prices:
   ```bash
   python scripts/main.py --ticker AAPL --predict_days 5 --save_model
   ```

---

## Configuration File

The project uses a JSON configuration file (`config/config.json`) to define parameters for the LSTM model and preprocessing steps. Here’s a sample configuration:

```json
{
  "window_size": 60,
  "epochs": 100,
  "batch_size": 32,
  "learning_rate": 0.001,
  "units": 50,
  "dropout_rate": 0.2,
  "period": "5y"
}
```

### Configuration Parameters:

- **`window_size`**: The number of previous days of data used to predict future stock prices.
- **`epochs`**: The number of times the model trains on the entire dataset.
- **`batch_size`**: Number of training samples to propagate through the network before updating weights.
- **`learning_rate`**: Controls how much to change the model's weights during optimization.
- **`units`**: Number of LSTM units (memory cells) in each LSTM layer.
- **`dropout_rate`**: Regularization technique to prevent overfitting by randomly dropping neurons during training.
- **`period`**: Specifies the historical period to fetch the stock data (e.g., 1y, 5y).

---

## Key Features

- **Flexible Model Configuration**: Change model parameters like window size, batch size, learning rate, and more through a configuration file.
- **Automatic Data Fetching**: The script fetches the latest stock data using `yfinance` based on the provided stock ticker and period.
- **Model Saving**: The trained models are saved in a `models/` directory with unique names for each stock ticker.
- **Prediction Plots**: The predictions are saved as plots in the `plots/` directory, allowing for easy visualization of the model's performance.

---

## Technologies and Libraries Used

- **Python 3.12**: The core programming language used.
- **TensorFlow/Keras**: For building and training the LSTM model.
- **scikit-learn**: Used for data preprocessing, scaling features, and shuffling.
- **yfinance**: A library for fetching historical stock price data.
- **matplotlib**: Used for visualizing the actual vs predicted stock prices.

---

## Model Explanation

### Why LSTM?

LSTM (Long Short-Term Memory) networks are a type of Recurrent Neural Network (RNN) specifically designed to handle sequential data, such as time-series data. The key advantage of LSTMs is their ability to remember information over long sequences, which is crucial for predicting stock prices based on historical data.

### Model Architecture

1. **Input Layer**: The input consists of historical stock prices and technical indicators (MA50, MA100, and Volume) over a sliding window (defined by `window_size`).
2. **LSTM Layers**: 
   - First LSTM layer with return sequences enabled to capture the temporal dependencies.
   - Second LSTM layer without return sequences, followed by Dropout layers to prevent overfitting.
3. **Dense Layer**: A fully connected layer that outputs the predicted stock price.
4. **Output Layer**: A single neuron outputting the predicted stock price.

---

## Metrics and Reasoning

### 1. **Mean Squared Error (MSE)**:
   - **Definition**: Measures the average squared difference between actual and predicted values.
   - **Why Use It?**: MSE is useful for regression problems like stock price prediction as it penalizes larger errors more than smaller errors, thus pushing the model to make more accurate predictions.

### 2. **Dropout**:
   - **Definition**: A regularization technique that randomly "drops" neurons during training to prevent the model from overfitting.
   - **Why Use It?**: Overfitting is a common problem in time-series prediction, and using dropout helps the model generalize better to unseen data.

### 3. **Learning Rate**:
   - **Definition**: The rate at which the model updates its weights.
   - **Why Use It?**: A small learning rate (e.g., 0.001) ensures that the model converges smoothly to a solution without overshooting, but it may need more epochs to train effectively.

### 4. **Early Stopping**:
   - **Definition**: Stops training when the model stops improving.
   - **Why Use It?**: Prevents the model from wasting time on further training when it has reached its optimal performance, thereby preventing overfitting.

---

## How to Run the Project

1. **Set your configuration**:
   - Modify the parameters in `config/config.json` to your liking (e.g., window size, learning rate, epochs).

2. **Run the main script**:
   - Use the following command to train a model for a particular stock ticker and predict future prices:
   ```bash
   python scripts/main.py --ticker AAPL --predict_days 5 --save_model
   ```

3. **View Prediction Plots**:
   - After running the script, a prediction plot will be saved in the `plots/<ticker>/` directory.
   - The plot compares the actual stock prices with the predicted ones.

---

## Future Improvements

1. **Hyperparameter Tuning**:
   - Implement grid search or random search to fine-tune the model's hyperparameters such as learning rate, dropout rate, and the number of LSTM units.

2. **Cross-Validation**:
   - Implement k-fold cross-validation to ensure the model generalizes well to unseen data and isn’t overfitting on a specific set of stock prices.

3. **Additional Features**:
   - Incorporate additional technical indicators (e.g., RSI, Bollinger Bands) to improve prediction accuracy.

4. **More Advanced Models**:
   - Experiment with other advanced time-series models like GRU (Gated Recurrent Units) or even transformer-based architectures.

---

