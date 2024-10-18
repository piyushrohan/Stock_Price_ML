import numpy as np


def prepare_test_data(df, scaled_data, training_data_len, config):
    test_data = scaled_data[training_data_len - config['window_size']:, :]
    x_test, y_test = [], df['Close'].values[training_data_len:]

    for i in range(config['window_size'], len(test_data)):
        x_test.append(test_data[i - config['window_size']:i])

    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], x_test.shape[2]))
    return x_test, y_test


# Make predictions using the model
def make_prediction(model, x_test, scaler_close):
    predictions = model.predict(x_test)
    predictions = scaler_close.inverse_transform(predictions)  # Only inverse transform 'Close' price
    return predictions
