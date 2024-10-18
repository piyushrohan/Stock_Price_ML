from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.optimizers import Adam


def build_model(x_train, config=None):
    """
    Trains an LSTM model based on the configuration provided.
    Args:
        x_train: Training features
        y_train: Training target values
        x_valid: Validation features
        y_valid: Validation target values
        config: Configuration parameters such as epochs, batch size, etc.
    Returns:
        model: The trained LSTM model
    """
    model = Sequential()

    # Input layer and first LSTM with increased units
    model.add(Input(shape=(x_train.shape[1], x_train.shape[2])))
    model.add(LSTM(units=config['units'] * 4, return_sequences=True))
    model.add(Dropout(config['dropout_rate']))

    # Second LSTM layer with slightly fewer units, as it's common to decrease the size in deeper layers
    model.add(LSTM(units=config['units'] * 2, return_sequences=False))
    model.add(Dropout(config['dropout_rate']))

    # Dense layers to adjust for complexity and non-linearity
    model.add(Dense(units=config['units'], activation='relu'))
    model.add(Dense(units=1))  # Output layer with 1 unit for price prediction

    # Use Adam optimizer with default learning rate from config
    model.compile(optimizer=Adam(learning_rate=config['learning_rate']), loss='mean_squared_error')

    return model


def train_model(model, x_train, y_train, config=None):
    # Fit the model with or without validation data, based on the provided data
    model.fit(
        x_train, y_train,
        batch_size=config['batch_size'], epochs=config['epochs'])


def save_model(model, model_path):
    """
    Saves the trained model to the specified file path.
    Args:
        model: Trained Keras model
        model_path: Path where the model should be saved
    """
    model.save(model_path)
    print(f"Model saved at {model_path}")


def load_model_from_path(model_path):
    """
    Loads a trained model from the specified file path.
    Args:
        model_path: Path to the saved model file
    Returns:
        The loaded Keras model
    """
    try:
        model = load_model(model_path)
        print(f"Model loaded from {model_path}")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None
