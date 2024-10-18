from fastapi import FastAPI
from scripts.model import load_model
from scripts.preprocessing import fetch_stock_data, prepare_data

app = FastAPI()


@app.get("/predict")
def predict(ticker: str, period: str = "5y"):
    df = fetch_stock_data(ticker, period)
    if df is None:
        return {"error": "Data not available"}

    x_train, y_train, scaler_close, scaled_data, training_data_len = prepare_data(df)
    model = load_model("models/lstm_model.h5")

    future_price = predict_future_price(df, model, scaler_close)

    return {"ticker": ticker, "predicted_price": future_price}
