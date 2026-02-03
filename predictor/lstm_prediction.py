import os
import numpy as np
import pandas as pd
import yfinance as yf

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping

LOOKBACK = 60
SAVED_DIR = "predictor/saved"


def _paths(ticker: str):
    t = ticker.upper().replace("^", "")
    model_path = os.path.join(SAVED_DIR, f"{t}_lstm.keras")
    scaler_path = os.path.join(SAVED_DIR, f"{t}_scaler.csv")
    return model_path, scaler_path


def _save_scaler(scaler: MinMaxScaler, scaler_path: str):
    s = pd.DataFrame({
        "data_min": scaler.data_min_.flatten(),
        "data_max": scaler.data_max_.flatten(),
        "scale": scaler.scale_.flatten(),
        "min": scaler.min_.flatten(),
    })
    s.to_csv(scaler_path, index=False)


def _load_scaler(scaler_path: str) -> MinMaxScaler:
    s = pd.read_csv(scaler_path)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.data_min_ = s["data_min"].values
    scaler.data_max_ = s["data_max"].values
    scaler.scale_ = s["scale"].values
    scaler.min_ = s["min"].values
    scaler.n_features_in_ = 1
    return scaler


def _build_model():
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(LOOKBACK, 1)),
        LSTM(64),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse")
    return model


def _make_windows(series_scaled: np.ndarray):
    X, y = [], []
    for i in range(LOOKBACK, len(series_scaled)):
        X.append(series_scaled[i - LOOKBACK:i])
        y.append(series_scaled[i])
    X = np.array(X).reshape(-1, LOOKBACK, 1)
    y = np.array(y).reshape(-1, 1)
    return X, y


def predict_next_days(ticker: str, days: int):
    os.makedirs(SAVED_DIR, exist_ok=True)

    ticker = ticker.upper().strip()
    model_path, scaler_path = _paths(ticker)

    # downloads close prices for THIS ticker
    df = yf.download(ticker, period="5y", interval="1d")[["Close"]].dropna()
    if df is None or df.empty or len(df) < LOOKBACK + 10:
        raise ValueError("Not enough data for this ticker.")

    values = df.values

    # load or train
    if os.path.exists(model_path) and os.path.exists(scaler_path):
        model = load_model(model_path)
        scaler = _load_scaler(scaler_path)
    else:
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled = scaler.fit_transform(values)

        X, y = _make_windows(scaled)

        model = _build_model()
        es = EarlyStopping(monitor="loss", patience=3, restore_best_weights=True)
        model.fit(X, y, epochs=10, batch_size=32, verbose=0, callbacks=[es])

        model.save(model_path)
        _save_scaler(scaler, scaler_path)

    # predict forward
    scaled_all = scaler.transform(values)
    window = scaled_all[-LOOKBACK:].reshape(1, LOOKBACK, 1)

    future_scaled = []
    for _ in range(days):
        next_val = model.predict(window, verbose=0)[0][0]
        future_scaled.append([next_val])
        window = np.append(window[0, 1:, 0], next_val).reshape(1, LOOKBACK, 1)

    future = scaler.inverse_transform(np.array(future_scaled)).flatten()
    return [round(float(v), 2) for v in future]

