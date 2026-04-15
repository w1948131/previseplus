import os
import numpy as np
import pandas as pd
import yfinance as yf

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping

LOOKBACK = 60  # past trading days used to predict next value
SAVED_DIR = "predictor/saved"  # trained models and scalers are stored in this path
FEATURES = ["Close", "MA20", "MA50", "RSI", "Volume"]  # 5 input features for greater model accuracy


# generates file path for the trained LSTM model & scaler
def _paths(ticker: str):
    t = ticker.upper().replace("^", "")
    model_path  = os.path.join(SAVED_DIR, f"{t}_lstm.keras")
    scaler_path = os.path.join(SAVED_DIR, f"{t}_scaler.csv")
    return model_path, scaler_path


# Persists minmax scaler parameters for all features
def _save_scaler(scaler: MinMaxScaler, scaler_path: str):
    s = pd.DataFrame({
        "data_min": scaler.data_min_.flatten(),
        "data_max": scaler.data_max_.flatten(),
        "scale":    scaler.scale_.flatten(),
        "min":      scaler.min_.flatten(),
    })
    s.to_csv(scaler_path, index=False)


# Re-loads saved MinMaxScaler
def _load_scaler(scaler_path: str, n_features: int) -> MinMaxScaler:
    s = pd.read_csv(scaler_path)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.data_min_      = s["data_min"].values
    scaler.data_max_      = s["data_max"].values
    scaler.scale_         = s["scale"].values
    scaler.min_           = s["min"].values
    scaler.n_features_in_ = n_features
    return scaler


# Calculates RSI from a price series
def _compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain  = delta.clip(lower=0)
    loss  = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs  = avg_gain / (avg_loss + 1e-10)  # avoid division by zero
    rsi = 100 - (100 / (1 + rs))
    return rsi


# Downloads price data and adds technical indicator features
def _build_features(ticker: str) -> pd.DataFrame:
    df = yf.download(ticker, period="5y", interval="1d")[["Close", "Volume"]].dropna()
    if df is None or df.empty or len(df) < LOOKBACK + 50:
        raise ValueError("Not enough data for this ticker.")

    # flatten multiindex if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    close = df["Close"].squeeze()

    # technical indicators
    df["MA20"] = close.rolling(window=20).mean()
    df["MA50"] = close.rolling(window=50).mean()
    df["RSI"]  = _compute_rsi(close)

    df = df[FEATURES].dropna()
    return df


# LSTM model architecture - now takes multiple features as input
def _build_model(n_features: int):
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(LOOKBACK, n_features)),  # LSTM layer identifies patterns
        LSTM(64),  # refines features
        Dense(1)   # output layer predicts next close price
    ])
    model.compile(optimizer="adam", loss="mse")
    return model


# Creates sliding-window sequences for LSTM training
def _make_windows(series_scaled: np.ndarray):
    X, y = [], []
    for i in range(LOOKBACK, len(series_scaled)):
        X.append(series_scaled[i - LOOKBACK:i])   # past LOOKBACK days of all features
        y.append(series_scaled[i, 0])              # predict next close price (index 0)
    X = np.array(X)                                # shape (N, LOOKBACK, n_features)
    y = np.array(y).reshape(-1, 1)
    return X, y


# Loads (or trains-and-caches) the model and scaler for a ticker
def _load_or_train(ticker: str):
    os.makedirs(SAVED_DIR, exist_ok=True)

    ticker = ticker.upper().strip()
    model_path, scaler_path = _paths(ticker)

    df     = _build_features(ticker)
    values = df.values  # shape (N, n_features)
    n_features = values.shape[1]

    if os.path.exists(model_path) and os.path.exists(scaler_path):
        model  = load_model(model_path)
        scaler = _load_scaler(scaler_path, n_features)
    else:
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled = scaler.fit_transform(values)

        X, y = _make_windows(scaled)

        model = _build_model(n_features)
        es = EarlyStopping(monitor="loss", patience=3, restore_best_weights=True)
        model.fit(X, y, epochs=10, batch_size=32, verbose=0, callbacks=[es])

        model.save(model_path)
        _save_scaler(scaler, scaler_path)

    return model, scaler, values, df.index

# Rolls the model forward from a given scaled window for `days` steps
# Only the close price (index 0) is predicted; other features are held at last known values
def _forecast_from_window(model, scaler, window_scaled: np.ndarray, days: int):
    n_features = window_scaled.shape[1]
    window = window_scaled[-LOOKBACK:].reshape(1, LOOKBACK, n_features)

    future_scaled = []
    last_row = window_scaled[-1].copy()  # holds last known values of all features

    for _ in range(days):
        next_close_scaled = model.predict(window, verbose=0)[0][0]  # predict next close

        # build next row: update close price, keep other features at last known values
        next_row = last_row.copy()
        next_row[0] = next_close_scaled  # index 0 is Close
        future_scaled.append(next_row)
        last_row = next_row

        # slide window forward
        window = np.append(window[0, 1:, :], [next_row], axis=0).reshape(1, LOOKBACK, n_features)

    # inverse transform to get real prices
    future_arr = np.array(future_scaled)
    future_full = scaler.inverse_transform(future_arr)
    future_close = future_full[:, 0]

    return [round(float(v), 2) for v in future_close]


# predicts future closing prices from today
def predict_next_days(ticker: str, days: int):
    model, scaler, values, _ = _load_or_train(ticker)
    scaled_all = scaler.transform(values)
    return _forecast_from_window(model, scaler, scaled_all[-LOOKBACK:], days)


#  backtesting to evaluate model accuracy
def backtest(ticker: str, eval_days: int = 60):
    
    model, scaler, values, index = _load_or_train(ticker)

    total_needed = LOOKBACK + eval_days
    if len(values) < total_needed:
        raise ValueError(
            f"Not enough history to backtest {eval_days} days "
            f"(need {total_needed}, have {len(values)})."
        )

    # input window: LOOKBACK days immediately before the test period
    # test period: last eval_days days (the "future" from the model's POV)
    input_values  = values[-(eval_days + LOOKBACK): -eval_days]
    actual_values = values[-eval_days:, 0]  # close price only
    test_dates    = index[-eval_days:]

    scaled_input = scaler.transform(input_values)
    predicted    = _forecast_from_window(model, scaler, scaled_input, eval_days)
    actual       = [round(float(v), 2) for v in actual_values.flatten()]
    dates        = [str(d.date()) for d in test_dates]

    actual_arr = np.array(actual)
    pred_arr   = np.array(predicted)

    rmse = float(np.sqrt(np.mean((actual_arr - pred_arr) ** 2)))
    mape = float(np.mean(np.abs((actual_arr - pred_arr) / actual_arr)) * 100)

    if mape < 2:
        reliability = "High"
    elif mape <= 5:
        reliability = "Medium"
    else:
        reliability = "Low"

    return {
        "predicted":   predicted,
        "actual":      actual,
        "dates":       dates,
        "rmse":        round(rmse, 2),
        "mape":        round(mape, 2),
        "reliability": reliability,
    }


# Prophet forecast for long-term trend prediction
# needed as current predicitons are suitbale for short term 
def prophet_forecast(ticker: str, days: int):

    from prophet import Prophet

    # download 5 years of data - prophet needs column names 'ds' and 'y'
    df = yf.download(ticker, period="5y", interval="1d")[["Close"]].dropna()
    if df is None or df.empty or len(df) < 30:
        raise ValueError("Not enough data for Prophet forecast.")

    # prophet requires specific column names
    df = df.reset_index()
    df.columns = ["ds", "y"]
    if isinstance(df["y"].iloc[0], pd.Series):
        df["y"] = df["y"].apply(lambda x: x.iloc[0])
    df["ds"] = pd.to_datetime(df["ds"]).dt.tz_localize(None)
    df["y"]  = pd.to_numeric(df["y"], errors="coerce")
    df = df.dropna()

    # fit model, no weekly seasonality to avoid jagged band
    model = Prophet(daily_seasonality=False, weekly_seasonality=False, yearly_seasonality=True, interval_width=0.5)
    model.fit(df)

    # create future dataframe and predict
    future   = model.make_future_dataframe(periods=days)
    forecast = model.predict(future)

    # return only the future portion
    future_forecast = forecast.tail(days)
    return {
        "dates":  [str(d.date()) for d in future_forecast["ds"]],
        "values": [round(float(v), 2) for v in future_forecast["yhat"]],
        "upper":  [round(float(v), 2) for v in future_forecast["yhat_upper"]],
        "lower":  [round(float(v), 2) for v in future_forecast["yhat_lower"]],
    }