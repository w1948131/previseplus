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
        LSTM(128, return_sequences=True, input_shape=(LOOKBACK, n_features)),  # LSTM layer identifies patterns
        LSTM(128),  # refines features
        Dense(365)  # outputs full year at once 
    ])
    model.compile(optimizer="adam", loss="mse")
    return model


# Creates sliding-window sequences for LSTM training
def _make_windows(series_scaled: np.ndarray):
    X, y = [], []
    for i in range(LOOKBACK, len(series_scaled) - 365):
        X.append(series_scaled[i - LOOKBACK:i])   # past LOOKBACK days of all features
        y.append(series_scaled[i:i + 365, 0])     # 365days of close prices 
    X = np.array(X)                               # shape (N, LOOKBACK, n_features)
    y = np.array(y)
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
        model.fit(X, y, epochs=50, batch_size=32, verbose=0, callbacks=[es])

        model.save(model_path)
        _save_scaler(scaler, scaler_path)

    return model, scaler, values, df.index

# Rolls the model forward from a given scaled window for `days` steps
# Only the close price (index 0) is predicted; other features are held at last known values
def _forecast_from_window(model, scaler, window_scaled: np.ndarray, days: int):
    n_features = window_scaled.shape[1]
    window = window_scaled[-LOOKBACK:].reshape(1, LOOKBACK, n_features)


    all_predictions_scaled = model.predict(window, verbose=0)[0]
    
    dummy = np.zeros((365, n_features))
    dummy[:, 0] = all_predictions_scaled
    all_predictions_real = scaler.inverse_transform(dummy)[:, 0]
    
    return [round(float(v), 2) for v in all_predictions_real[:days]]

    
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
    
    
# News sentiment analysis using VADER and NewsAPI
# Fetches recent headlines and scores sentiment for a given ticker
def get_sentiment(ticker: str, company_name: str = ""):
    import requests
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    from datetime import datetime, timezone

    api_key = os.environ.get("NEWS_API_KEY", "d2379e2cdfba413091dbebe97a224f00")

    # use company name if provided, otherwise use ticker
    short_name = " ".join(company_name.split()[:2]) if company_name else ticker
    query = f"{ticker} +{short_name}"
    url = (
        f"https://newsapi.org/v2/everything?"
        f"q={query}&language=en&sortBy=publishedAt&pageSize=20&apiKey={api_key}"
    )

    try:
        response = requests.get(url, timeout=10)
        articles = response.json().get("articles", [])

        # filter to only articles mentioning ticker or company name in headline
        keywords = [ticker.lower(), short_name.lower().split()[0]]
        filtered = [
            a for a in articles
            if a.get("title") and any(k in a["title"].lower() for k in keywords)
        ]

        # use filtered if we have enough, otherwise fall back to all articles
        articles = filtered if len(filtered) >= 3 else articles

        headlines = [{"title": a["title"], "url": a.get("url", "#"), "publishedAt": a.get("publishedAt", "")} for a in articles if a.get("title")][:10]

    except Exception:
        return {"score": "N/A", "label": "N/A", "headlines": []}

    if not headlines:
        return {"score": "N/A", "label": "N/A", "headlines": []}

    # score each headline using VADER with recency weighting
    analyzer = SentimentIntensityAnalyzer()
    scores = []
    weights = []
    now = datetime.now(timezone.utc)

    for i, headline in enumerate(headlines):
        compound = analyzer.polarity_scores(headline["title"])["compound"]

        # calculate recency weight - more recent articles get higher weight
        try:
            pub_date = datetime.fromisoformat(headline["publishedAt"].replace("Z", "+00:00"))
            days_old = (now - pub_date).days
            weight = 1 / (1 + days_old)  # articles from today get weight 1, older get less
        except Exception:
            weight = 0.5  # default weight if date parsing fails

        scores.append(compound)
        weights.append(weight)

    # weighted average score
    total_weight = sum(weights)
    avg_score = round(sum(s * w for s, w in zip(scores, weights)) / total_weight, 4)

    if avg_score > 0.05:
        label = "Positive"
    elif avg_score < -0.05:
        label = "Negative"
    else:
        label = "Neutral"

    return {
        "score":     avg_score,
        "label":     label,
        "headlines": [{"title": h["title"], "url": h["url"], "score": round(analyzer.polarity_scores(h["title"])["compound"], 4)} for h in headlines],
    }