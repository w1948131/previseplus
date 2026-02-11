import os
import numpy as np
import pandas as pd
import yfinance as yf

from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping

LOOKBACK = 60 # past trading days used to predict next value
SAVED_DIR = "predictor/saved" #trained models and scalers are stored in this path

# generates file path for the trained LSTM model & scaler
def _paths(ticker: str):
    t = ticker.upper().replace("^", "")
    model_path = os.path.join(SAVED_DIR, f"{t}_lstm.keras")
    scaler_path = os.path.join(SAVED_DIR, f"{t}_scaler.csv")
    return model_path, scaler_path

#for training 
# Persists minmax scaler parameters, allows it to be used again
def _save_scaler(scaler: MinMaxScaler, scaler_path: str):
    s = pd.DataFrame({ #using pandas dataframe to store scalers
        "data_min": scaler.data_min_.flatten(), # flatten makes it 1d array
        "data_max": scaler.data_max_.flatten(),
        "scale": scaler.scale_.flatten(), #scaling calculated by min max
        "min": scaler.min_.flatten(),
    })
    s.to_csv(scaler_path, index=False) # saved into csv file to be used

# re loads saved minmaxscaler for consistent preprocessing 
def _load_scaler(scaler_path: str) -> MinMaxScaler:
    s = pd.read_csv(scaler_path) #loads the saved scaler
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.data_min_ = s["data_min"].values #/ restores values so new market data is normalised to training
    scaler.data_max_ = s["data_max"].values
    scaler.scale_ = s["scale"].values 
    scaler.min_ = s["min"].values #/ 
    scaler.n_features_in_ = 1 # use close price for scaling
    return scaler

# where LSTM model architecture is defined
def _build_model():
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(LOOKBACK, 1)), #LSTM layer identify patterns from past prices
        LSTM(64), # another layer that refines features
        Dense(1) # Output layer that predicts prices
    ])
    model.compile(optimizer="adam", loss="mse")  #Training preparation for learning and evaluation
    return model

#  creates suitable sequences for LSTM to learn from
def _make_windows(series_scaled: np.ndarray):
    X, y = [], []
    for i in range(LOOKBACK, len(series_scaled)): # loops through 60 days of data to predict
        X.append(series_scaled[i - LOOKBACK:i]) # past 60 days of  prices added to x
        y.append(series_scaled[i]) # real next day price
    X = np.array(X).reshape(-1, LOOKBACK, 1) # X and Y is converted to numpy array to be used by tensor
    y = np.array(y).reshape(-1, 1)
    return X, y

# Predicts closing prices with LSTM model on chosen ticker 
def predict_next_days(ticker: str, days: int):
    os.makedirs(SAVED_DIR, exist_ok=True) # directory exists to store models/scalers

    ticker = ticker.upper().strip() # set ticker format
    model_path, scaler_path = _paths(ticker) # path to the relevant tickers model + scaler

    # takes 5yrs of daily closing prices from yfinance 
    df = yf.download(ticker, period="5y", interval="1d")[["Close"]].dropna() 
    if df is None or df.empty or len(df) < LOOKBACK + 10: # validation for amount of data 
        raise ValueError("Not enough data for this ticker.")

    values = df.values # pd format to numpy array

    # loads an existing model and scaler
    if os.path.exists(model_path) and os.path.exists(scaler_path):
        model = load_model(model_path)
        scaler = _load_scaler(scaler_path)
    else:
        scaler = MinMaxScaler(feature_range=(0, 1)) #creates new scalar on else
        scaled = scaler.fit_transform(values)

        X, y = _make_windows(scaled) #scaled into 60 days and next day price

        model = _build_model() # LSTM model is created 
        es = EarlyStopping(monitor="loss", patience=3, restore_best_weights=True) # training stops if no improvement
        model.fit(X, y, epochs=10, batch_size=32, verbose=0, callbacks=[es]) # trains model on historical price windows

        model.save(model_path)  # save trained model and scalar
        _save_scaler(scaler, scaler_path)

   # Predicts future prices
    scaled_all = scaler.transform(values)  # scales historical closes for specific ticker
    window = scaled_all[-LOOKBACK:].reshape(1, LOOKBACK, 1) # formatted into sequences for LSTM 

    future_scaled = [] # list of future predictions
    for _ in range(days):
        next_val = model.predict(window, verbose=0)[0][0] #next day predicted price
        future_scaled.append([next_val])
        window = np.append(window[0, 1:, 0], next_val).reshape(1, LOOKBACK, 1) # allows for multi day predictions

    future = scaler.inverse_transform(np.array(future_scaled)).flatten() # scaled into real price prediciton
    return [round(float(v), 2) for v in future] # correctly formatted output

