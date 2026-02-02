import os
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

TICKER = "^GSPC"
LOOKBACK = 60
EPOCHS = 10
BATCH_SIZE = 32

# 1) Download 5y data
df = yf.download(TICKER, period="5y", interval="1d")
close = df[["Close"]].dropna().values

# 2) Scale
scaler = MinMaxScaler()
scaled = scaler.fit_transform(close)

# 3) Make sequences
X, y = [], []
for i in range(LOOKBACK, len(scaled)):
    X.append(scaled[i-LOOKBACK:i])
    y.append(scaled[i])

X = np.array(X).reshape(-1, LOOKBACK, 1)
y = np.array(y)

# 4) Build model
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(LOOKBACK, 1)),
    Dropout(0.2),
    LSTM(50),
    Dropout(0.2),
    Dense(1)
])
model.compile(optimizer="adam", loss="mse")

# 5) Train ONCE
model.fit(X, y, epochs=EPOCHS, batch_size=BATCH_SIZE)

# 6) Save model + scaler
os.makedirs("backend/predictor/saved", exist_ok=True)
model.save("backend/predictor/saved/sp500_lstm.keras")

pd.DataFrame({
    "data_min": scaler.data_min_,
    "data_max": scaler.data_max_,
    "scale": scaler.scale_,
    "min": scaler.min_
}).to_csv("backend/predictor/saved/sp500_scaler.csv", index=False)

print("Saved model: backend/predictor/saved/sp500_lstm.keras")
print("Saved scaler: backend/predictor/saved/sp500_scaler.csv")
