import os
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

#Contains model training on s&p500 historical data

#s&p500 ticker
TICKER = "^GSPC"
LOOKBACK = 60
EPOCHS = 10 # training iterations
BATCH_SIZE = 32 

#  past 5 years of s&p500 data
df = yf.download(TICKER, period="5y", interval="1d")
close = df[["Close"]].dropna().values # filters to close prices only 

# close prices are normalised to [0,1]
scaler = MinMaxScaler()
scaled = scaler.fit_transform(close)

# time series data to learning format
X, y = [], []
for i in range(LOOKBACK, len(scaled)):
    X.append(scaled[i-LOOKBACK:i]) # past 60 day window
    y.append(scaled[i])

X = np.array(X).reshape(-1, LOOKBACK, 1) #reshape for LSTM input, numpy format
y = np.array(y)

# defines LSTM architecture
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(LOOKBACK, 1)),
    Dropout(0.2),
    LSTM(50),
    Dropout(0.2),
    Dense(1)
])
model.compile(optimizer="adam", loss="mse")

# LSTM model trains on the historical s&p500 data
model.fit(X, y, epochs=EPOCHS, batch_size=BATCH_SIZE)


os.makedirs("backend/predictor/saved", exist_ok=True) 
model.save("backend/predictor/saved/sp500_lstm.keras") # model is saved 

pd.DataFrame({ # table is created in pandas 
    "data_min": scaler.data_min_,
    "data_max": scaler.data_max_,
    "scale": scaler.scale_,
    "min": scaler.min_
}).to_csv("backend/predictor/saved/sp500_scaler.csv", index=False) # saved into csv file

print("Saved model: backend/predictor/saved/sp500_lstm.keras")
print("Saved scaler: backend/predictor/saved/sp500_scaler.csv")
