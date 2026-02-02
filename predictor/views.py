from django.shortcuts import render, redirect
from django.contrib.auth.decorators import login_required
from .lstm_prediction import predict_next_days
from plotly.offline import plot
import plotly.graph_objects as go 
import pandas as pd
import datetime as dt  
import yfinance as yf
import json

# Create your views here.
def home(request):
    return render(request, "home.html")

def search(request):
    if request.method == "POST":
        ticker = request.POST.get("ticker")
        days = request.POST.get("days")
        return redirect("predict", ticker_value=ticker, number_of_days=days)

    return render(request, "search.html")


@login_required
def dashboard(request):
   
    # download 1 month daily data for these tickers
    data = yf.download(
        tickers = ["AAPL", "AMZN", "NVDA", "META", "MSFT", "JPM"],
        
        group_by="ticker",
        period="1mo",
        interval="1d",
        threads=True
    )

    data.reset_index(level=0, inplace=True)

    fig_left = go.Figure()
    fig_left.add_trace(
        go.Scatter(x=data['Date'], y=data['AAPL']['Close'], name ="AAPL")
    )
    fig_left.add_trace(
        go.Scatter(x=data['Date'], y=data['AMZN']['Close'], name ="AMZN")
    )
    fig_left.add_trace(
        go.Scatter(x=data['Date'], y=data['NVDA']['Close'], name ="NVDA")
    )
    fig_left.add_trace(
        go.Scatter(x=data['Date'], y=data['META']['Close'], name ="META")
    )
    fig_left.add_trace(
        go.Scatter(x=data['Date'], y=data['MSFT']['Close'], name ="MSFT")
    )
    fig_left.add_trace(
        go.Scatter(x=data['Date'], y=data['JPM']['Close'], name ="JPM")
    )
    
    fig_left.update_layout(
        title = "Monthly Active Stocks",
        paper_bgcolor="#14151b",
        plot_bgcolor="#14151b",
        font_color="white"
    )
    

    plot_div_left = plot(fig_left, auto_open=False, output_type="div", include_plotlyjs="cdn")

    return render(request, "dashboard.html", {
        "plot_div_left": plot_div_left,
    })

# renders ticker list
def ticker(request):
    ticker_df = pd.read_csv("predictor/Data/new_tickers.csv")
    json_ticker = ticker_df.reset_index().to_json(orient="records")
    ticker_list = json.loads(json_ticker)
    
    return render(request, "ticker.html", {
        "ticker_list": ticker_list,
    })
    
def predict(request, ticker_value, number_of_days):
    
    try:
        number_of_days = int(number_of_days)
    except:
        return render(request, "Invalid_Days_Format.html", {})
    
    if number_of_days < 0:
        return render(request, "Negative_Days.html", {})
    if number_of_days > 365:
        return render(request, "Overflow_days.html", {})
    
    valid_df = pd.read_csv("predictor/Data/new_tickers.csv")
    col = "Symbol" if "Symbol" in valid_df.columns else valid_df.columns[0]
    valid_set = set(valid_df[col].astype(str).str.upper())
    
    if ticker_value not in valid_set:
        return render(request, "Invalid_Ticker.html", {})
    
    
   # ---------------- live stock data  ----------------
    try:
        df = yf.download(tickers=ticker_value, period="5d", interval="5m")
        if df is None or df.empty:
            return render(request, "Invalid_Ticker.html", {})
    except:
        return render(request, "API_Down.html", {})

    # ---------------- Live candlestick graph ----------------
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df["Open"],
        high=df["High"],
        low=df["Low"],
        close=df["Close"],
        name="market data"
    ))
    fig.update_layout(
        title=f"{ticker_value} live share price evolution",
        yaxis_title="Stock Price (USD per Shares)",
        paper_bgcolor="#14151b",
        plot_bgcolor="#14151b",
        font_color="white"
    )
    plot_div = plot(fig, auto_open=False, output_type="div", include_plotlyjs=False)

    # ---------------- Load ticker info table ----------------
    info_df = pd.read_csv("predictor/Data/Tickers.csv")
    info_df = info_df.drop(columns=["Last Sale"]) # dont need as info is not split from csv
    info_df.columns = ["Symbol", "Name", "Net_Change", "Percent_Change", "Market_Cap",
                       "Country", "IPO_Year", "Volume", "Sector", "Industry"]

    Symbol = Name = Net_Change = Percent_Change = Market_Cap = "N/A"
    Country = IPO_Year = Volume = Sector = Industry = "N/A"

    row = info_df[info_df["Symbol"].astype(str).str.upper() == ticker_value]
    if not row.empty:
        row = row.iloc[0]
        Symbol = row["Symbol"]
        Name = row["Name"]
        Net_Change = row["Net_Change"]
        Percent_Change = row["Percent_Change"]
        Market_Cap = row["Market_Cap"]
        Country = row["Country"]
        IPO_Year = row["IPO_Year"]
        Volume = row["Volume"]
        Sector = row["Sector"]
        Industry = row["Industry"]

    # Prediction + prediction graph 
    
    forecast = predict_next_days(ticker_value, number_of_days)  
    confidence = "Trained LSTM Model"

    pred_dates = [dt.datetime.today() + dt.timedelta(days=i) for i in range(len(forecast))]
    pred_fig = go.Figure([go.Scatter(x=pred_dates, y=forecast)])
    pred_fig.update_layout(
        title=f"Predicted Stock price of {ticker_value} for next {number_of_days} days",
        paper_bgcolor="#14151b",
        plot_bgcolor="#14151b",
        font_color="white"
    )
    plot_div_pred = plot(pred_fig, auto_open=False, output_type="div", include_plotlyjs=False)

    #  Rendering results.html 
    return render(request, "results.html", {
        "plot_div": plot_div,
        "plot_div_pred": plot_div_pred,
        "confidence": confidence,
        "forecast": forecast,
        "ticker_value": ticker_value,
        "number_of_days": number_of_days,

        "Symbol": Symbol,
        "Name": Name,
        "Net_Change": Net_Change,
        "Percent_Change": Percent_Change,
        "Market_Cap": Market_Cap,
        "Country": Country,
        "IPO_Year": IPO_Year,
        "Volume": Volume,
        "Sector": Sector,
        "Industry": Industry,
    })
        
    
    
    

