from django.shortcuts import render, redirect
from django.contrib.auth.decorators import login_required
from .lstm_prediction import predict_next_days
from plotly.offline import plot
import plotly.graph_objects as go 
import pandas as pd
import numpy as np 
import datetime as dt  
import yfinance as yf
import json

# Create your views here.
def home(request):
    return render(request, "home.html")

#ticker search
def search(request):
    if request.method == "POST":
        ticker = request.POST.get("ticker")
        days = request.POST.get("days")
        return redirect("predict", ticker_value=ticker, number_of_days=days)

    return render(request, "search.html")


@login_required
def dashboard(request):
   
    # pull stock info from yfinance ---- Future goal: pull most popular instead
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
    
    #dashboard plot style
    fig_left.update_layout(
        title = "Monthly Active Stocks",
        paper_bgcolor="#14151b",
        plot_bgcolor="#14151b",
        font_color="white"
    )
    
    #plot to html
    plot_div_left = plot(fig_left, auto_open=False, output_type="div", include_plotlyjs="cdn")

    return render(request, "dashboard.html", {
        "plot_div_left": plot_div_left,
        
    })

# renders list of tickers
def ticker(request):
    ticker_df = pd.read_csv("predictor/Data/new_tickers.csv")
    json_ticker = ticker_df.reset_index().to_json(orient="records")
    ticker_list = json.loads(json_ticker) #made into list so it can be looped
    
    return render(request, "ticker.html", {
        "ticker_list": ticker_list,
    })
    
    
#prediciton from stock search    
def predict(request, ticker_value, number_of_days):
    
    try:
        ticker_value=ticker_value.upper().strip()
        number_of_days = int(number_of_days) #validates days format
    except:
        return render(request, "Invalid_Days_Format.html", {})  #not added yet
    
    if number_of_days < 0:
        return render(request, "Negative_Days.html", {}) #not added yet
    if number_of_days > 365:
        return render(request, "Overflow_days.html", {}) # not added yet
    
    valid_df = pd.read_csv("predictor/Data/new_tickers.csv") #ticker validation
    col = "Symbol" if "Symbol" in valid_df.columns else valid_df.columns[0] #checks ticker symbol
    valid_set = set(valid_df[col].astype(str).str.upper())
    
    if ticker_value not in valid_set:
        return render(request, "Invalid_Ticker.html", {}) # not added yet
  
    data = yf.download( #3month chart action
        
        ticker_value,
        period="3mo",
        interval="1d",
        threads=True
    )
    
    if isinstance(data.columns, pd.MultiIndex): # must multi style indexing for ticker graph
        close = data["Close"][ticker_value]
    else:
        close = data["Close"]

    data = data.reset_index()

    fig_left = go.Figure()
    fig_left.add_trace(go.Scatter(x=data["Date"], y=close.values))
    #recent price action of stock graph
    fig_left.update_layout(
        title=f"{ticker_value} Recent share price evolution",
        yaxis_title="Stock Price (USD per Shares)",
        paper_bgcolor="#14151b",
        plot_bgcolor="#14151b",
        font_color="white"
    )
    
    plot_div = plot(fig_left, auto_open=False, output_type="div", include_plotlyjs="cdn")

    # loads relevant ticker table 
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

    
    # LSTM prediction and graph creation
    forecast = predict_next_days(ticker_value, number_of_days)  
    confidence = "Trained LSTM Model"

    pred_dates = [dt.datetime.today() + dt.timedelta(days=i) for i in range(len(forecast))] # future dates creation
    pred_fig = go.Figure([go.Scatter(x=pred_dates, y=forecast)])
    pred_fig.update_layout(
        title=f"Predicted Stock price of {ticker_value} for next {number_of_days} days",
        paper_bgcolor="#14151b",
        plot_bgcolor="#14151b",
        font_color="white"
    )
    plot_div_pred = plot(pred_fig, auto_open=False, output_type="div", include_plotlyjs=False)
    
    
    # Evaluation metrics to reliability score
    try: 
        eval_days = 60 
        
        hist = yf.download(ticker_value, period="6mo", interval="1d", progress=False)[["Close"]].dropna()
        
        actual = hist["Close"].values[-eval_days:] # actual close prices
        predicted = predict_next_days(ticker_value, eval_days) # prediction close prices
        
        rmse = np.sqrt(np.mean((actual - predicted) ** 2)) # average prediction error mainly for professional traders
        mape = np.mean(np.abs((actual - predicted) / actual)) * 100 #average percentage error
        
        # easy translation of MAPE for amateur traders
        if mape < 2:
            relability = "High"
        elif mape <= 5:
            relability = "Medium"
        else: 
            relability = "low"
    except:
        rmse = mape = "N/A"
        relability = "N/A"

    #  Rendering results.html 
    return render(request, "results.html", {
        "plot_div": plot_div,
        "plot_div_pred": plot_div_pred,
        "confidence": confidence,
        "forecast": forecast,
        "ticker_value": ticker_value,
        "number_of_days": number_of_days,
    #ticker table  
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
        "relability": relability,
        "rmse": round(rmse , 2) if rmse != "N/A" else rmse,
        "mape": round(mape, 2) if mape != "N/A" else mape,
    })
        
    
    
    

