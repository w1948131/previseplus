from django.shortcuts import render, redirect
from django.contrib.auth.decorators import login_required
from .lstm_prediction import predict_next_days, backtest, prophet_forecast, get_sentiment
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

    #return render(request, "search.html")
    
    # Stock mapping for prediction page
    try:
        df = pd.read_csv("predictor/Data/Tickers.csv")
        valid_df = pd.read_csv("predictor/Data/new_tickers.csv")
        valid_set = set(valid_df["Symbol"].astype(str).str.upper())
        df = df[df["Symbol"].astype(str).str.upper().isin(valid_set)]
        df = df[df["Sector"].notna()]
        
        sectors = {}
        for _, row in df.iterrows():
            s = row["Sector"]
            if s not in sectors:
                sectors[s] = []
            sectors[s].append({
                "symbol": str(row["Symbol"]).upper(),
                "name": str(row["Name"]).strip(),
            })
        for s in sectors:
            sectors[s] = sorted(sectors[s], key=lambda x: x["symbol"])
        sectors = dict(sorted(sectors.items()))
    except Exception:
        sectors = {}
 
    return render(request, "search.html", {"sectors": json.dumps(sectors)})
        


@login_required
def dashboard(request):
    tickers = ["AAPL", "AMZN", "NVDA", "META", "MSFT", "JPM"]
    
    # fetch price data
    try:
        data = yf.download(
            tickers=tickers,
            group_by="ticker",
            period="1mo",
            interval="1d",
            threads=True,
            timeout=30
        )
        data.reset_index(level=0, inplace=True)

        fig_left = go.Figure()
        for sym in tickers:
            fig_left.add_trace(go.Scatter(x=data['Date'], y=data[sym]['Close'], name=sym))

        #dashboard plot style
        fig_left.update_layout(
            title="Monthly Active Stocks",
            paper_bgcolor="#14151b",
            plot_bgcolor="#14151b",
            font_color="white"
        )

        #plot to html
        plot_div_left = plot(fig_left, auto_open=False, output_type="div", include_plotlyjs="cdn")

    except Exception:
        plot_div_left = None

    # fetch sentiment and volume for bubble chart
    bubble_data = []
    for sym in tickers:
        try:
            info = yf.Ticker(sym).fast_info
            volume = int(info.three_month_average_volume or 1000000)
            price = round(float(info.last_price or 0), 2)
            prev_close = round(float(info.previous_close or 0), 2)
            change_pct = round(((price - prev_close) / prev_close) * 100, 2) if prev_close else 0
            market_cap = int(info.market_cap or 0)
        except Exception:
            volume = 1000000
            price = 0
            change_pct = 0
            market_cap = 0

        try:
            sent = get_sentiment(sym, sym)
            score = sent["score"] if sent["score"] != "N/A" else 0
            label = sent["label"] if sent["label"] != "N/A" else "Neutral"
        except Exception:
            score = 0
            label = "Neutral"

        bubble_data.append({
            "ticker": sym,
            "volume": volume,
            "score":  score,
            "label":  label,
            "price": price,
            "change_pct": change_pct,
            "market_cap": market_cap,
            
        })

    return render(request, "dashboard.html", {
        "plot_div_left": plot_div_left,
        "bubble_data":   json.dumps(bubble_data),
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
        threads=True,
        timeout=30
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
    #forecast = predict_next_days(ticker_value, number_of_days)  
    #confidence = "Trained LSTM Model"
    
   
    try:
        forecast = predict_next_days(ticker_value, number_of_days)
        pred_dates = [dt.datetime.today() + dt.timedelta(days=i) for i in range(len(forecast))] # future dates creation
        pred_fig = go.Figure([go.Scatter(x=pred_dates, y=forecast)])
        pred_fig.update_layout(
            title=f"Predicted Stock price of {ticker_value} for next {number_of_days} days",
            paper_bgcolor="#14151b",
            plot_bgcolor="#14151b",
            font_color="white"
        )
        plot_div_pred = plot(pred_fig, auto_open=False, output_type="div", include_plotlyjs=False)
    except ValueError as e:
        return redirect(f"/search?error={ticker_value}")
    
    
    #long term forecast
    try:
        pf = prophet_forecast(ticker_value, number_of_days)
        
        if pf and forecast:
            offset = forecast[0] - pf["values"][0]
            pf["values"] = [round(v + offset, 2) for v in pf["values"]]
            pf["upper"]  = [round(v + offset, 2) for v in pf["upper"]]
            pf["lower"]  = [round(v + offset, 2) for v in pf["lower"]]
            
    except Exception:
        pf = None
        
        
    # news sentiment analysis
    try:
        sentiment = get_sentiment(ticker_value, Name if Name != "N/A" else ticker_value)
    except Exception:
        sentiment = {"score": "N/A", "label": "N/A", "headlines": []}
        
        
    # adjust forecast based on news sentiment score
    if sentiment and sentiment["score"] != "N/A":
        sentiment_adjustment = 1 + (sentiment["score"] * 0.02)
        forecast = [round(p * sentiment_adjustment, 2) for p in forecast]

    # ensemble: average LSTM and Prophet to reduce compounding error
    if pf:
        forecast = [round((l + p) / 2, 2) for l, p in zip(forecast, pf["values"])]
        
    
    # Evaluation metrics to reliability score
    try: 
        bt = backtest(ticker_value, eval_days=30)
        
        bt_fig = go.Figure()
        bt_fig.add_trace(go.Scatter(x=bt["dates"], y=bt["actual"], name="Actual", line=dict(color="#4fc3f7")))
        bt_fig.add_trace(go.Scatter(x=bt["dates"], y=bt["predicted"], name="Predicted", line=dict(color="#ff3b3b", dash="dash")))
        bt_fig.update_layout(
            title=f"{ticker_value} Backtest: Last 30 Trading Days",
            yaxis_title="Price ($ USD)",
            paper_bgcolor="#14151b",
            plot_bgcolor="#14151b",
            font_color="white",
            legend=dict(bgcolor="rgba(0,0,0,0)"),
        )
        plot_div_backtest =  plot(bt_fig, auto_open=False, output_type="div", include_plotlyjs=False)
        
        relability = bt["reliability"]
        rmse = bt["rmse"]
        mape = bt["mape"]
  
    except Exception:
        plot_div_backtest = None
        rmse = mape = "N/A"
        relability = "N/A"
             

    #  Rendering results.html 
    return render(request, "results.html", {
        "plot_div": plot_div,
        "plot_div_pred": plot_div_pred,
        "plot_div_backtest": plot_div_backtest,
        "pf": pf,
        "forecast": forecast,
        "ticker_value": ticker_value,
        "number_of_days": number_of_days,
        "sentiment": sentiment,
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
        
    
    
    

