```python
import pandas as pd
import numpy as np
```


```python
def calculate_returns(df, periods=None):
    if periods is None:
        periods = ['1d', '1w', '1m', '3m', '6m', '1y', 'ytd']
    returns = {}
    prices = df['Close'].values
    if len(prices) > 0:
        current_price = prices[-1]
        if len(prices) > 1 and '1d' in periods:
            returns['1d'] = (current_price / prices[-2] - 1) * 100
        if len(prices) > 5 and '1w' in periods:
            returns['1w'] = (current_price / prices[-6] - 1) * 100
        if len(prices) > 21 and '1m' in periods:
            returns['1m'] = (current_price / prices[-21] - 1) * 100
        if len(prices) > 63 and '3m' in periods:
            returns['3m'] = (current_price / prices[-63] - 1) * 100
        if len(prices) > 126 and '6m' in periods:
            returns['6m'] = (current_price / prices[-127] - 1) * 100
        if len(prices) > 252 and '1y' in periods:
            returns['1y'] = (current_price / prices[-253] - 1) * 100
        if 'ytd' in periods and df['Date'].dt.year.nunique() > 1:
            start_of_year_idx = df[df['Date'].dt.year == df['Date'].dt.year.max()].index[0]
            start_of_year_price = df.loc[start_of_year_idx, 'Close']
            returns['ytd'] = (current_price / start_of_year_price - 1) * 100
        return returns
```


```python
data = {'Date': pd.to_datetime(['2025-05-10', '2025-05-11', '2025-05-12', '2025-05-13', '2025-05-14',
                               '2025-05-15', '2025-05-16', '2025-05-17', '2024-12-31']),
        'Close': [100, 102, 105, 103, 106, 108, 110, 112, 95]}
df = pd.DataFrame(data)
df = df.sort_values(by='Date').reset_index(drop=True)
returns = calculate_returns(df)
number_of_returns = len(returns)
print(returns)
print(number_of_returns)
x = returns.values()

```

    {'1d': np.float64(1.8181818181818077), '1w': np.float64(6.666666666666665), 'ytd': np.float64(12.00000000000001)}
    3



```python
def calculate_risk_metrics(df):
    """
    Calculate basic risk metrics

    Args:
        df (pandas.DataFrame): Stock price data with 'Close' column

    Returns:
        dict: Dictionary of risk metrics
    """
    metrics = {}

    # Calculate daily returns
    returns = df['Close'].pct_change().dropna()

    # Annualized volatility
    metrics['volatility'] = returns.std() * np.sqrt(252) * 100  # Annualized and in percentage

    # Sharpe ratio (assuming risk-free rate of 0 for simplicity)
    mean_return = returns.mean()
    metrics['sharpe_ratio'] = (mean_return * 252) / (returns.std() * np.sqrt(252)) if returns.std() != 0 else 0

    # Maximum drawdown
    cum_returns = (1 + returns).cumprod()
    running_max = cum_returns.cummax()
    drawdown = (cum_returns / running_max - 1)
    metrics['max_drawdown'] = drawdown.min() * 100  # In percentage

    # Downside deviation (returns below 0)
    negative_returns = returns[returns < 0]
    metrics['downside_deviation'] = negative_returns.std() * np.sqrt(252) * 100 if len(negative_returns) > 0 else 0

    # Sortino ratio (using 0 as minimum acceptable return)
    metrics['sortino_ratio'] = (mean_return * 252) / (metrics['downside_deviation'] / 100) if metrics['downside_deviation'] != 0 else 0

    return metrics
```


```python
def calculate_relative_strength(stock_df, market_df, period=90):
    """
    Calculate relative strength compared to market

    Args:
        stock_df (pandas.DataFrame): Stock price data with 'Close' column
        market_df (pandas.DataFrame): Market index price data with 'Close' column
        period (int): Period for calculation in days

    Returns:
        float: Relative strength value
    """
    if stock_df.empty or market_df.empty:
        return None

    # Trim both dataframes to the same period
    start_date = max(stock_df['Date'].min(), market_df['Date'].min())
    end_date = min(stock_df['Date'].max(), market_df['Date'].max())

    stock_df = stock_df[(stock_df['Date'] >= start_date) & (stock_df['Date'] <= end_date)]
    market_df = market_df[(market_df['Date'] >= start_date) & (market_df['Date'] <= end_date)]

    # Calculate percentage change over the period
    if len(stock_df) > period and len(market_df) > period:
        stock_change = (stock_df['Close'].iloc[-1] / stock_df['Close'].iloc[-period] - 1) * 100
        market_change = (market_df['Close'].iloc[-1] / market_df['Close'].iloc[-period] - 1) * 100

        # Relative strength
        if market_change != 0:
            return stock_change / market_change

    return None

def format_large_number(num):
    """
    Format large numbers with K, M, B, T suffixes

    Args:
        num (float): Number to format

    Returns:
      str: Formatted number
    """
    if num is None:
        return "N/A"

    if isinstance(num, str):
        return num

    suffix = ""
    magnitude = 0

    if abs(num) >= 1.0e12:
        suffix = "T"
        magnitude = 1.0e12
    elif abs(num) >= 1.0e9:
        suffix = "B"
        magnitude = 1.0e9
    elif abs(num) >= 1.0e6:
        suffix = "M"
        magnitude = 1.0e6
    elif abs(num) >= 1.0e3:
        suffix = "K"
        magnitude = 1.0e3

    if magnitude > 0:
        return f"{num/magnitude:.2f}{suffix}"
    else:
        return f"{num:.2f}"

```


```python
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
from io import StringIO
```


```python
def validate_ticker(ticker):
    """
    Validates if the ticker symbol exists in Yahoo Finance

    Args:
        ticker (str): Stock ticker symbol

    Returns:
        bool: True if valid, False otherwise
    """
    ticker_obj = yf.Ticker(ticker)
    try:
        info = ticker_obj.info
        # Check if we got a valid response with a name field
        if 'longName' in info:
            return True
        return False
    except:
        return False
```


```python
def get_stock_data(ticker, period="1y", interval="1d"):
    """
    Fetches historical stock data

    Args:
        ticker (str): Stock ticker symbol
        period (str): Time period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
        interval (str): Data interval (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo)

    Returns:
        pandas.DataFrame: Historical stock data
    """
    try:
        ticker_obj = yf.Ticker(ticker)
        df = ticker_obj.history(period=period, interval=interval)

        # Reset index to make Date a column
        df = df.reset_index()

        # Ensure all necessary columns exist
        if 'Date' not in df.columns and 'Datetime' in df.columns:
            df['Date'] = df['Datetime']

        if 'Stock Splits' in df.columns:
            df = df.rename(columns={'Stock Splits': 'Stock_Splits'})

        # Add simple moving averages
        if len(df) > 20:
            df['SMA20'] = df['Close'].rolling(window=20).mean()
        if len(df) > 50:
            df['SMA50'] = df['Close'].rolling(window=50).mean()
        if len(df) > 200:
            df['SMA200'] = df['Close'].rolling(window=200).mean()

        return df
    except Exception as e:
        print(f"Error fetching data for {ticker}: {e}")
        return pd.DataFrame()
```


```python
def get_company_info(ticker):
    """
    Fetches detailed company information

    Args:
        ticker (str): Stock ticker symbol

    Returns:
        dict: Company information
    """
    try:
        ticker_obj = yf.Ticker(ticker)
        info = ticker_obj.info
        return info
    except Exception as e:
        print(f"Error fetching company info for {ticker}: {e}")
        return {}

```


```python
def get_financial_data(ticker):
    """
    Fetches financial statements data

    Args:
        ticker (str): Stock ticker symbol

    Returns:
        tuple: (income_statement, balance_sheet, cash_flow)
    """
    try:
        ticker_obj = yf.Ticker(ticker)
        income_stmt = ticker_obj.income_stmt
        balance_sheet = ticker_obj.balance_sheet
        cash_flow = ticker_obj.cashflow

        return (income_stmt, balance_sheet, cash_flow)
    except Exception as e:
        print(f"Error fetching financial data for {ticker}: {e}")
        return (pd.DataFrame(), pd.DataFrame(), pd.DataFrame())
```


```python
def get_financial_ratios(ticker):
    """
    Calculate basic financial ratios from available data

    Args:
        ticker (str): Stock ticker symbol

    Returns:
        dict: Financial ratios
    """
    try:
        info = get_company_info(ticker)

        ratios = {}

        # Price ratios
        if 'trailingPE' in info:
            ratios['P/E Ratio'] = info.get('trailingPE')
        if 'priceToBook' in info:
            ratios['Price to Book'] = info.get('priceToBook')
        if 'enterpriseToEbitda' in info:
            ratios['EV/EBITDA'] = info.get('enterpriseToEbitda')
        if 'enterpriseToRevenue' in info:
            ratios['EV/Revenue'] = info.get('enterpriseToRevenue')

        # Profitability ratios
        if 'returnOnEquity' in info:
            ratios['ROE'] = info.get('returnOnEquity')
        if 'returnOnAssets' in info:
            ratios['ROA'] = info.get('returnOnAssets')
        if 'profitMargins' in info:
            ratios['Profit Margin'] = info.get('profitMargins')

        # Dividend ratios
        if 'dividendYield' in info:
            ratios['Dividend Yield'] = info.get('dividendYield')
        if 'payoutRatio' in info:
            ratios['Payout Ratio'] = info.get('payoutRatio')

        # Growth and efficiency
        if 'revenueGrowth' in info:
            ratios['Revenue Growth'] = info.get('revenueGrowth')
        if 'earningsGrowth' in info:
            ratios['Earnings Growth'] = info.get('earningsGrowth')

        return ratios
    except Exception as e:
        print(f"Error calculating financial ratios for {ticker}: {e}")
        return {}

```


```python
def get_market_news():
    """
    Fetches latest market news

    Returns:
        list: News articles
    """
    try:
        # Yahoo Finance doesn't have a direct news API in yfinance
        # We'll get news for major indices as a proxy for market news
        indices = ['^GSPC', '^DJI', '^IXIC']  # S&P 500, Dow Jones, NASDAQ
        news_items = []

        for index in indices:
            ticker = yf.Ticker(index)
            news = ticker.news
            if news:
                news_items.extend(news)

        # Remove duplicates
        unique_news = []
        seen_titles = set()

        for item in news_items:
            if item['title'] not in seen_titles:
                seen_titles.add(item['title'])
                unique_news.append(item)

        return unique_news[:10]  # Return 10 latest unique news items
    except Exception as e:
        print(f"Error fetching market news: {e}")
        return []
```


```python
def get_recommendations(ticker):
    """
    Get analyst recommendations for a stock

    Args:
        ticker (str): Stock ticker symbol

    Returns:
        pandas.DataFrame: Analyst recommendations
    """
    try:
        ticker_obj = yf.Ticker(ticker)
        recommendations = ticker_obj.recommendations
        return recommendations
    except Exception as e:
        print(f"Error fetching recommendations for {ticker}: {e}")
        return pd.DataFrame()
```


```python
def get_economic_indicators():
    """
    Get basic economic indicators

    Returns:
        dict: Economic indicators
    """
    indicators = {}

    try:
        # Get major indices data
        indices = {
            'S&P 500': '^GSPC',
            'Dow Jones': '^DJI',
            'NASDAQ': '^IXIC',
            'Russell 2000': '^RUT',
            'VIX (Volatility Index)': '^VIX',
            'US 10-Year Treasury': '^TNX',
            'Gold': 'GC=F',
            'Crude Oil': 'CL=F'
        }

        for name, ticker in indices.items():
            try:
                data = yf.Ticker(ticker).history(period="5d")
                if not data.empty:
                    latest_close = data['Close'].iloc[-1]
                    prev_close = data['Close'].iloc[-2] if len(data) > 1 else latest_close
                    change_pct = ((latest_close - prev_close) / prev_close) * 100

                    indicators[name] = {
                        'value': latest_close,
                        'change_pct': change_pct
                    }
            except Exception as e:
                print(f"Error fetching {name} data: {e}")

        return indicators
    except Exception as e:
        print(f"Error fetching economic indicators: {e}")
        return {}
```


```python
def get_peer_comparison(ticker):
    """
    Get peer companies for comparison

    Args:
        ticker (str): Stock ticker symbol

    Returns:
        list: List of peer company tickers
    """
    try:
        # Get sector/industry of the ticker
        info = get_company_info(ticker)

        if not info or 'sector' not in info:
            return []

        sector = info.get('sector', '')
        industry = info.get('industry', '')

        # For simplicity, we'll use predefined common peers for major sectors
        sector_peers = {
            'Technology': ['AAPL', 'MSFT', 'GOOGL', 'META', 'AMZN'],
            'Healthcare': ['JNJ', 'PFE', 'UNH', 'MRK', 'ABT'],
            'Financial Services': ['JPM', 'BAC', 'WFC', 'C', 'GS'],
            'Consumer Cyclical': ['AMZN', 'HD', 'NKE', 'SBUX', 'MCD'],
            'Communication Services': ['GOOGL', 'META', 'VZ', 'CMCSA', 'DIS'],
            'Energy': ['XOM', 'CVX', 'COP', 'SLB', 'EOG'],
            'Industrials': ['HON', 'UNP', 'BA', 'CAT', 'MMM'],
            'Basic Materials': ['LIN', 'APD', 'SHW', 'ECL', 'NEM'],
            'Consumer Defensive': ['PG', 'KO', 'PEP', 'WMT', 'COST'],
            'Utilities': ['NEE', 'DUK', 'SO', 'D', 'AEP'],
            'Real Estate': ['AMT', 'PLD', 'CCI', 'EQIX', 'PSA']
        }

        # Check if we have predefined peers for this sector
        peers = sector_peers.get(sector, [])

        # Remove the ticker itself from the list of peers
        ticker_upper = ticker.upper()
        peers = [peer for peer in peers if peer != ticker_upper]

        # If we have too few peers, try to add some based on the industry
        if len(peers) < 3:
            # This would be an area to improve with a more sophisticated approach
            # For now, we'll return what we have
            pass

        return peers[:5]  # Return at most 5 peers
    except Exception as e:
        print(f"Error finding peers for {ticker}: {e}")
        return []


```


```python
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
```


```python
def create_price_chart(df, title="Stock Price Chart", moving_averages=True):
    """
    Creates an interactive price chart with Plotly

    Args:
        df (pandas.DataFrame): Stock data with OHLC prices
        title (str): Chart title
        moving_averages (bool): Whether to include moving averages

    Returns:
        plotly.graph_objects.Figure: Interactive price chart
    """
    fig = go.Figure()

    # Add candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=df['Date'],
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            name="OHLC"
        )
    )

    # Add moving averages if requested and available
    if moving_averages:
        if 'SMA20' in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df['Date'],
                    y=df['SMA20'],
                    line=dict(color='blue', width=1),
                    name="20-day MA"
                )
            )

        if 'SMA50' in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df['Date'],
                    y=df['SMA50'],
                    line=dict(color='orange', width=1),
                    name="50-day MA"
                )
            )

        if 'SMA200' in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df['Date'],
                    y=df['SMA200'],
                    line=dict(color='red', width=1),
                    name="200-day MA"
                )
            )

    # Update layout
    fig.update_layout(
        title=title,
        yaxis_title='Price',
        xaxis_title='Date',
        xaxis_rangeslider_visible=False,
        height=500,
        margin=dict(l=50, r=50, t=50, b=50)
    )

    return fig

def create_volume_chart(df, title="Trading Volume"):
    """
    Creates a volume chart with Plotly

    Args:
        df (pandas.DataFrame): Stock data with Volume
        title (str): Chart title

    Returns:
        plotly.graph_objects.Figure: Volume chart
    """
    fig = go.Figure()

    # Add volume bars
    colors = ['red' if row['Open'] > row['Close'] else 'green' for _, row in df.iterrows()]

    fig.add_trace(
        go.Bar(
            x=df['Date'],
            y=df['Volume'],
            marker_color=colors,
            name="Volume"
        )
    )
     # Update layout
    fig.update_layout(
        title=title,
        yaxis_title='Volume',
        xaxis_title='Date',
        height=300,
        margin=dict(l=50, r=50, t=50, b=50)
    )

    return fig


```


```python

def create_price_volume_chart(df, title="Price and Volume"):
    """
    Create a combined price and volume chart

    Args:
        df (pandas.DataFrame): Stock data with OHLC and Volume
        title (str): Chart title

    Returns:
        plotly.graph_objects.Figure: Combined price and volume chart
    """
    # Create figure with secondary y-axis
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        vertical_spacing=0.03,
                        row_heights=[0.7, 0.3],
                        subplot_titles=("Price", "Volume"))
        # Add candlestick trace
    fig.add_trace(
        go.Candlestick(
            x=df['Date'],
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            name="OHLC"
        ),
        row=1, col=1
    )

    # Add moving averages
    if 'SMA20' in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df['Date'],
                y=df['SMA20'],
                line=dict(color='blue', width=1),
                name="20-day MA"
            ),
            row=1, col=1
        )

    if 'SMA50' in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df['Date'],
                y=df['SMA50'],
                line=dict(color='orange', width=1),
                name="50-day MA"
            ),
            row=1, col=1
        )

    if 'SMA200' in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df['Date'],
                y=df['SMA200'],
                line=dict(color='red', width=1),
                name="200-day MA"
            ),
            row=1, col=1
                 )

    # Add volume bars
    colors = ['red' if row['Open'] > row['Close'] else 'green' for _, row in df.iterrows()]

    fig.add_trace(
        go.Bar(
            x=df['Date'],
            y=df['Volume'],
            marker_color=colors,
            name="Volume"
        ),
        row=2, col=1
    )

    # Update layout
    fig.update_layout(
        title=title,
        height=700,
        margin=dict(l=50, r=50, t=50, b=50),
        xaxis_rangeslider_visible=False
    )
    return fig

```


```python
import plotly.graph_objects as go

def create_returns_chart(df, title="Stock Returns"):
    """
    Creates a bar chart of daily stock returns.

    Args:
        df (pandas.DataFrame): Stock data with 'Date' and 'Close' columns.
        title (str): Title for the chart.

    Returns:
        plotly.graph_objects.Figure: The generated chart.
    """
    returns_df = df.copy()
    returns_df['Daily_Return'] = df['Close'].pct_change() * 100
    returns_df = returns_df.dropna()
    fig = go.Figure()

    # Add returns bars
    colors = ['red' if ret < 0 else 'green' for ret in returns_df['Daily_Return']]

    fig.add_trace(
        go.Bar(
            x=returns_df['Date'],
            y=returns_df['Daily_Return'],
            marker_color=colors,
            name="Daily Returns"
        )
    )

    # Update layout
    fig.update_layout(
        title=title,
        yaxis_title='Returns (%)',
        xaxis_title='Date',
        height=400,
        margin=dict(l=50, r=50, t=50, b=50)
    )

    return fig

```


```python
def create_comparison_chart(stocks_data, title="Stock Price Comparison"):
    """
    Creates a comparison chart of multiple stocks

    Args:
        stocks_data (dict): Dictionary with ticker symbols as keys and dataframes as values
        title (str): Chart title

    Returns:
        plotly.graph_objects.Figure: Comparison chart
    """
    fig = go.Figure()

    for ticker, df in stocks_data.items():
        if not df.empty:
            fig.add_trace(
                go.Scatter(
                    x=df['Date'],
                    y=df['Close'],
                    mode='lines',
                    name=ticker
                )
            )

    # Update layout
    fig.update_layout(
        title=title,
        yaxis_title='Price',
        xaxis_title='Date',
        height=500,
        margin=dict(l=50, r=50, t=50, b=50)
    )

    return fig

def create_normalized_comparison_chart(stocks_data, title="Normalized Price Comparison (%)"):
    fig = go.Figure()

    for ticker, df in stocks_data.items():
        if not df.empty:
            # Normalize to percentage change from first value
            first_value = df['Close'].iloc[0]
            normalized = ((df['Close'] / first_value) - 1) * 100

            fig.add_trace(
                go.Scatter(
                    x=df['Date'],
                    y=normalized,
                    mode='lines',
                    name=ticker
                )
            )

    # Update layout
    fig.update_layout(
        title=title,
        yaxis_title='Change (%)',
        xaxis_title='Date',
        height=500,
        margin=dict(l=50, r=50, t=50, b=50)
    )

    return fig


```


```python
def create_financial_metric_comparison(metrics_data, title="Financial Metrics Comparison"):
    """
    Creates a bar chart comparing financial metrics across companies

    Args:
        metrics_data (dict): Dictionary with companies as keys and metric values as values
        title (str): Chart title

    Returns:
        plotly.graph_objects.Figure: Bar chart
    """
    # Extract data for plotting
    companies = list(metrics_data.keys())
    metric_names = list(metrics_data[companies[0]].keys()) if companies else []

    # Create a figure for each metric
    figs = {}

    for metric in metric_names:
        values = [metrics_data[company].get(metric, 0) for company in companies]

        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                x=companies,
                y=values,
                text=values,
                textposition='auto',
                name=metric
            )
        )

        fig.update_layout(
            title=f"{title} - {metric}",
            yaxis_title=metric,
            xaxis_title='Company',
            height=400,
            margin=dict(l=50, r=50, t=50, b=50)
        )

        figs[metric] = fig

    return figs

```


```python
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
import base64
from io import StringIO
```


```python
from utils.data_fetcher import (
    get_stock_data, 
    get_company_info, 
    validate_ticker, 
    get_financial_data,
    get_financial_ratios
)
```


```python
from utils.chart_creator import (
    create_price_chart, 
    create_volume_chart, 
    create_price_volume_chart,
    create_returns_chart
)
from utils.financial_calculations import (
    calculate_returns, 
    calculate_volatility, 
    calculate_risk_metrics,
    format_large_number
)
```


```python
import importlib
import utils.chart_creator
import utils.financial_calculations

importlib.reload(utils.chart_creator)
importlib.reload(utils.financial_calculations)

```




    <module 'utils.financial_calculations' from '/Users/amakki/Documents/Coding/Replit/Maks-Stocks-App/utils/financial_calculations.py'>




```python
def download_link(df, filename, text):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">{text}</a>'
    return href
```


```python
def display_stock_input():
    st.sidebar.header("stock selection")
    ticker = st.sidebar.text_input("enter stock symbol (e.g., AAPL, MSFT)", "AAPL")
    ticker = ticker.upper().strip()
    if ticker:
        if validate_ticker(ticker):
            return ticker
        else:
            st.sidebar.error(f"Invalid stock symbol: {ticker}")
            return None
    return None
```


```python
def display_time_period_selector():
    """
    Display time period selector
    
    Returns:
        tuple: (period, interval)
    """
    st.sidebar.header("Time Period")
    
    period_options = {
        "1 Day": "1d",
        "5 Days": "5d",
        "1 Month": "1mo",
        "3 Months": "3mo",
        "6 Months": "6mo",
        "Year to Date": "ytd",
        "1 Year": "1y",
        "2 Years": "2y",
        "5 Years": "5y",
        "10 Years": "10y",
        "Max": "max"
    }
    
    interval_options = {
        "1 Day": "1d",
        "1 Week": "1wk",
        "1 Month": "1mo"
    }
    
    # For shorter periods, offer more granular intervals
    short_interval_options = {
        "1 Minute": "1m",
        "5 Minutes": "5m",
        "15 Minutes": "15m",
        "30 Minutes": "30m",
        "60 Minutes": "60m",
        "1 Day": "1d"
    }
    selected_period_name = st.sidebar.selectbox(
        "Select Time Period",
        list(period_options.keys()),
        index=6  # Default to 1 Year
    )
    
    period = period_options[selected_period_name]
    
    # Adjust available intervals based on selected period
    if period in ["1d", "5d"]:
        interval_name = st.sidebar.selectbox(
            "Select Interval",
            list(short_interval_options.keys()),
            index=5  # Default to 1 Day
        )
        interval = short_interval_options[interval_name]
    else:
        interval_name = st.sidebar.selectbox(
            "Select Interval",
            list(interval_options.keys()),
            index=0  # Default to 1 Day
        )
        interval = interval_options[interval_name]
    
    return (period, interval)
```


```python
def display_stock_overview(ticker, company_info):
    """
    Display stock overview section
    
    Args:
        ticker (str): Stock ticker symbol
        company_info (dict): Company information
    """
    st.header("Stock Overview")
    
    if not company_info:
        st.error("Failed to retrieve company information.")
        return
    
    # Layout with columns
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        company_name = company_info.get('longName', 'N/A')
        st.subheader(f"{company_name} ({ticker})")
        
        sector = company_info.get('sector', 'N/A')
        industry = company_info.get('industry', 'N/A')
        
        st.write(f"**Sector:** {sector}")
        st.write(f"**Industry:** {industry}")
        
        website = company_info.get('website', '')
        if website:
            st.write(f"**Website:** [{website}]({website})")
    
    with col2:
        current_price = company_info.get('currentPrice', 'N/A')
        previous_close = company_info.get('previousClose', 'N/A')
        
        if current_price != 'N/A' and previous_close != 'N/A':
            price_change = current_price - previous_close
            price_change_percent = (price_change / previous_close) * 100
            
            price_color = "green" if price_change >= 0 else "red"
            change_symbol = "+" if price_change >= 0 else ""
            st.metric(
                "Current Price",
                f"${current_price:.2f}",
                f"{change_symbol}{price_change:.2f} ({change_symbol}{price_change_percent:.2f}%)",
                delta_color=price_color
            )
        else:
            st.metric("Current Price", "N/A")
        
        # Day range
        day_low = company_info.get('dayLow', 'N/A')
        day_high = company_info.get('dayHigh', 'N/A')
        
        if day_low != 'N/A' and day_high != 'N/A':
            st.write(f"**Day Range:** ${day_low:.2f} - ${day_high:.2f}")
    
    with col3:
        # 52-week range
        year_low = company_info.get('fiftyTwoWeekLow', 'N/A')
        year_high = company_info.get('fiftyTwoWeekHigh', 'N/A')
        
        if year_low != 'N/A' and year_high != 'N/A':
            st.write(f"**52-Week Range:** ${year_low:.2f} - ${year_high:.2f}")
        
        # Market cap
        market_cap = company_info.get('marketCap', 'N/A')
        if market_cap != 'N/A':
            st.write(f"**Market Cap:** {format_large_number(market_cap)}")
        volume = company_info.get('volume', 'N/A')
        avg_volume = company_info.get('averageVolume', 'N/A')
        
        if volume != 'N/A':
            st.write(f"**Volume:** {format_large_number(volume)}")
        
        if avg_volume != 'N/A':
            st.write(f"**Avg. Volume:** {format_large_number(avg_volume)}")
```


```python
def display_stock_price_charts(ticker, df, period, interval):
    """
    Display stock price charts
    
    Args:
        ticker (str): Stock ticker symbol
        df (pandas.DataFrame): Stock data
        period (str): Time period
        interval (str): Data interval
    """
    st.header("Price Charts")
    
    if df.empty:
        st.error("No price data available for the selected period.")
        return
    
    # Chart type selector
    chart_type = st.radio(
        "Select Chart Type",
        ["Candlestick", "Price & Volume", "Line", "Returns"],
        horizontal=True
    )
    
    if chart_type == "Candlestick":
        fig = create_price_chart(df, title=f"{ticker} Stock Price")
        st.plotly_chart(fig, use_container_width=True)
        
        # Volume chart below
        volume_fig = create_volume_chart(df, title=f"{ticker} Trading Volume")
        st.plotly_chart(volume_fig, use_container_width=True)
    
    elif chart_type == "Price & Volume":
        fig = create_price_volume_chart(df, title=f"{ticker} Price and Volume")
        st.plotly_chart(fig, use_container_width=True)
    elif chart_type == "Line":
        # Create a simple line chart
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=df['Date'],
                y=df['Close'],
                mode='lines',
                name="Close Price"
            )
        )
        
        # Add moving averages
        if 'SMA20' in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df['Date'],
                    y=df['SMA20'],
                    line=dict(color='blue', width=1, dash='dash'),
                    name="20-day MA"
                )
            )
        if 'SMA50' in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df['Date'],
                    y=df['SMA50'],
                    line=dict(color='orange', width=1, dash='dash'),
                    name="50-day MA"
                )
            )
        
        if 'SMA200' in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df['Date'],
                    y=df['SMA200'],
                    line=dict(color='red', width=1, dash='dash'),
                    name="200-day MA"
                )
            )
        
        fig.update_layout(
            title=f"{ticker} Stock Price (Line)",
            yaxis_title='Price',
            xaxis_title='Date',
            height=500,
            margin=dict(l=50, r=50, t=50, b=50)
        )
        
        st.plotly_chart(fig, use_container_width=True)
    elif chart_type == "Returns":
        returns_fig = create_returns_chart(df, title=f"{ticker} Daily Returns")
        st.plotly_chart(returns_fig, use_container_width=True)
    
    # Data table with download option
    with st.expander("View and Download Price Data"):
        st.dataframe(df)
        
        # Generate download link
        st.markdown(
            download_link(df, f"{ticker}_stock_data.csv", "Download CSV"),
            unsafe_allow_html=True
        )
```


```python
def display_key_metrics(ticker, df, company_info):
    """
    Display key financial metrics
    
    Args:
        ticker (str): Stock ticker symbol
        df (pandas.DataFrame): Stock price data
        company_info (dict): Company information
    """
    st.header("Key Metrics")
    
    if df.empty or not company_info:
        st.error("Insufficient data to calculate metrics.")
        return
    
    # Calculate metrics
    returns = calculate_returns(df)
    
    # Format metrics for display
    formatted_returns = {k: f"{v:.2f}%" for k, v in returns.items()}
    
    # Get financial ratios
    ratios = get_financial_ratios(ticker)
    
    # Create columns for different metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("Returns")
        metrics_data = []
        
        for period, value in formatted_returns.items():
            period_name = {
                '1d': 'Daily',
                '1w': 'Weekly',
                '1m': 'Monthly',
                '3m': '3-Month',
                '6m': '6-Month',
                '1y': 'Annual',
                'ytd': 'Year-to-Date'
            }.get(period, period)
            metrics_data.append({'Period': period_name, 'Return': value})
        
        if metrics_data:
            st.table(pd.DataFrame(metrics_data))
    
    with col2:
        st.subheader("Valuation")
        valuation_data = []
        
        # P/E ratio
        pe_ratio = ratios.get('P/E Ratio', company_info.get('trailingPE', 'N/A'))
        if pe_ratio != 'N/A' and pe_ratio is not None:
            valuation_data.append({'Metric': 'P/E Ratio', 'Value': f"{pe_ratio:.2f}"})
        
        # Price to Book
        pb_ratio = ratios.get('Price to Book', company_info.get('priceToBook', 'N/A'))
        if pb_ratio != 'N/A' and pb_ratio is not None:
            valuation_data.append({'Metric': 'Price/Book', 'Value': f"{pb_ratio:.2f}"})
        
        # EV/EBITDA
        ev_ebitda = ratios.get('EV/EBITDA', company_info.get('enterpriseToEbitda', 'N/A'))
        if ev_ebitda != 'N/A' and ev_ebitda is not None:
            valuation_data.append({'Metric': 'EV/EBITDA', 'Value': f"{ev_ebitda:.2f}"})
        
        # Dividend yield
        div_yield = ratios.get('Dividend Yield', company_info.get('dividendYield', 'N/A'))
        if div_yield != 'N/A' and div_yield is not None:
            valuation_data.append({'Metric': 'Dividend Yield', 'Value': f"{div_yield*100:.2f}%"})
        
        # Forward P/E
        forward_pe = company_info.get('forwardPE', 'N/A')
        if forward_pe != 'N/A' and forward_pe is not None:
            valuation_data.append({'Metric': 'Forward P/E', 'Value': f"{forward_pe:.2f}"})
        
        if valuation_data:
            st.table(pd.DataFrame(valuation_data))
    with col3:
        st.subheader("Risk Metrics")
        
        risk_metrics = calculate_risk_metrics(df)
        risk_data = []
        
        if 'volatility' in risk_metrics:
            risk_data.append({'Metric': 'Volatility (Ann.)', 'Value': f"{risk_metrics['volatility']:.2f}%"})
        
        if 'sharpe_ratio' in risk_metrics:
            risk_data.append({'Metric': 'Sharpe Ratio', 'Value': f"{risk_metrics['sharpe_ratio']:.2f}"})
        
        if 'max_drawdown' in risk_metrics:
            risk_data.append({'Metric': 'Max Drawdown', 'Value': f"{risk_metrics['max_drawdown']:.2f}%"})
        
        if 'downside_deviation' in risk_metrics:
            risk_data.append({'Metric': 'Downside Dev.', 'Value': f"{risk_metrics['downside_deviation']:.2f}%"})
        
        if 'sortino_ratio' in risk_metrics:
            risk_data.append({'Metric': 'Sortino Ratio', 'Value': f"{risk_metrics['sortino_ratio']:.2f}"})
        
        beta = company_info.get('beta', 'N/A')
        if beta != 'N/A' and beta is not None:
            risk_data.append({'Metric': 'Beta', 'Value': f"{beta:.2f}"})
        
        if risk_data:
            st.table(pd.DataFrame(risk_data))

```


```python
def display_company_info(company_info):
    """
    Display detailed company information
    
    Args:
        company_info (dict): Company information
    """
    st.header("Company Information")
    
    if not company_info:
        st.error("Company information not available.")
        return
    
    # Company description
    with st.expander("Business Description", expanded=True):
        long_business_summary = company_info.get('longBusinessSummary', 'No description available.')
        st.write(long_business_summary)
    
    # Key statistics
    with st.expander("Key Statistics"):
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Financial Stats")
            
            # Revenue and earnings
            revenue = company_info.get('totalRevenue', 'N/A')
            if revenue != 'N/A' and revenue is not None:
                st.write(f"**Revenue:** {format_large_number(revenue)}")
            
            gross_profit = company_info.get('grossProfits', 'N/A')
            if gross_profit != 'N/A' and gross_profit is not None:
                st.write(f"**Gross Profit:** {format_large_number(gross_profit)}")
            
            net_income = company_info.get('netIncomeToCommon', 'N/A')
            if net_income != 'N/A' and net_income is not None:
                st.write(f"**Net Income:** {format_large_number(net_income)}")
            
            eps = company_info.get('trailingEPS', 'N/A')
            if eps != 'N/A' and eps is not None:
                st.write(f"**EPS (TTM):** ${eps:.2f}")
            
            # Profitability
            profit_margin = company_info.get('profitMargins', 'N/A')
            if profit_margin != 'N/A' and profit_margin is not None:
                st.write(f"**Profit Margin:** {profit_margin*100:.2f}%")
            
            operating_margin = company_info.get('operatingMargins', 'N/A')
            if operating_margin != 'N/A' and operating_margin is not None:
                st.write(f"**Operating Margin:** {operating_margin*100:.2f}%")
        with col2:
            st.subheader("Trading Stats")
            
            # Trading info
            shares_outstanding = company_info.get('sharesOutstanding', 'N/A')
            if shares_outstanding != 'N/A' and shares_outstanding is not None:
                st.write(f"**Shares Outstanding:** {format_large_number(shares_outstanding)}")
            
            float_shares = company_info.get('floatShares', 'N/A')
            if float_shares != 'N/A' and float_shares is not None:
                st.write(f"**Float:** {format_large_number(float_shares)}")
            
            short_ratio = company_info.get('shortRatio', 'N/A')
            if short_ratio != 'N/A' and short_ratio is not None:
                st.write(f"**Short Ratio:** {short_ratio:.2f}")
            
            # Dividends
            dividend_rate = company_info.get('dividendRate', 'N/A')
            if dividend_rate != 'N/A' and dividend_rate is not None:
                st.write(f"**Dividend Rate:** ${dividend_rate:.2f}")
            
            dividend_yield = company_info.get('dividendYield', 'N/A')
            if dividend_yield != 'N/A' and dividend_yield is not None:
                st.write(f"**Dividend Yield:** {dividend_yield*100:.2f}%")
            
            payout_ratio = company_info.get('payoutRatio', 'N/A')
            if payout_ratio != 'N/A' and payout_ratio is not None:
                st.write(f"**Payout Ratio:** {payout_ratio*100:.2f}%")

    # Institutional ownership and insider transactions
    with st.expander("Ownership & Insiders"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Institutional Ownership")
            
            held_percent_institutions = company_info.get('heldPercentInstitutions', 'N/A')
            if held_percent_institutions != 'N/A' and held_percent_institutions is not None:
                st.write(f"**Institutional Ownership:** {held_percent_institutions*100:.2f}%")
            
            held_percent_insiders = company_info.get('heldPercentInsiders', 'N/A')
            if held_percent_insiders != 'N/A' and held_percent_insiders is not None:
                st.write(f"**Insider Ownership:** {held_percent_insiders*100:.2f}%")
        
        with col2:
            st.subheader("Recent Activity")
            
            net_insider_buying = company_info.get('netInfobuyerShares', 'N/A')
            if net_insider_buying != 'N/A' and net_insider_buying is not None:
                buy_text = "Buying" if net_insider_buying > 0 else "Selling"
                st.write(f"**Insider Activity:** {buy_text}")
            
            # Analyst recommendations
            target_mean_price = company_info.get('targetMeanPrice', 'N/A')
            current_price = company_info.get('currentPrice', 'N/A')
            
            if target_mean_price != 'N/A' and current_price != 'N/A' and target_mean_price is not None and current_price is not None:
                upside = ((target_mean_price / current_price) - 1) * 100
                st.write(f"**Target Price:** ${target_mean_price:.2f} ({upside:.2f}% upside)")
            
            recommend_mean = company_info.get('recommendationMean', 'N/A')
            if recommend_mean != 'N/A' and recommend_mean is not None:
                st.write(f"**Analyst Rating:** {recommend_mean:.1f}/5.0")

```


```python
def display_financial_data(ticker):
    """
    Display financial statements data
    
    Args:
        ticker (str): Stock ticker symbol
    """
    st.header("Financial Statements")
    
    # Fetch financial data
    income_stmt, balance_sheet, cash_flow = get_financial_data(ticker)
    
    if income_stmt.empty and balance_sheet.empty and cash_flow.empty:
        st.error("Financial statements data not available.")
        return
    
    # Create tabs for different statements
    tabs = st.tabs(["Income Statement", "Balance Sheet", "Cash Flow"])
    
    with tabs[0]:
        if not income_stmt.empty:
            # Format the dataframe for display
            income_stmt = income_stmt.fillna("N/A")
            
            # Convert large numbers to millions
            for col in income_stmt.columns:
                income_stmt[col] = income_stmt[col].apply(
                    lambda x: format_large_number(x) if not isinstance(x, str) else x
                )
            
            st.dataframe(income_stmt)
            
            # Download link
            st.markdown(
                download_link(income_stmt, f"{ticker}_income_statement.csv", "Download Income Statement CSV"),
                unsafe_allow_html=True
            )
        else:
            st.write("Income statement data not available.")
    with tabs[1]:
        if not balance_sheet.empty:
            # Format the dataframe for display
            balance_sheet = balance_sheet.fillna("N/A")
            
            # Convert large numbers to millions
            for col in balance_sheet.columns:
                balance_sheet[col] = balance_sheet[col].apply(
                    lambda x: format_large_number(x) if not isinstance(x, str) else x
                )
            
            st.dataframe(balance_sheet)
            
            # Download link
            st.markdown(
                download_link(balance_sheet, f"{ticker}_balance_sheet.csv", "Download Balance Sheet CSV"),
                unsafe_allow_html=True
            )
        else:
            st.write("Balance sheet data not available.")
    
    with tabs[2]:
        if not cash_flow.empty:
            # Format the dataframe for display
            cash_flow = cash_flow.fillna("N/A")
            
            # Convert large numbers to millions
            for col in cash_flow.columns:
                cash_flow[col] = cash_flow[col].apply(
                    lambda x: format_large_number(x) if not isinstance(x, str) else x
                )
            
            st.dataframe(cash_flow)
            
            # Download link
            st.markdown(
                download_link(cash_flow, f"{ticker}_cash_flow.csv", "Download Cash Flow CSV"),
                unsafe_allow_html=True
            )
        else:
            st.write("Cash flow data not available.")
```


```python
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from datetime import datetime, timedelta
import base64
from io import StringIO
```


```python
from utils.data_fetcher import (
    get_stock_data,
    get_company_info,
    validate_ticker,
    get_peer_comparison,
    get_financial_ratios
)
from utils.chart_creator import (
    create_comparison_chart,
    create_normalized_comparison_chart,
    create_financial_metric_comparison
)
from utils.financial_calculations import (
    calculate_returns,
    calculate_risk_metrics,
    format_large_number
)

```


```python
def download_link(df, filename, text):
    """
    Generate a download link for a dataframe
    
    Args:
        df (pandas.DataFrame): Dataframe to download
        filename (str): Filename for the downloaded file
        text (str): Text to display for the download link
        
    Returns:
        str: HTML download link
    """
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">{text}</a>'
    return href

```


```python
def display_comparison_input(current_ticker=None):
    """
    Display input for comparison stocks
    
    Args:
        current_ticker (str): Current main ticker being analyzed
    
    Returns:
        list: List of valid ticker symbols to compare
    """
    st.subheader("Compare Stocks")
    
    # Initialize comparison tickers
    if 'comparison_tickers' not in st.session_state:
        # If we have a current ticker, suggest peers for comparison
        if current_ticker:
            peers = get_peer_comparison(current_ticker)
            # Start with up to 3 peers
            st.session_state.comparison_tickers = peers[:min(3, len(peers))]
        else:
            st.session_state.comparison_tickers = []
    
    # Get current comparison tickers
    comparison_tickers = st.session_state.comparison_tickers.copy()
    
    # Input for manual ticker entry
    col1, col2 = st.columns([3, 1])
    with col1:
        new_ticker = st.text_input("Add ticker for comparison").upper().strip()
    
    with col2:
        add_clicked = st.button("Add")
    
    if add_clicked and new_ticker:
        if validate_ticker(new_ticker):
            if new_ticker not in comparison_tickers and new_ticker != current_ticker:
                comparison_tickers.append(new_ticker)
                st.session_state.comparison_tickers = comparison_tickers
                st.success(f"Added {new_ticker} for comparison")
            else:
                st.warning(f"{new_ticker} is already in the comparison list or is the main ticker")
        else:
            st.error(f"Invalid ticker: {new_ticker}")
    
    # Display current comparison list with remove buttons
    if comparison_tickers:
        st.write("Current comparison list:")
        cols = st.columns(len(comparison_tickers))
        to_remove = None
        
        for i, ticker in enumerate(comparison_tickers):
            with cols[i]:
                st.write(ticker)
                if st.button(f"Remove", key=f"remove_{ticker}"):
                    to_remove = i
        
        if to_remove is not None:
            comparison_tickers.pop(to_remove)
            st.session_state.comparison_tickers = comparison_tickers
            st.rerun()
    
    return comparison_tickers
```


```python
def display_price_comparison(main_ticker, comparison_tickers, period="1y", interval="1d"):
    """
    Display price comparison charts
    
    Args:
        main_ticker (str): Main ticker symbol
        comparison_tickers (list): List of comparison ticker symbols
        period (str): Time period
        interval (str): Data interval
    """
    if not comparison_tickers:
        st.info("Add tickers above to compare prices")
        return
    
    st.subheader("Price Comparison")
    
    # Fetch data for all tickers
    stocks_data = {}
    # First, fetch the main ticker data
    main_df = get_stock_data(main_ticker, period=period, interval=interval)
    if not main_df.empty:
        stocks_data[main_ticker] = main_df
    
    # Then fetch comparison tickers
    for ticker in comparison_tickers:
        df = get_stock_data(ticker, period=period, interval=interval)
        if not df.empty:
            stocks_data[ticker] = df
    
    if not stocks_data:
        st.error("Failed to retrieve price data for comparison.")
        return
    
    # Create charts
    chart_type = st.radio(
        "Chart Type",
        ["Absolute Prices", "Normalized (%)"],
        horizontal=True
    )
    
    if chart_type == "Absolute Prices":
        fig = create_comparison_chart(stocks_data, title="Stock Price Comparison")
        st.plotly_chart(fig, use_container_width=True)
    else:
        fig = create_normalized_comparison_chart(stocks_data, title="Normalized Price Comparison (%)")
        st.plotly_chart(fig, use_container_width=True)
    
    # Display data table for download
    with st.expander("View and Download Comparison Data"):
        # Create a combined dataframe with date and closing prices for each stock
        combined_df = pd.DataFrame()
        
        for ticker, df in stocks_data.items():
            if combined_df.empty:
                combined_df['Date'] = df['Date']
            
            combined_df[f"{ticker} Close"] = df['Close']
        
        if not combined_df.empty:
            st.dataframe(combined_df)
            
            # Generate download link
            tickers_text = "_".join([main_ticker] + comparison_tickers)
            st.markdown(
                download_link(combined_df, f"{tickers_text}_comparison.csv", "Download Comparison Data (CSV)"),
                unsafe_allow_html=True
            )
```


```python
def display_financial_comparison(main_ticker, comparison_tickers):
    """
    Display financial metric comparison
    
    Args:
        main_ticker (str): Main ticker symbol
        comparison_tickers (list): List of comparison ticker symbols
    """
    if not comparison_tickers:
        st.info("Add tickers above to compare financial metrics")
        return
    
    st.subheader("Financial Metrics Comparison")
    
    # Get financial metrics for all tickers
    tickers = [main_ticker] + comparison_tickers
    metrics_data = {}
    
    for ticker in tickers:
        ratios = get_financial_ratios(ticker)
        if ratios:
            metrics_data[ticker] = ratios
    
    if not metrics_data:
        st.error("Failed to retrieve financial metrics for comparison.")
        return
    
    # Group metrics by category for display
    metric_categories = {
        "Valuation": ["P/E Ratio", "Price to Book", "EV/EBITDA", "EV/Revenue"],
        "Profitability": ["ROE", "ROA", "Profit Margin"],
        "Dividends": ["Dividend Yield", "Payout Ratio"],
        "Growth": ["Revenue Growth", "Earnings Growth"]
    }
    # Create financial comparison tables
    category_tabs = st.tabs(list(metric_categories.keys()))
    
    for i, (category, tab) in enumerate(zip(metric_categories.keys(), category_tabs)):
        with tab:
            metrics = metric_categories[category]
            
            # Create a dataframe for this category
            category_data = []
            
            for metric in metrics:
                row = {"Metric": metric}
                
                for ticker in tickers:
                    if ticker in metrics_data and metric in metrics_data[ticker]:
                        value = metrics_data[ticker][metric]
                        
                        # Format value based on metric type
                        if metric in ["Dividend Yield", "Payout Ratio", "ROE", "ROA", "Profit Margin", 
                                     "Revenue Growth", "Earnings Growth"]:
                            row[ticker] = f"{value*100:.2f}%" if value is not None else "N/A"
                        else:
                            row[ticker] = f"{value:.2f}" if value is not None else "N/A"
                    else:
                        row[ticker] = "N/A"
                
                category_data.append(row)
            
            if category_data:
                df = pd.DataFrame(category_data)
                st.dataframe(df, use_container_width=True)
                
                # Generate visualization for the metrics
                for metric in metrics:
                    metric_row = next((r for r in category_data if r["Metric"] == metric), None)
                    if metric_row:
                        values = {}
                        
                        for ticker in tickers:
                            if ticker in metric_row and metric_row[ticker] != "N/A":
                                # Extract numerical value
                                try:
                                    if "%" in metric_row[ticker]:
                                        values[ticker] = float(metric_row[ticker].replace("%", ""))
                                    else:
                                        values[ticker] = float(metric_row[ticker])
                                except:
                                    pass
                        
                        if values:
                            # Create bar chart
                            fig = go.Figure()
                            
                            fig.add_trace(
                                go.Bar(
                                    x=list(values.keys()),
                                    y=list(values.values()),
                                    text=[f"{v:.2f}" for v in values.values()],
                                    textposition='auto'
                                )
                            )
                            
                            fig.update_layout(
                                title=f"{metric} Comparison",
                                height=400,
                                margin=dict(l=50, r=50, t=50, b=50)
                            )
                            st.plotly_chart(fig, use_container_width=True)
            else:
                st.info(f"No {category} metrics available for comparison")

def display_performance_comparison(main_ticker, comparison_tickers, period="1y"):
    """
    Display performance comparison metrics
    
    Args:
        main_ticker (str): Main ticker symbol
        comparison_tickers (list): List of comparison ticker symbols
        period (str): Time period
    """
    if not comparison_tickers:
        st.info("Add tickers above to compare performance")
        return
    
    st.subheader("Performance Comparison")
    
    # Get stock data and calculate returns for all tickers
    tickers = [main_ticker] + comparison_tickers
    returns_data = {}
    risk_data = {}
    
    for ticker in tickers:
        df = get_stock_data(ticker, period=period)
        if not df.empty:
            # Calculate returns
            returns = calculate_returns(df)
            if returns:
                returns_data[ticker] = returns
            
            # Calculate risk metrics
            risk_metrics = calculate_risk_metrics(df)
            if risk_metrics:
                risk_data[ticker] = risk_metrics
    
    if not returns_data:
        st.error("Failed to calculate performance metrics for comparison.")
        return
    
    # Create returns comparison table
    st.write("Returns Comparison")
    
    # Get all periods from returns data
    all_periods = set()
    for ticker_returns in returns_data.values():
        all_periods.update(ticker_returns.keys())
    period_labels = {
        '1d': 'Daily',
        '1w': 'Weekly',
        '1m': 'Monthly',
        '3m': '3-Month',
        '6m': '6-Month',
        '1y': 'Annual',
        'ytd': 'Year-to-Date'
    }
    
    # Create a dataframe for returns
    returns_rows = []
    
    for period in all_periods:
        row = {"Period": period_labels.get(period, period)}
        
        for ticker in tickers:
            if ticker in returns_data and period in returns_data[ticker]:
                row[ticker] = f"{returns_data[ticker][period]:.2f}%"
            else:
                row[ticker] = "N/A"
        
        returns_rows.append(row)
    
    if returns_rows:
        # Sort by period order
        period_order = {'1d': 0, '1w': 1, '1m': 2, '3m': 3, '6m': 4, 'ytd': 5, '1y': 6}
        returns_rows.sort(key=lambda x: period_order.get(
            next((k for k, v in period_labels.items() if v == x["Period"]), ''), 100
        ))
        
        returns_df = pd.DataFrame(returns_rows)
        st.dataframe(returns_df, use_container_width=True)
    
    # Create risk metrics comparison
    st.write("Risk Metrics Comparison")

    # Create a dataframe for risk metrics
    risk_rows = []
    
    risk_metric_labels = {
        'volatility': 'Volatility (Ann.)',
        'sharpe_ratio': 'Sharpe Ratio',
        'max_drawdown': 'Max Drawdown',
        'downside_deviation': 'Downside Dev.',
        'sortino_ratio': 'Sortino Ratio'
    }
    
    for metric, label in risk_metric_labels.items():
        row = {"Metric": label}
        
        for ticker in tickers:
            if ticker in risk_data and metric in risk_data[ticker]:
                value = risk_data[ticker][metric]
                
                if metric in ['volatility', 'max_drawdown', 'downside_deviation']:
                    row[ticker] = f"{value:.2f}%"
                else:
                    row[ticker] = f"{value:.2f}"
            else:
                row[ticker] = "N/A"
        
        risk_rows.append(row)
    
    if risk_rows:
        risk_df = pd.DataFrame(risk_rows)
        st.dataframe(risk_df, use_container_width=True)
    
    # Create visualizations for returns and risk metrics
    col1, col2 = st.columns(2)

    with col1:
        # Visualize one period of returns (e.g., 1-month)
        vis_period = '1m'
        period_label = period_labels.get(vis_period, vis_period)
        
        values = {}
        for ticker in tickers:
            if ticker in returns_data and vis_period in returns_data[ticker]:
                values[ticker] = returns_data[ticker][vis_period]
        
        if values:
            fig = go.Figure()
            
            # Sort tickers by return values
            sorted_tickers = sorted(values.keys(), key=lambda x: values[x], reverse=True)
            sorted_values = [values[t] for t in sorted_tickers]
            
            # Colors based on positive/negative returns
            colors = ['green' if v >= 0 else 'red' for v in sorted_values]
            
            fig.add_trace(
                go.Bar(
                    x=sorted_tickers,
                    y=sorted_values,
                    marker_color=colors,
                    text=[f"{v:.2f}%" for v in sorted_values],
                    textposition='auto'
                )
            )
            
            fig.update_layout(
                title=f"{period_label} Returns (%)",
                height=400,
                margin=dict(l=50, r=50, t=50, b=50)
            )
            
            st.plotly_chart(fig, use_container_width=True)
    with col2:
        # Visualize volatility
        vol_values = {}
        for ticker in tickers:
            if ticker in risk_data and 'volatility' in risk_data[ticker]:
                vol_values[ticker] = risk_data[ticker]['volatility']
        
        if vol_values:
            fig = go.Figure()
            
            # Sort tickers by volatility
            sorted_tickers = sorted(vol_values.keys(), key=lambda x: vol_values[x])
            sorted_values = [vol_values[t] for t in sorted_tickers]
            
            fig.add_trace(
                go.Bar(
                    x=sorted_tickers,
                    y=sorted_values,
                    text=[f"{v:.2f}%" for v in sorted_values],
                    textposition='auto'
                )
            )
            
            fig.update_layout(
                title="Annualized Volatility (%)",
                height=400,
                margin=dict(l=50, r=50, t=50, b=50)
            )
            
            st.plotly_chart(fig, use_container_width=True)

                    
```


```python
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta

from utils.data_fetcher import (
    get_market_news, 
    get_economic_indicators,
    get_stock_data
)
from utils.financial_calculations import (
    format_large_number
)
```


```python
def display_market_overview():
    """
    Display market overview section with major indices
    """
    st.header("Market Overview")
    
    # Get economic indicators
    indicators = get_economic_indicators()
    
    if not indicators:
        st.error("Failed to retrieve market data.")
        return
    
    # Format the indicators into columns
    cols = st.columns(4)
    i = 0
    
    # List to ensure consistent order
    ordered_indices = [
        'S&P 500', 'Dow Jones', 'NASDAQ', 'Russell 2000',
        'VIX (Volatility Index)', 'US 10-Year Treasury', 'Gold', 'Crude Oil'
    ]
    for index in ordered_indices:
        if index in indicators:
            data = indicators[index]
            value = data['value']
            change_pct = data['change_pct']
            
            col = cols[i % 4]
            with col:
                color = "green" if change_pct >= 0 else "red"
                change_str = f"+{change_pct:.2f}%" if change_pct >= 0 else f"{change_pct:.2f}%"
                
                # Format values based on index type
                if index in ['US 10-Year Treasury', 'VIX (Volatility Index)']:
                    value_str = f"{value:.2f}"
                else:
                    value_str = f"{value:.2f}" if value < 1000 else f"{value:,.2f}"
                
                st.metric(index, value_str, change_str, delta_color=color)
            
            i += 1

            

```


```python
def display_market_news():
    """
    Display market news section
    """
    st.header("Market News")
    
    news = get_market_news()
    
    if not news:
        st.error("Failed to retrieve market news.")
        return
    
    # Display news in expandable format
    for i, item in enumerate(news):
        with st.expander(f"{item['title']}", expanded=i==0):
            if 'publisher' in item:
                st.caption(f"Source: {item['publisher']}")
            # Format the date
            if 'providerPublishTime' in item:
                publish_time = datetime.fromtimestamp(item['providerPublishTime'])
                st.caption(f"Published: {publish_time.strftime('%Y-%m-%d %H:%M')}")
            
            # Display summary if available
            if 'summary' in item:
                st.write(item['summary'])
            
            # Link to full article
            if 'link' in item:
                st.markdown(f"[Read full article]({item['link']})")

```


```python
def display_sector_performance():
    """
    Display sector performance section
    """
    st.header("Sector Performance")
    
    # Major sector ETFs as proxies for sector performance
    sectors = {
        "Technology": "XLK",
        "Healthcare": "XLV",
        "Financials": "XLF",
        "Energy": "XLE",
        "Consumer Discretionary": "XLY",
        "Consumer Staples": "XLP",
        "Industrials": "XLI",
        "Materials": "XLB",
        "Utilities": "XLU",
        "Real Estate": "XLRE",
        "Communication Services": "XLC"
    }
    # Get data for each sector
    sector_data = {}
    for name, ticker in sectors.items():
        df = get_stock_data(ticker, period="5d")
        if not df.empty:
            # Calculate 1-day return
            latest = df.iloc[-1]
            prev = df.iloc[-2] if len(df) > 1 else latest
            
            if prev['Close'] > 0:
                daily_return = (latest['Close'] / prev['Close'] - 1) * 100
            else:
                daily_return = 0
            
            # Calculate 1-week return
            week_ago = df.iloc[0] if len(df) > 0 else latest
            
            if week_ago['Close'] > 0:
                weekly_return = (latest['Close'] / week_ago['Close'] - 1) * 100
            else:
                weekly_return = 0
            
            sector_data[name] = {
                'daily': daily_return,
                'weekly': weekly_return
            }
    if not sector_data:
        st.error("Failed to retrieve sector performance data.")
        return
    
    # Create visualization
    view_option = st.radio(
        "View",
        ["Daily Returns", "Weekly Returns"],
        horizontal=True
    )
    
    # Sort data by selected period
    period_key = 'daily' if view_option == "Daily Returns" else 'weekly'
    sorted_sectors = sorted(
        sector_data.items(),
        key=lambda x: x[1][period_key],
        reverse=True
    )
    # Sort data by selected period
    period_key = 'daily' if view_option == "Daily Returns" else 'weekly'
    sorted_sectors = sorted(
        sector_data.items(),
        key=lambda x: x[1][period_key],
        reverse=True
    )
    
    # Create horizontal bar chart
    fig = go.Figure()
    
    sector_names = [s[0] for s in sorted_sectors]
    returns = [s[1][period_key] for s in sorted_sectors]
    colors = ['green' if r >= 0 else 'red' for r in returns]
    
    fig.add_trace(
        go.Bar(
            y=sector_names,
            x=returns,
            orientation='h',
            marker_color=colors,
            text=[f"{r:.2f}%" for r in returns],
            textposition='auto'
        )
    )
    
    period_label = "1-Day" if period_key == 'daily' else "1-Week"
    
    fig.update_layout(
        title=f"Sector Performance ({period_label})",
        xaxis_title="Return (%)",
        height=500,
        margin=dict(l=50, r=50, t=50, b=50)
    )
    
    st.plotly_chart(fig, use_container_width=True)
```


```python
def display_economic_calendar():
    """
    Display economic calendar
    """
    st.header("Economic Calendar")
    
    # This data would typically come from an API, but for demo purposes we'll create a sample
    today = datetime.now()
    
    # Generate a few days of sample data
    dates = [today + timedelta(days=i) for i in range(-2, 5)]
    
    # Sample economic events
    events = [
        {"date": dates[0], "time": "08:30 AM", "event": "Initial Jobless Claims", "actual": "235K", "forecast": "230K", "previous": "232K", "impact": "Medium"},
        {"date": dates[0], "time": "10:00 AM", "event": "Existing Home Sales", "actual": "4.32M", "forecast": "4.30M", "previous": "4.28M", "impact": "Medium"},
        {"date": dates[1], "time": "08:30 AM", "event": "Durable Goods Orders", "actual": "0.5%", "forecast": "0.4%", "previous": "-0.2%", "impact": "High"},
        {"date": dates[1], "time": "10:00 AM", "event": "Michigan Consumer Sentiment", "actual": "69.1", "forecast": "68.5", "previous": "68.0", "impact": "High"},
        {"date": dates[2], "time": "09:45 AM", "event": "PMI Composite", "actual": "", "forecast": "51.0", "previous": "50.9", "impact": "High"},
        {"date": dates[3], "time": "08:30 AM", "event": "GDP Growth Rate", "actual": "", "forecast": "2.8%", "previous": "2.5%", "impact": "High"},
        {"date": dates[4], "time": "08:30 AM", "event": "Core PCE Price Index", "actual": "", "forecast": "0.3%", "previous": "0.2%", "impact": "High"},
        {"date": dates[4], "time": "10:00 AM", "event": "Pending Home Sales", "actual": "", "forecast": "1.0%", "previous": "-1.2%", "impact": "Medium"},
    ]
    # Convert to DataFrame
    df = pd.DataFrame(events)
    
    # Format dates
    df['date_str'] = df['date'].dt.strftime('%A, %b %d')
    
    # Group by date
    grouped = df.groupby('date_str')
    
    # Create tabs for each date
    tab_labels = list(grouped.groups.keys())
    tabs = st.tabs(tab_labels)
    
    for i, (date_str, tab) in enumerate(zip(grouped.groups.keys(), tabs)):
        with tab:
            group_data = grouped.get_group(date_str)
            
            for _, row in group_data.iterrows():
                impact_color = {
                    "High": "🔴 ",
                    "Medium": "🟠 ",
                    "Low": "🟢 "
                }.get(row['impact'], "")
                
                st.markdown(f"**{row['time']} - {impact_color}{row['event']}**")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    if row['actual']:
                        st.markdown(f"**Actual:** {row['actual']}")
                    else:
                        st.markdown("**Actual:** Not released")
                
                with col2:
                    st.markdown(f"**Forecast:** {row['forecast']}")
                with col3:
                    st.markdown(f"**Previous:** {row['previous']}")
                
                st.divider()
    
    st.caption("Note: This is sample data. In a production environment, this would use real economic calendar data from an API.")

                
 
```


```python
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from datetime import datetime, timedelta

# Import utility modules
from utils.data_fetcher import (
    get_stock_data, 
    get_company_info, 
    validate_ticker
)

# Import components
from components.stock_info import (
    display_stock_input,
    display_time_period_selector,
    display_stock_overview,
    display_stock_price_charts,
    display_key_metrics,
    display_company_info,
    display_financial_data
)
from components.market_analysis import (
    display_market_overview,
    display_market_news,
    display_sector_performance,
    display_economic_calendar
)
from components.comparison import (
    display_comparison_input,
    display_price_comparison,
    display_financial_comparison,
    display_performance_comparison
)

# Set page config
st.set_page_config(
    page_title="Financial Analysis Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)
# App title
st.title("Financial Analysis Dashboard")

# Sidebar - inputs
ticker = display_stock_input()
period, interval = display_time_period_selector()

# Main content
if ticker:
    # Fetch data
    df = get_stock_data(ticker, period=period, interval=interval)
    company_info = get_company_info(ticker)
    
    # Create tabs for different sections
    tabs = st.tabs([
        "Stock Overview", 
        "Financial Statements", 
        "Market Analysis", 
        "Stock Comparison"
    ])
    
    # Tab 1: Stock Overview
    with tabs[0]:
        display_stock_overview(ticker, company_info)
        display_stock_price_charts(ticker, df, period, interval)
        display_key_metrics(ticker, df, company_info)
        display_company_info(company_info)
    
    # Tab 2: Financial Statements
    with tabs[1]:
        display_financial_data(ticker)
    
    # Tab 3: Market Analysis
    with tabs[2]:
        display_market_overview()
        display_sector_performance()
        display_market_news()
        display_economic_calendar()
    # Tab 4: Stock Comparison
    with tabs[3]:
        comparison_tickers = display_comparison_input(current_ticker=ticker)
        display_price_comparison(ticker, comparison_tickers, period=period, interval=interval)
        display_financial_comparison(ticker, comparison_tickers)
        display_performance_comparison(ticker, comparison_tickers, period=period)
else:
    # If no ticker is selected yet, show market overview
    display_market_overview()
    display_market_news()

# Footer
st.markdown("---")
st.caption("Data provided by Yahoo Finance. This dashboard is for informational purposes only and does not constitute financial advice.")

```

  
