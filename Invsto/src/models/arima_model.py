from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from typing import Tuple
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('seaborn')

class ARIMAModel:
    def __init__(self, p: int, d: int, q: int):
        self.p = p
        self.d = d
        self.q = q
        self.model = None

    def check_stationarity(self, data: pd.Series) -> bool:
        """
        Check if the time series is stationary using ADF test
        """
        result = adfuller(data)
        return result[1] < 0.05

    def fit(self, data: pd.Series):
        """
        Fit ARIMA model to the data
        """
        self.model = ARIMA(data, order=(self.p, self.d, self.q))
        self.model_fit = self.model.fit()
        return self.model_fit

    def predict(self, steps: int) -> np.ndarray:
        """
        Make predictions using the fitted model
        """
        return self.model_fit.forecast(steps) 

def fetch_stock_data(symbol, start_date, end_date):
    """Fetch stock data from Yahoo Finance"""
    ticker = yf.Ticker(symbol)
    df = ticker.history(start=start_date, end=end_date)
    return df

def create_features(df):
    """Create technical indicators"""
    # Price-based features
    df['Returns'] = df['Close'].pct_change()
    df['Log_Returns'] = np.log(df['Close']/df['Close'].shift(1))
    
    # Moving averages
    for window in [5, 10, 20, 50]:
        df[f'MA_{window}'] = df['Close'].rolling(window=window).mean()
        df[f'MA_Volume_{window}'] = df['Volume'].rolling(window=window).mean()
    
    # Volatility
    df['Volatility'] = df['Returns'].rolling(window=20).std()
    
    # Momentum
    df['Momentum'] = df['Close'] - df['Close'].shift(5)
    
    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    return df.dropna() 

# Define stock symbols and date range
symbols = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'META']
start_date = '2020-01-01'
end_date = '2024-01-01'

# Fetch data for all stocks
stock_data = {}
for symbol in symbols:
    print(f"Fetching data for {symbol}...")
    df = fetch_stock_data(symbol, start_date, end_date)
    df = create_features(df)
    stock_data[symbol] = df
    print(f"Successfully processed {symbol}") 