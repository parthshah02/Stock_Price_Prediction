import yfinance as yf
import pandas as pd
from typing import List, Dict
import logging

class StockDataLoader:
    def __init__(self, symbols: List[str], start_date: str, end_date: str):
        self.symbols = symbols
        self.start_date = start_date
        self.end_date = end_date
        self.logger = logging.getLogger(__name__)

    def fetch_data(self) -> Dict[str, pd.DataFrame]:
        """
        Fetch historical data for multiple stocks using yfinance
        """
        stock_data = {}
        for symbol in self.symbols:
            try:
                ticker = yf.Ticker(symbol)
                df = ticker.history(start=self.start_date, end=self.end_date)
                stock_data[symbol] = df
                self.logger.info(f"Successfully fetched data for {symbol}")
            except Exception as e:
                self.logger.error(f"Error fetching data for {symbol}: {str(e)}")
        
        return stock_data