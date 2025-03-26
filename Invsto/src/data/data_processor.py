import pandas as pd
import numpy as np
from typing import Dict
import logging

class StockDataProcessor:
    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean the dataset by handling missing values and outliers
        """
        # Handle missing values
        df = df.ffill()  # Forward fill
        df = df.bfill()  # Backward fill for any remaining NaNs
        
        # Remove outliers using IQR method
        Q1 = df['Close'].quantile(0.25)
        Q3 = df['Close'].quantile(0.75)
        IQR = Q3 - Q1
        df = df[~((df['Close'] < (Q1 - 1.5 * IQR)) | (df['Close'] > (Q3 + 1.5 * IQR)))]
        
        return df

    def prepare_time_series(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare data for time series analysis
        """
        # Ensure datetime index
        df.index = pd.to_datetime(df.index)
        
        # Sort by date
        df = df.sort_index()
        
        # Resample to business days and forward fill
        df = df.resample('B').ffill()
        
        return df 