import pandas as pd
import numpy as np
from typing import List

class FeatureEngineer:
    @staticmethod
    def create_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        Create technical indicators and features for stock prediction
        """
        # Price-based features
        df['Returns'] = df['Close'].pct_change()
        df['Log_Returns'] = np.log(df['Close']/df['Close'].shift(1))
        
        # Moving averages
        for window in [5, 10, 20, 50]:
            df[f'MA_{window}'] = df['Close'].rolling(window=window).mean()
            df[f'MA_Volume_{window}'] = df['Volume'].rolling(window=window).mean()
        
        # Volatility
        df['Volatility'] = df['Returns'].rolling(window=20).std()
        
        # Price momentum
        df['Momentum'] = df['Close'] - df['Close'].shift(5)
        
        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        return df.dropna() 