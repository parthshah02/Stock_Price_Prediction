from src.data.data_loader import StockDataLoader
from src.data.data_processor import StockDataProcessor
from src.features.feature_engineering import FeatureEngineer
from src.models.arima_model import ARIMAModel
from src.models.gradient_boosting_model import GBModel
import logging
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta

def setup_logging():
    logging.basicConfig(level=logging.INFO)

def plot_stock_analysis(df, symbol, arima_predictions=None, gb_predictions=None, y_test=None):
    """
    Create comprehensive plots for stock analysis with smaller figure size
    """
    # Clear any existing plots
    plt.clf()
    
    # Create new figure with reduced size
    fig = plt.figure(figsize=(16, 10))  # Reduced from (24, 16)
    
    # Add main title with stock symbol
    plt.suptitle(f'Stock Analysis for {symbol}', fontsize=16, y=0.95, fontweight='bold')
    
    # Adjust spacing between subplots
    plt.subplots_adjust(
        top=0.85,    # More space for main title
        bottom=0.1,
        left=0.1,
        right=0.9,
        wspace=0.25, # Reduced from 0.3
        hspace=0.35  # Increased slightly for better vertical separation
    )
    
    # 1. Historical Price and Volume
    ax1 = plt.subplot(2, 2, 1)
    ax1.plot(df.index, df['Close'], label='Close Price', color='blue', linewidth=1.5)
    ax1.set_title(f'{symbol} Historical Price', fontsize=12)
    ax1.set_xlabel('Date', fontsize=10)
    ax1.set_ylabel('Price', fontsize=10)
    ax1.legend(fontsize=8)
    ax1.tick_params(axis='x', rotation=45, labelsize=8)
    ax1.tick_params(axis='y', labelsize=8)
    ax1.grid(True, alpha=0.3)
    
    # 2. Technical Indicators
    ax2 = plt.subplot(2, 2, 2)
    ax2.plot(df.index, df['MA_20'], label='20-day MA', color='red', linewidth=1.5)
    ax2.plot(df.index, df['MA_50'], label='50-day MA', color='green', linewidth=1.5)
    ax2.plot(df.index, df['Close'], label='Close Price', color='blue', alpha=0.5)
    ax2.set_title(f'{symbol} Technical Indicators', fontsize=12)
    ax2.set_xlabel('Date', fontsize=10)
    ax2.set_ylabel('Price', fontsize=10)
    ax2.legend(fontsize=8)
    ax2.tick_params(axis='x', rotation=45, labelsize=8)
    ax2.tick_params(axis='y', labelsize=8)
    ax2.grid(True, alpha=0.3)
    
    # 3. Model Predictions
    ax3 = plt.subplot(2, 2, 3)
    if y_test is not None:
        ax3.plot(df.index[-len(y_test):], y_test, label='Actual', color='blue', linewidth=1.5)
        if gb_predictions is not None:
            ax3.plot(df.index[-len(gb_predictions):], gb_predictions, 
                    label='GB Predictions', color='green', linewidth=1.5)
        if arima_predictions is not None:
            last_date = df.index[-1]
            future_dates = [last_date + timedelta(days=x) for x in range(1, 31)]
            ax3.plot(future_dates, arima_predictions, 
                    label='ARIMA Predictions', color='red', linewidth=1.5)
    ax3.set_title(f'{symbol} Price Predictions', fontsize=12)
    ax3.set_xlabel('Date', fontsize=10)
    ax3.set_ylabel('Price', fontsize=10)
    ax3.legend(fontsize=8)
    ax3.tick_params(axis='x', rotation=45, labelsize=8)
    ax3.tick_params(axis='y', labelsize=8)
    ax3.grid(True, alpha=0.3)
    
    # 4. Returns Distribution
    ax4 = plt.subplot(2, 2, 4)
    ax4.hist(df['Returns'].dropna(), bins=50, color='skyblue', edgecolor='black')
    ax4.set_title(f'{symbol} Returns Distribution', fontsize=12)
    ax4.set_xlabel('Returns', fontsize=10)
    ax4.set_ylabel('Frequency', fontsize=10)
    ax4.tick_params(axis='x', labelsize=8)
    ax4.tick_params(axis='y', labelsize=8)
    ax4.grid(True, alpha=0.3)
    
    # Adjust layout
    plt.tight_layout()
    
    # Display the plot
    plt.show()
    
    # Clear the figure
    plt.close(fig)

def main():
    setup_logging()
    
    # Initialize components
    symbols = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'META']
    data_loader = StockDataLoader(symbols, '2020-01-01', '2024-01-01')
    data_processor = StockDataProcessor()
    
    # Load all stock data first
    print("Fetching data for all stocks...")
    stock_data = data_loader.fetch_data()
    
    # Process each stock
    for symbol in symbols:  # Use symbols list to maintain order
        print(f"\n{'='*50}")
        print(f"Processing {symbol}...")
        print(f"{'='*50}")
        
        try:
            df = stock_data[symbol]
            
            # Clean and prepare data
            df = data_processor.clean_data(df)
            df = data_processor.prepare_time_series(df)
            
            # Feature engineering
            df = FeatureEngineer.create_features(df)
            
            # ARIMA Model
            arima = ARIMAModel(p=1, d=1, q=1)
            arima_predictions = None
            
            try:
                if arima.check_stationarity(df['Close']):
                    arima_model = arima.fit(df['Close'])
                    arima_predictions = arima.predict(30)
                else:
                    print(f"{symbol}: Data is not stationary, differencing might be needed")
            except Exception as e:
                print(f"{symbol}: Error in ARIMA modeling: {str(e)}")
            
            # Gradient Boosting Model
            feature_cols = ['Returns', 'Volatility', 'Momentum', 'RSI']
            gb_model = GBModel()
            X_train, X_test, y_train, y_test = gb_model.prepare_data(
                df, 'Close', feature_cols
            )
            gb_model.fit(X_train, y_train)
            gb_predictions = gb_model.predict(X_test)
            
            # Print metrics
            print(f"\nResults for {symbol}:")
            if arima_predictions is not None:
                print("ARIMA RMSE:", np.sqrt(mean_squared_error(
                    df['Close'][-30:], arima_predictions)))
            print("GB RMSE:", np.sqrt(mean_squared_error(y_test, gb_predictions)))
            
            # Plot the results
            print(f"\nGenerating plots for {symbol}...")
            plot_stock_analysis(df, symbol, arima_predictions, gb_predictions, y_test)
            
            print(f"\nFinished processing {symbol}")
            
            # Add a small pause between plots (optional)
            plt.pause(1)
            
        except Exception as e:
            print(f"Error processing {symbol}: {str(e)}")
            continue
    
    print("\nAll stocks have been processed!")

if __name__ == "__main__":
    main() 