import yfinance as yf # type: ignore
import pandas as pd
from datetime import datetime, timedelta
from sklearn.model_selection import TimeSeriesSplit
import numpy as np
import pytz

"""
# Get Microsoft stock data
msft = yf.Ticker("MSFT")

# Get 1-minute data for the last 7 days (max allowed by Yahoo Finance)
data = msft.history(period="7d", interval="1m")

# Save to CSV
#data.to_csv("MSFT_1min.csv")

# Display basic info
print(data.head())
print("\nData shape:", data.shape)
"""

"""
Outputs explained: 
Datetime - Timestamp (in EST). information from 9AM 
Open - opening price (aka someone bought MSFT shares for $432.93 at that minute)
High - highest price sold during that minute
Low - lowest price sold during that minute
Volume - number of sharest traded that minute
Dividends - Cash payments companies make to shareholders from their profits
Stock Splits - When a company changes its share count without changing market value (ex. a 2-1 split means you get 2 stocks for every 1 stock you own)

"""


def get_1min_data(ticker, days=7):
    """Get clean 1-minute market hours data"""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    try:
        # Download data
        data = yf.download(
            tickers=ticker,
            interval="1m",
            start=start_date,
            end=end_date,
            prepost=False,  # Only regular trading hours
            auto_adjust=True,
            threads=True
        )
        
        if data.empty:
            print(f"No data returned for {ticker}")
            return None
            
        # Handle timezone conversion
        if data.index.tz is None:
            data = data.tz_localize('UTC')
        data = data.tz_convert('America/New_York')
        
        # Clean and filter
        data = data[data['Volume'] > 0]  # Remove zero-volume periods
        data = data.between_time('09:30', '16:00')  # Market hours only
        
        if data.empty:
            print("No data remaining after market hours filtering")
            return None
            
        # Save with timezone-aware timestamps
        filename = f"{ticker}_1min_clean_{start_date.date()}_to_{end_date.date()}.csv"
        data.to_csv(filename, index=True)
        print(f"Saved {len(data)} market minutes to {filename}")
        return data
    
    except Exception as e:
        print(f"Error fetching data: {e}")
        return None

def split_by_dates(data, train_ratio=0.6, val_ratio=0.2):
    """Split by trading days with proper date comparison and empty set handling"""
    try:
        # Get unique trading dates correctly
        trading_days = pd.Series(data.index.date).unique()
        if len(trading_days) < 2:
            raise ValueError("Not enough trading days to split")
            
        train_end = int(len(trading_days) * train_ratio)
        val_end = train_end + int(len(trading_days) * val_ratio)
        
        # Ensure we have at least 1 day for each set
        train_end = max(1, train_end)
        val_end = max(train_end + 1, val_end)
        
        # Get the actual date objects for splitting
        train_end_date = trading_days[train_end-1]  # inclusive end
        val_end_date = trading_days[val_end-1] if val_end < len(trading_days) else trading_days[-1]
        
        # Compare dates using .date()
        train = data[data.index.date <= train_end_date]
        val = data[(data.index.date > train_end_date) & 
                   (data.index.date <= val_end_date)] if val_end < len(trading_days) else pd.DataFrame()
        test = data[data.index.date > val_end_date] if val_end < len(trading_days) else pd.DataFrame()
        
        return train, val, test
        
    except Exception as e:
        print(f"Error splitting data: {e}")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

def safe_get_range(df):
    """Safely get date range for potentially empty DataFrames"""
    if df.empty:
        return "No data"
    return f"{df.index[0]} to {df.index[-1]}"

if __name__ == "__main__":
    # 1. Get clean data
    msft_data = get_1min_data("MSFT", days=7)
    
    if msft_data is not None:
        print("\nFirst 5 market hours:")
        print(msft_data.head())
        print("\nLast 5 market hours:")
        print(msft_data.tail())
        print(f"\nData shape: {msft_data.shape}")
        print(f"Market days: {msft_data.index.date.min()} to {msft_data.index.date.max()}")
        print(f"Market hours: {msft_data.index.time.min()} to {msft_data.index.time.max()}")
        
        # 2. Split by trading days with 60/20/20 ratio
        train, val, test = split_by_dates(msft_data)
        
        print("\nSplit sizes:")
        print(f"Training: {len(train)} rows ({safe_get_range(train)})")
        print(f"Validation: {len(val)} rows ({safe_get_range(val)})")
        print(f"Test: {len(test)} rows ({safe_get_range(test)})")
        
        # Additional validation
        if len(train) == 0 or len(val) == 0 or len(test) == 0:
            print("\nWarning: One or more splits are empty. Adjust your split ratios or get more data.")

"""
time coverage: full trading days from May 12-16, 2025 (no missing days)
Split Effectiveness:
Training Set: 60% of data (3 days: May 12-14)
Validation Set: 20% (1 day: May 15)
Test Set: 20% (1 day: May 16)
"""