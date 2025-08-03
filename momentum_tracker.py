# momentum_tracker_date_range

import pandas as pd
import numpy as np
import requests
from io import StringIO
import time
import random
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from datetime import datetime, timedelta
from datetime import time as dt_time
import matplotlib.pyplot as plt
import yfinance as yf

# FinViz Configuration (backup option)
FINVIZ_AUTH = "d20e04ee-c9dd-4077-bac0-6139037bafd2"
REQUEST_DELAY = 2

class DateRangeAnalyzer:
    def __init__(self, ticker, start_date, end_date,
                 fast_period=8, slow_period=26, signal_period=9,
                 initial_capital=10000, commission=0.001,
                 trailing_stop_pct=0.10,  # New parameter
                 data_source='yfinance', frequency='1d'):
        """
        Initialize with trailing stop protection
        
        Parameters:
        - trailing_stop_pct: Percentage for trailing stop (e.g., 0.10 = 10%)
        """
        self.ticker = ticker
        self.start_date = pd.to_datetime(start_date).date() if isinstance(start_date, str) else start_date
        self.end_date = pd.to_datetime(end_date).date() if isinstance(end_date, str) else end_date
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal_period = signal_period
        self.initial_capital = initial_capital
        self.commission = commission
        self.trailing_stop_pct = trailing_stop_pct
        self.data_source = data_source
        self.frequency = frequency
        self.data = None
        self.model = None
        self.trade_log = []
        
    def download_data(self):
        """Download data for the date range"""
        print(f"Downloading data for {self.ticker}")
        print(f"Date range: {self.start_date} to {self.end_date}")
        print(f"Frequency: {self.frequency}")
        
        if self.data_source == 'yfinance':
            return self._download_yfinance_data()
        elif self.data_source == 'sample':
            return self._generate_sample_data()
        elif self.data_source == 'finviz':
            return self._download_finviz_data()
        else:
            raise ValueError("data_source must be 'yfinance', 'sample', or 'finviz'")
    
    def _download_yfinance_data(self):
        """Download data using yfinance"""
        try:
            print("Attempting to download from Yahoo Finance...")
            ticker_obj = yf.Ticker(self.ticker)
            
            # Convert frequency for yfinance
            yf_interval_map = {
                '1m': '1m',
                '5m': '5m', 
                '15m': '15m',
                '30m': '30m',
                '1h': '1h',
                '1d': '1d',
                '1wk': '1wk',
                '1mo': '1mo'
            }
            
            interval = yf_interval_map.get(self.frequency, '1d')
            
            # Download data for the date range
            data = ticker_obj.history(
                start=self.start_date,
                end=self.end_date + timedelta(days=1),  # Add 1 day to include end_date
                interval=interval,
                prepost=False
            )
            
            if data.empty:
                print("No data found for this date range. Falling back to sample data...")
                return self._generate_sample_data()
            
            # Standardize column names
            data.columns = [col.lower() for col in data.columns]
            
            self.data = data
            print(f"Downloaded {len(self.data)} records from Yahoo Finance")
            return self.data
            
        except Exception as e:
            print(f"Yahoo Finance download failed: {str(e)}")
            print("Falling back to sample data...")
            return self._generate_sample_data()
    
    def _download_finviz_data(self):
        """Download data from FinViz (backup method)"""
        print("Attempting FinViz download...")
        print("FinViz API requires proper authentication and may not work.")
        print("Falling back to sample data...")
        return self._generate_sample_data()
    
    def _generate_sample_data(self):
        """Generate realistic sample data for the date range"""
        print(f"Generating sample data from {self.start_date} to {self.end_date}")
        
        # Create date range based on frequency
        if self.frequency == '1d':
            dates = pd.date_range(start=self.start_date, end=self.end_date, freq='D')
            # Filter out weekends for daily data
            dates = dates[dates.weekday < 5]  # Monday=0, Sunday=6
        elif self.frequency == '1h':
            start_datetime = datetime.combine(self.start_date, dt_time(9, 30))  # Market open
            end_datetime = datetime.combine(self.end_date, dt_time(16, 0))     # Market close
            dates = pd.date_range(start=start_datetime, end=end_datetime, freq='H')
            # Filter for market hours (9:30 AM to 4:00 PM, weekdays only)
            dates = dates[(dates.hour >= 9) & (dates.hour <= 16) & (dates.weekday < 5)]
        elif self.frequency == '1m':
            start_datetime = datetime.combine(self.start_date, dt_time(9, 30))
            end_datetime = datetime.combine(self.end_date, dt_time(16, 0))
            dates = pd.date_range(start=start_datetime, end=end_datetime, freq='min')
            # Filter for market hours
            dates = dates[(dates.hour >= 9) & (dates.hour <= 16) & (dates.weekday < 5)]
            dates = dates[~((dates.hour == 9) & (dates.minute < 30))]  # Remove pre-9:30
        else:
            dates = pd.date_range(start=self.start_date, end=self.end_date, freq='D')
        
        if len(dates) == 0:
            raise ValueError("Invalid date range - no dates generated")
        
        # Generate realistic price movements
        base_price = 150.0
        if self.frequency == '1d':
            volatility = 0.02  # 2% daily volatility
        elif self.frequency == '1h':
            volatility = 0.008  # Hourly volatility
        else:
            volatility = 0.002  # Minute volatility
        
        # Create realistic price walk with trend
        returns = np.random.normal(0, volatility, len(dates))
        
        # Add some longer-term trend
        trend = np.linspace(0, 0.1, len(dates))  # 10% upward trend over period
        noise = np.random.normal(0, volatility * 0.5, len(dates))
        
        prices = base_price * np.exp(np.cumsum(returns + trend/len(dates) + noise))
        
        # Create OHLCV data
        data = pd.DataFrame(index=dates)
        data['open'] = prices * (1 + np.random.normal(0, volatility/4, len(dates)))
        data['close'] = prices * (1 + np.random.normal(0, volatility/4, len(dates)))
        
        # High and low should respect open/close
        oc_max = np.maximum(data['open'], data['close'])
        oc_min = np.minimum(data['open'], data['close'])
        
        data['high'] = oc_max * (1 + np.abs(np.random.normal(0, volatility/3, len(dates))))
        data['low'] = oc_min * (1 - np.abs(np.random.normal(0, volatility/3, len(dates))))
        
        # Volume depends on frequency
        if self.frequency == '1d':
            base_volume = 1000000  # 1M shares daily
        elif self.frequency == '1h':
            base_volume = 150000   # 150K shares hourly
        else:
            base_volume = 5000     # 5K shares per minute
            
        data['volume'] = np.random.poisson(base_volume, len(dates))
        
        self.data = data
        print(f"Generated {len(self.data)} sample records")
        return self.data
        
    def calculate_macd(self):
        """Calculate MACD and related features"""
        if self.data is None:
            raise ValueError("Data not loaded. Call download_data() first.")
            
        close = self.data['close']
        
        # Calculate MACD components
        ema_fast = close.ewm(span=self.fast_period, adjust=False).mean()
        ema_slow = close.ewm(span=self.slow_period, adjust=False).mean()
        macd = ema_fast - ema_slow
        signal = macd.ewm(span=self.signal_period, adjust=False).mean()
        histogram = macd - signal
        
        # Add features to dataframe
        self.data['macd'] = macd
        self.data['signal'] = signal
        self.data['histogram'] = histogram
        
        # Create time-based features
        self.data['year'] = self.data.index.year
        self.data['month'] = self.data.index.month
        self.data['day'] = self.data.index.day
        if self.frequency in ['1h', '1m']:
            self.data['hour'] = self.data.index.hour
            self.data['minute'] = self.data.index.minute
        self.data['weekday'] = self.data.index.weekday
        self.data['time_elapsed'] = range(len(self.data))
        
        # Create lagged features
        self.data['close_lag1'] = self.data['close'].shift(1)
        self.data['close_lag2'] = self.data['close'].shift(2)
        self.data['macd_lag1'] = self.data['macd'].shift(1)
        self.data['signal_lag1'] = self.data['signal'].shift(1)
        self.data['histogram_lag1'] = self.data['histogram'].shift(1)
        
        # Create momentum features
        self.data['price_change'] = self.data['close'].pct_change()
        self.data['macd_change'] = self.data['macd'].pct_change()
        
        # Rolling features
        window = min(5, len(self.data) // 4)  # Adaptive window size
        if window > 1:
            self.data['close_sma'] = self.data['close'].rolling(window=window).mean()
            self.data['volume_sma'] = self.data['volume'].rolling(window=window).mean()
        
        # Create target - will price increase in next period?
        self.data['target'] = (self.data['close'].shift(-1) > self.data['close']).astype(int)
        
        # Drop NA values
        original_len = len(self.data)
        self.data = self.data.dropna()
        print(f"Calculated MACD features. Kept {len(self.data)}/{original_len} records after removing NAs")
        
        return self.data
        
    def simple_trading_strategy(self):
        """Simple MACD-based trading strategy for the date range"""
        if self.data is None or 'macd' not in self.data.columns:
            raise ValueError("MACD not calculated. Call calculate_macd() first.")
            
        capital = self.initial_capital
        shares = 0
        position_open = False
        buy_price = 0
        highest_price = 0 # for trailing stop
        portfolio_values = []
        dynamic_stop = None #for adaptive trailing stops
        
        print(f"\nExecuting simple MACD strategy with {self.trailing_stop_pct*100:.1f}% trailing stop...")

        for i, (timestamp, row) in enumerate(self.data.iterrows()):
            current_price = row['close']
            macd = row['macd']
            signal_line = row['signal']
            
            # Calculate current portfolio value

            portfolio_value = shares * current_price if position_open else capital
            portfolio_values.append(portfolio_value)

            #trailing stop logic
            if position_open:
                if current_price > highest_price:
                    highest_price = current_price

                    if hasattr(self, 'use_volatility_stop') and self.use_volatility_stop:
                        recent_volatility = self.data['close'].pct_change().rolling(5).std().iloc[i]
                        self.trailing_stop_pct = min(0.20, max(0.05, recent_volatility * 2))

                #check trailing stop condition
                stop_price = highest_price * (1 - self.trailing_stop_pct)
                if current_price <= stop_price:
                    sell_value = shares * current_price * (1 - self.commission)
                    profit = sell_value - (shares * buy_price)
                    capital = sell_value

                    self.trade_log.append({
                        'timestamp': timestamp,
                        'action': 'SELL (Trailing Stop)',
                        'price': current_price,
                        'shares': shares,
                        'profit': profit,
                        'macd': macd,
                        'signal': signal_line,
                        'portfolio_value': capital,
                        'stop_pct': self.trailing_stop_pct
                    })

                    shares = 0
                    position_open = False
                    date_str = timestamp.strftime('%Y-%m-%d')
                    print(f"{date_str} - Trailing strop at ${current_price:.2f} (Profit: ${profit:.2f})")
                    continue

            #MACD
            if not position_open and macd > signal_line  and capital > 0:
                shares = (capital * (1 - self.commission))/current_price
                buy_price = current_price
                highest_price = current_price
                position_open = True

                self.trade_log.append({
                    'timestamp': timestamp,
                    'action': 'BUY',
                    'price': current_price,
                    'shares': shares,
                    'macd': macd,
                    'signal': signal_line,
                    'portfolio_value': capital
                })

                date_str = timestamp.strftime('%Y-%m-%d')
                print(f"{date_str} - BUY at ${current_price:.2f} (Stop: {self.trailing_stop_pct*100:.1f}%)")

            #sell signal (in case trailing stop is not triggered)
            elif position_open and macd < signal_line:
                sell_value = shares * current_price * (1 - self.commission)
                profit = sell_value - (shares * buy_price)
                capital = sell_value

                self.trade_log.append({
                    'timestamp': timestamp,
                    'action': 'SELL (MACD)',
                    'price': current_price,
                    'shares': shares,
                    'macd': macd,
                    'signal': signal_line,
                    'portfolio_value': capital
                })

                shares = 0
                position_open = False
                date_str = timestamp.strftime('%Y-%m-%d')
                print(f"{date_str} - SELL at ${current_price:.2f} (Profit: ${profit:.2f})")

        #close final position if still ope
        if position_open:
            final_price = self.data.iloc[-1]['close']
            sell_value = shares * final_price * (1 - self.commission)
            profit = sell_value - (shares * buy_price)
            capital = sell_value

            self.trade_log.append({
                'timestamp': self.data.index[-1],
                'action': 'SELL (Final)',
                'price': final_price,
                'profit': profit,
                'portfolio_value': capital
            })
            print(f"Final SELL at ${final_price:.2f} (Profit: ${profit:.2f})")


        
        return {
            'initial_capital': self.initial_capital,
            'final_capital': capital,
            #'total_return': total_return,
            #'return_pct': return_pct,
            #'annualized_return': annualized_return,
            #'buy_hold_return': buy_hold_return,
            #'alpha': return_pct - buy_hold_return,
            'num_trades': len([t for t in self.trade_log if t['action'] in ['BUY', 'SELL']]),
            #'num_days': num_days,
            'portfolio_values': portfolio_values,
            'trade_log': self.trade_log
        }
        
        
        
    def plot_analysis(self):
        """Plot price, MACD, and trading signals"""
        if self.data is None:
            raise ValueError("No data to plot")
            
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
        
        # Plot 1: Price and trading signals
        ax1.plot(self.data.index, self.data['close'], label='Close Price', linewidth=1.5)
        
        # Mark buy/sell points
        buy_plotted = False
        sell_plotted = False
        for trade in self.trade_log:
            if trade['action'] == 'BUY':
                ax1.scatter(trade['timestamp'], trade['price'], color='green', marker='^', s=100, 
                           label='Buy' if not buy_plotted else "")
                buy_plotted = True
            elif trade['action'] in ['SELL', 'SELL_FINAL']:
                ax1.scatter(trade['timestamp'], trade['price'], color='red', marker='v', s=100, 
                           label='Sell' if not sell_plotted else "")
                sell_plotted = True
        
        ax1.set_title(f'{self.ticker} Price and Trading Signals - {self.start_date} to {self.end_date}')
        ax1.set_ylabel('Price ($)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: MACD
        if 'macd' in self.data.columns:
            ax2.plot(self.data.index, self.data['macd'], label='MACD', color='blue')
            ax2.plot(self.data.index, self.data['signal'], label='Signal', color='red')
            ax2.bar(self.data.index, self.data['histogram'], label='Histogram', alpha=0.3, color='gray')
            ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            ax2.set_title('MACD Indicator')
            ax2.set_ylabel('MACD Value')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        plt.xlabel('Date')
        plt.tight_layout()
        plt.show()
        
    def run_analysis(self):
        """Run complete analysis for the date range"""
        print(f"\n{'='*70}")
        print(f"ANALYZING {self.ticker}")
        print(f"DATE RANGE: {self.start_date} to {self.end_date}")
        print(f"FREQUENCY: {self.frequency}")
        print(f"DATA SOURCE: {self.data_source}")
        print(f"{'='*70}")
        
        # Download and process data
        self.download_data()
        self.calculate_macd()
        
        # Run trading strategy
        results = self.simple_trading_strategy()
        
        # Display results
        print(f"\nSTRATEGY PERFORMANCE:")
        print(f"Initial Capital: ${results['initial_capital']:,.2f}")
        print(f"Final Capital: ${results['final_capital']:,.2f}")
        #print(f"Total Return: ${results['total_return']:,.2f} ({results['return_pct']:.2f}%)")
        #print(f"Annualized Return: {results['annualized_return']:.2f}%")
        #print(f"Buy & Hold Return: {results['buy_hold_return']:.2f}%")
        #print(f"Alpha (Outperformance): {results['alpha']:.2f}%")
        #print(f"Number of Trades: {results['num_trades']}")
        #print(f"Analysis Period: {results['num_days']} days")
        
        # Plot results
        self.plot_analysis()
        
        return results

# Example usage functions
def analyze_date_range(ticker, start_date, end_date, frequency='1d', data_source='sample'):
    """
    Analyze a date range with specified frequency
    
    Parameters:
    - ticker: Stock symbol (e.g., 'AAPL')
    - start_date: Start date string (e.g., '2024-01-01')
    - end_date: End date string (e.g., '2024-01-31')
    - frequency: Data frequency ('1d', '1h', '1m')
    - data_source: 'yfinance', 'sample', or 'finviz'
    """
    analyzer = DateRangeAnalyzer(
        ticker=ticker,
        start_date=start_date,
        end_date=end_date,
        frequency=frequency,
        data_source=data_source
    )
    
    return analyzer.run_analysis()

def analyze_multiple_periods(ticker, periods, frequency='1d', data_source='sample'):
    """
    Analyze multiple date ranges and compare results
    
    Parameters:
    - ticker: Stock symbol
    - periods: List of (start_date, end_date) tuples
    - frequency: Data frequency
    - data_source: Data source
    """
    results = []
    
    for i, (start_date, end_date) in enumerate(periods):
        print(f"\n{'='*50}")
        print(f"PERIOD {i+1}: {start_date} to {end_date}")
        print(f"{'='*50}")
        
        result = analyze_date_range(ticker, start_date, end_date, frequency, data_source)
        result['period'] = f"{start_date} to {end_date}"
        results.append(result)
    
    # Summary comparison
    print(f"\n{'='*70}")
    print("PERIOD COMPARISON SUMMARY")
    print(f"{'='*70}")
    print(f"{'Period':<25} {'Return %':<10} {'Alpha %':<10} {'Trades':<10}")
    print("-" * 70)
    
    for result in results:
        print(f"{result['period']:<25} {result['return_pct']:<10.2f} {result['alpha']:<10.2f} {result['num_trades']:<10}")
    
    return results

if __name__ == "__main__":
    # Example 1: Analyze AAPL for a month (daily data)
    results1 = analyze_date_range("REPL", "2025-07-9", "2025-07-16", frequency='1d', data_source='yfinance')
    
    
    # Example 3: Compare multiple periods
    # periods = [
    #     ("2024-01-01", "2024-01-31"),
    #     ("2024-02-01", "2024-02-29"),
    #     ("2024-03-01", "2024-03-31")
    # ]
    # results3 = analyze_multiple_periods("AAPL", periods, frequency='1d', data_source='sample')
    
    # Example 4: Try with real data (requires yfinance)
    # results4 = analyze_date_range("AAPL", "2024-01-01", "2024-01-31", frequency='1d', data_source='yfinance')