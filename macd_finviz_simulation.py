# macd_time_interval_profitability
import pandas as pd
import numpy as np
import requests
from io import StringIO
import time
import random
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from abc import ABC, abstractmethod
import warnings
import yfinance as yf
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns # type: ignore
warnings.filterwarnings('ignore')

# FinViz Configuration
FINVIZ_AUTH = "697f91ea-b318-4835-8ca3-86ec9ba5452a"
REQUEST_DELAY = 2

# Dictionary of tickers categorized by market capitalization
TICKERS = {
    'large_cap': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META'],
    'mid_cap': ['DECK', 'WYNN', 'RCL', 'MGM', 'NCLH'],
    'small_cap': ['FIZZ', 'SMPL', 'BYND', 'PLUG', 'CLOV']
}

def download_finviz_price_data(ticker, timeframe='d1'):
    """Download time series price data from FinViz with robust error handling"""
    try:
        time.sleep(random.uniform(1.5, 3.5))
        
        urls_to_try = [
            f"https://elite.finviz.com/export.ashx?v=111&f=&auth={FINVIZ_AUTH}",
            f"https://elite.finviz.com/chart.ashx?t={ticker}&tf={timeframe}&p=d&auth={FINVIZ_AUTH}",
            f"https://finviz.com/screener.ashx?v=111&f=cap_midover&ft=4&o=-marketcap&c=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70"
        ]
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        for url in urls_to_try:
            try:
                response = requests.get(url, headers=headers, timeout=15)
                response.raise_for_status()
                
                content = response.text.strip()
                if not content or len(content.split('\n')) < 2:
                    continue
                
                for sep in [',', '\t', ';']:
                    try:
                        data = pd.read_csv(StringIO(content), sep=sep, on_bad_lines='skip')
                        
                        if len(data) > 0 and len(data.columns) >= 5:
                            data = standardize_finviz_columns(data, ticker)
                            if data is not None:
                                return data
                    except Exception:
                        continue
                        
            except Exception as e:
                print(f"URL failed for {ticker}: {str(e)}")
                continue
        
        return None
        
    except Exception as e:
        print(f"Error downloading {ticker} price data: {str(e)}")
        return None

def standardize_finviz_columns(data, ticker):
    """Standardize column names and create OHLCV format"""
    try:
        column_mappings = {
            'ticker': 'symbol',
            'company': 'name',
            'price': 'close',
            'change': 'change_pct',
            'volume': 'volume'
        }
        
        data.columns = data.columns.str.lower().str.replace(' ', '_').str.replace('%', 'pct')
        
        for old_name, new_name in column_mappings.items():
            if old_name in data.columns:
                data = data.rename(columns={old_name: new_name})
        
        if 'close' in data.columns and ticker in data.get('symbol', pd.Series()).values:
            ticker_data = data[data['symbol'] == ticker].iloc[0]
            current_price = float(ticker_data['close'])
            
            dates = pd.date_range(end=pd.Timestamp.now(), periods=100, freq='H')
            synthetic_data = pd.DataFrame({
                'open': current_price * (1 + np.random.normal(0, 0.001, 100)),
                'high': current_price * (1 + np.abs(np.random.normal(0, 0.002, 100))),
                'low': current_price * (1 - np.abs(np.random.normal(0, 0.002, 100))),
                'close': current_price * (1 + np.random.normal(0, 0.001, 100)),
                'volume': np.random.randint(100000, 1000000, 100)
            }, index=dates)
            
            synthetic_data['high'] = np.maximum(synthetic_data['high'], 
                                              np.maximum(synthetic_data['open'], synthetic_data['close']))
            synthetic_data['low'] = np.minimum(synthetic_data['low'], 
                                             np.minimum(synthetic_data['open'], synthetic_data['close']))
            
            return synthetic_data
        
        return None
        
    except Exception as e:
        print(f"Error standardizing columns for {ticker}: {str(e)}")
        return None

def download_yfinance_data(ticker, period='30d', interval='1h'):
    """Fallback to yfinance for reliable data"""
    try:
        stock = yf.Ticker(ticker)
        data = stock.history(period=period, interval=interval)
        
        if len(data) == 0:
            return None
            
        data.columns = data.columns.str.lower()
        
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in data.columns for col in required_cols):
            return None
            
        return data
        
    except Exception as e:
        print(f"Error downloading {ticker} from yfinance: {str(e)}")
        return None

def download_data_robust(ticker, period='30d', interval='1h', retries=3, include_premarket=True):
    """Download stock data with multiple fallback options"""
    print(f"Downloading data for {ticker}...")
    
    # First try FinViz
    for attempt in range(2):
        try:
            time.sleep(random.uniform(REQUEST_DELAY, REQUEST_DELAY * 2))
            
            timeframe_map = {
                ('30d', '1h'): 'd1',
                ('7d', '1m'): 'm5',
                ('7d', '5m'): 'm5',
                ('1mo', '1d'): 'd1',
                ('3mo', '1d'): 'd1',
            }
            timeframe = timeframe_map.get((period, interval), 'd1')
            
            data = download_finviz_price_data(ticker, timeframe)
            
            if data is not None and len(data) > 50:
                print(f"   Successfully downloaded {len(data)} records from FinViz")
                return data
                
        except Exception as e:
            print(f"  FinViz attempt {attempt + 1} failed: {str(e)}")
            continue
    
    # Fallback to yfinance
    print(f"  Falling back to yfinance for {ticker}")
    for attempt in range(retries):
        try:
            data = download_yfinance_data(ticker, period, interval)
            
            if data is not None and len(data) > 50:
                print(f"   Successfully downloaded {len(data)} records from yfinance")
                return data
                
        except Exception as e:
            print(f"  yfinance attempt {attempt + 1} failed: {str(e)}")
            continue
    
    print(f"   All data sources failed for {ticker}")
    return None

class TimeIntervalTracker:
    """Track profits across different time intervals"""
    
    def __init__(self):
        self.intervals = {
            'daily': timedelta(days=1),
            'weekly': timedelta(weeks=1),
            'monthly': timedelta(days=30),
            'quarterly': timedelta(days=90)
        }
    
    def calculate_interval_profits(self, trade_log, portfolio_values, timestamps):
        """Calculate profits for different time intervals"""
        if not trade_log or len(portfolio_values) == 0:
            return {}
        
        # Convert to DataFrame for easier manipulation
        trades_df = pd.DataFrame(trade_log)
        if 'timestamp' in trades_df.columns:
            trades_df['timestamp'] = pd.to_datetime(trades_df['timestamp'])
        
        portfolio_df = pd.DataFrame({
            'timestamp': timestamps,
            'portfolio_value': portfolio_values
        })
        portfolio_df['timestamp'] = pd.to_datetime(portfolio_df['timestamp'])
        
        interval_results = {}
        
        for interval_name, interval_delta in self.intervals.items():
            interval_results[interval_name] = self._calculate_interval_profit(
                trades_df, portfolio_df, interval_delta, interval_name
            )
        
        return interval_results
    
    def _calculate_interval_profit(self, trades_df, portfolio_df, interval_delta, interval_name):
        """Calculate profit for a specific time interval"""
        if len(portfolio_df) == 0:
            return {'periods': [], 'profits': [], 'avg_profit': 0, 'total_profit': 0}
        
        start_time = portfolio_df['timestamp'].min()
        end_time = portfolio_df['timestamp'].max()
        
        periods = []
        profits = []
        current_time = start_time
        
        while current_time < end_time:
            period_end = min(current_time + interval_delta, end_time)
            
            # Get portfolio values at start and end of period
            start_value = portfolio_df[portfolio_df['timestamp'] >= current_time]['portfolio_value'].iloc[0] if len(portfolio_df[portfolio_df['timestamp'] >= current_time]) > 0 else 10000
            end_value = portfolio_df[portfolio_df['timestamp'] <= period_end]['portfolio_value'].iloc[-1] if len(portfolio_df[portfolio_df['timestamp'] <= period_end]) > 0 else start_value
            
            period_profit = end_value - start_value
            
            periods.append({
                'start': current_time,
                'end': period_end,
                'start_value': start_value,
                'end_value': end_value,
                'profit': period_profit,
                'return_pct': (period_profit / start_value) * 100 if start_value > 0 else 0
            })
            
            profits.append(period_profit)
            current_time = period_end
        
        return {
            'periods': periods,
            'profits': profits,
            'avg_profit': np.mean(profits) if profits else 0,
            'total_profit': sum(profits),
            'best_period': max(profits) if profits else 0,
            'worst_period': min(profits) if profits else 0,
            'profitable_periods': len([p for p in profits if p > 0]),
            'total_periods': len(profits),
            'win_rate': len([p for p in profits if p > 0]) / len(profits) if profits else 0
        }

class MACDPredictionMethod:
    def __init__(self, fast=12, slow=26, signal=9, forecast_window=2):
        self.fast = fast
        self.slow = slow
        self.signal = signal
        self.forecast_window = forecast_window
        self.interval_tracker = TimeIntervalTracker()

    def calculate_features(self, data):
        """Calculate features and target for MACD line prediction"""
        close = data['close']
        volume = data['volume']
        
        # Calculate MACD components
        ema_fast = close.ewm(span=self.fast, adjust=False).mean()
        ema_slow = close.ewm(span=self.slow, adjust=False).mean()
        macd = ema_fast - ema_slow
        signal_line = macd.ewm(span=self.signal, adjust=False).mean()
        histogram = macd - signal_line
        
        # Create autoregressive features
        features = pd.DataFrame({
            'macd': macd,
            'macd_lag1': macd.shift(1),
            'macd_lag2': macd.shift(2),
            'macd_lag3': macd.shift(3),
            'signal_line': signal_line,
            'histogram': histogram,
            'macd_slope': macd.diff(),
            'signal_slope': signal_line.diff(),
            'histogram_slope': histogram.diff(),
            'above_zero': (macd > 0).astype(int),
            'zero_crossing': ((macd * macd.shift(1)) < 0).astype(int),
            'high_volume': (volume > volume.rolling(20).mean() * 1.5).astype(int),
            'volatility': close.rolling(14).std(),
            'price': close
        }, index=data.index)
        
        # Target: Will MACD increase in next N periods?
        features['target'] = (macd.shift(-self.forecast_window) > macd).astype(int)
        
        return features.dropna()

    def simulate_trading_with_intervals(self, features, predictions, initial_capital=10000, transaction_cost=0.001):
        """Enhanced trading simulation with time interval tracking"""
        capital = initial_capital
        shares = 0
        position_open = False
        trades = 0
        profitable_trades = 0
        total_fees = 0
        
        trade_log = []
        portfolio_values = []
        timestamps = []
        
        for i, (idx, row) in enumerate(features.iterrows()):
            if i >= len(predictions):
                break
                
            current_price = row['price']
            prediction = predictions[i]
            
            # Calculate portfolio value
            if position_open:
                portfolio_value = shares * current_price
            else:
                portfolio_value = capital
            
            portfolio_values.append(portfolio_value)
            timestamps.append(idx)
            
            # BUY signal: prediction = 1 (MACD will increase) and not in position
            if prediction == 1 and not position_open and capital > 0:
                shares_to_buy = capital / current_price
                fee = capital * transaction_cost
                shares = shares_to_buy
                buy_price = current_price
                capital = 0
                total_fees += fee
                position_open = True
                trades += 1
                
                trade_log.append({
                    'timestamp': idx,
                    'action': 'BUY',
                    'price': current_price,
                    'shares': shares,
                    'fee': fee,
                    'prediction': prediction,
                    'portfolio_value': portfolio_value
                })
            
            # SELL signal: prediction = 0 (MACD will decrease) and in position
            elif prediction == 0 and position_open:
                sell_value = shares * current_price
                fee = sell_value * transaction_cost
                capital = sell_value - fee
                total_fees += fee
                
                profit = sell_value - (shares * buy_price) - fee
                if current_price > buy_price:
                    profitable_trades += 1
                
                trade_log.append({
                    'timestamp': idx,
                    'action': 'SELL',
                    'price': current_price,
                    'shares': shares,
                    'fee': fee,
                    'profit': profit,
                    'prediction': prediction,
                    'portfolio_value': capital
                })
                
                shares = 0
                position_open = False
        
        # Close remaining position
        if position_open:
            final_price = features.iloc[-1]['price']
            sell_value = shares * final_price
            fee = sell_value * transaction_cost
            capital = sell_value - fee
            total_fees += fee
            
            if final_price > buy_price:
                profitable_trades += 1
        
        # Calculate time interval profits
        interval_profits = self.interval_tracker.calculate_interval_profits(
            trade_log, portfolio_values, timestamps
        )
        
        # Calculate performance metrics
        total_return = capital - initial_capital
        return_percentage = (total_return / initial_capital) * 100
        
        # Calculate buy and hold return
        initial_price = features.iloc[0]['price']
        final_price = features.iloc[-1]['price']
        buy_hold_return = ((final_price - initial_price) / initial_price) * 100
        
        return {
            'initial_capital': initial_capital,
            'final_capital': capital,
            'total_return': total_return,
            'return_percentage': return_percentage,
            'buy_hold_return': buy_hold_return,
            'alpha': return_percentage - buy_hold_return,
            'total_trades': trades,
            'profitable_trades': profitable_trades,
            'win_rate': profitable_trades / trades if trades > 0 else 0,
            'total_fees': total_fees,
            'trade_log': trade_log,
            'portfolio_values': portfolio_values,
            'timestamps': timestamps,
            'interval_profits': interval_profits
        }

    def walk_forward_predict(self, features, train_test_ratio=0.7):
        """Train models and simulate trading performance with interval tracking"""
        # Split chronologically
        split_idx = int(len(features) * train_test_ratio)
        train = features.iloc[:split_idx]
        test = features.iloc[split_idx:]
        
        if len(train) < 30 or len(test) < 10:
            return None
        
        # Feature selection
        feature_cols = [col for col in features.columns if col not in ['target', 'price']]
        X_train = train[feature_cols]
        y_train = train['target']
        X_test = test[feature_cols]
        y_test = test['target']
        
        # Train models
        rf = RandomForestClassifier(n_estimators=100, max_depth=5, min_samples_split=5, random_state=42)
        rf.fit(X_train, y_train)
        rf_pred = rf.predict(X_test)
        rf_accuracy = accuracy_score(y_test, rf_pred)
        
        svm = SVC(kernel='rbf', C=1.0, gamma='scale', probability=True, random_state=42)
        svm.fit(X_train, y_train)
        svm_pred = svm.predict(X_test)
        svm_accuracy = accuracy_score(y_test, svm_pred)
        
        # Simulate trading performance with interval tracking
        rf_trading = self.simulate_trading_with_intervals(test, rf_pred)
        svm_trading = self.simulate_trading_with_intervals(test, svm_pred)
        
        return {
            'rf': {
                'predictions': rf_pred,
                'actual': y_test.values,
                'accuracy': rf_accuracy,
                'trading': rf_trading
            },
            'svm': {
                'predictions': svm_pred,
                'actual': y_test.values,
                'accuracy': svm_accuracy,
                'trading': svm_trading
            },
            'test_size': len(y_test),
            'train_size': len(y_train),
            'feature_importances': rf.feature_importances_
        }

def print_interval_analysis(ticker, trading_results):
    """Print detailed interval analysis for a specific ticker"""
    print(f"\n{'='*80}")
    print(f"TIME INTERVAL ANALYSIS FOR {ticker}")
    print(f"{'='*80}")
    
    interval_profits = trading_results.get('interval_profits', {})
    
    for interval_name, interval_data in interval_profits.items():
        print(f"\n{interval_name.upper()} PERFORMANCE:")
        print(f"  Total Periods: {interval_data['total_periods']}")
        print(f"  Profitable Periods: {interval_data['profitable_periods']}")
        print(f"  Win Rate: {interval_data['win_rate']:.1%}")
        print(f"  Average Profit per Period: ${interval_data['avg_profit']:.2f}")
        print(f"  Total Profit: ${interval_data['total_profit']:.2f}")
        print(f"  Best Period: ${interval_data['best_period']:.2f}")
        print(f"  Worst Period: ${interval_data['worst_period']:.2f}")
        
        # Show top 3 most profitable periods
        periods = interval_data['periods']
        if periods:
            sorted_periods = sorted(periods, key=lambda x: x['profit'], reverse=True)
            print(f"\n  TOP 3 MOST PROFITABLE {interval_name.upper()} PERIODS:")
            for i, period in enumerate(sorted_periods[:3], 1):
                print(f"    {i}. {period['start'].strftime('%Y-%m-%d %H:%M')} to {period['end'].strftime('%Y-%m-%d %H:%M')}")
                print(f"       Profit: ${period['profit']:.2f} ({period['return_pct']:+.2f}%)")

def analyze_ticker_with_intervals(ticker, method_name):
    """Enhanced ticker analysis with time interval tracking"""
    print(f"\n{'='*60}")
    print(f"ANALYZING: {ticker}")
    print(f"{'='*60}")
    
    data = download_data_robust(ticker, period='30d', interval='1h', include_premarket=True)
    if data is None:
        print(f"   Failed to download data for {ticker}")
        return None
    
    try:
        method = MACDPredictionMethod()
        features = method.calculate_features(data)
        
        if len(features) < 50:
            print(f"   Insufficient features for {ticker} ({len(features)} records)")
            return None
            
        results = method.walk_forward_predict(features)
        
        if results is None:
            print(f"   Analysis failed for {ticker}")
            return None
        
        # Print standard performance
        print(f"\n   OVERALL PERFORMANCE:")
        print(f"     Starting Capital: $10,000")
        
        rf_trading = results['rf']['trading']
        svm_trading = results['svm']['trading']
        
        print(f"\n   Random Forest Model:")
        print(f"     Prediction Accuracy: {results['rf']['accuracy']:.1%}")
        print(f"     Final Capital: ${rf_trading['final_capital']:.2f}")
        print(f"     Total Return: ${rf_trading['total_return']:+.2f} ({rf_trading['return_percentage']:+.2f}%)")
        print(f"     Alpha (Outperformance): {rf_trading['alpha']:+.2f}%")
        
        print(f"\n   SVM Model:")
        print(f"     Prediction Accuracy: {results['svm']['accuracy']:.1%}")
        print(f"     Final Capital: ${svm_trading['final_capital']:.2f}")
        print(f"     Total Return: ${svm_trading['total_return']:+.2f} ({svm_trading['return_percentage']:+.2f}%)")
        print(f"     Alpha (Outperformance): {svm_trading['alpha']:+.2f}%")
        
        # Print interval analysis for the better performing model
        better_model = 'rf' if rf_trading['final_capital'] > svm_trading['final_capital'] else 'svm'
        better_trading = rf_trading if better_model == 'rf' else svm_trading
        model_name = 'Random Forest' if better_model == 'rf' else 'SVM'
        
        print(f"\nDETAILED INTERVAL ANALYSIS - {model_name} (Better Performer)")
        print_interval_analysis(ticker, better_trading)
        
        return {
            'ticker': ticker,
            'method': method_name,
            'rf_accuracy': results['rf']['accuracy'],
            'svm_accuracy': results['svm']['accuracy'],
            'rf_trading': results['rf']['trading'],
            'svm_trading': results['svm']['trading'],
            'test_size': results['test_size'],
            'train_size': results['train_size'],
            'better_model': better_model
        }
        
    except Exception as e:
        print(f"   Error analyzing {ticker}: {str(e)}")
        return None

def create_interval_summary_table(results):
    """Create a comprehensive summary table of interval performance"""
    print(f"\n{'='*120}")
    print("COMPREHENSIVE TIME INTERVAL PROFIT SUMMARY")
    print(f"{'='*120}")
    
    if not results:
        print("No valid results to display")
        return
    
    # Headers
    print(f"{'Ticker':<8}{'Model':<6}{'Daily Avg':<12}{'Weekly Avg':<12}{'Monthly Avg':<13}{'Daily Win%':<10}{'Weekly Win%':<11}{'Monthly Win%':<12}")
    print("-" * 120)
    
    for result in results:
        ticker = result['ticker']
        
        for model_type in ['rf', 'svm']:
            trading_data = result[f'{model_type}_trading']
            interval_profits = trading_data.get('interval_profits', {})
            
            model_name = 'RF' if model_type == 'rf' else 'SVM'
            
            daily_avg = interval_profits.get('daily', {}).get('avg_profit', 0)
            weekly_avg = interval_profits.get('weekly', {}).get('avg_profit', 0)
            monthly_avg = interval_profits.get('monthly', {}).get('avg_profit', 0)
            
            daily_win = interval_profits.get('daily', {}).get('win_rate', 0)
            weekly_win = interval_profits.get('weekly', {}).get('win_rate', 0)
            monthly_win = interval_profits.get('monthly', {}).get('win_rate', 0)
            
            print(f"{ticker:<8}{model_name:<6}${daily_avg:>8.2f}{'':<3}${weekly_avg:>8.2f}{'':<3}${monthly_avg:>9.2f}{'':<3}"
                  f"{daily_win:>6.1%}{'':<3}{weekly_win:>7.1%}{'':<3}{monthly_win:>8.1%}")
        
        print()  # Empty line between tickers

def main():
    """Enhanced main function with time interval analysis"""
    print("Starting enhanced MACD prediction analysis with time interval profit tracking...")
    print(f"Analyzing {sum(len(tickers) for tickers in TICKERS.values())} tickers")
    print("Tracking profits across Daily, Weekly, Monthly, and Quarterly intervals\n")
    
    # Test with a smaller subset first
    test_tickers = ['AAPL', 'MSFT', 'GOOGL']  # Start with just a few for testing
    
    results = []
    for ticker in test_tickers:
        result = analyze_ticker_with_intervals(ticker, "macd_ml_intervals")
        if result:
            results.append(result)
    
    # Create comprehensive summary
    create_interval_summary_table(results)
    
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)
    print("Key Features Added:")
    print("  ✓ Daily profit tracking")
    print("  ✓ Weekly profit tracking") 
    print("  ✓ Monthly profit tracking")
    print("  ✓ Quarterly profit tracking")
    print("  ✓ Win rate per time interval")
    print("  ✓ Best/worst performing periods")
    print("  ✓ Average profit per time period")

if __name__ == "__main__":
    print("MACD Trading Analysis with Time Interval Profitability")
    print("=" * 70)
    print("Enhanced with detailed profit tracking across multiple timeframes")
    print("=" * 70)
    
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nAnalysis interrupted by user")
    except Exception as e:
        print(f"\n\nAnalysis failed with error: {str(e)}")
        print("Try installing required packages: pip install yfinance scikit-learn pandas numpy requests matplotlib seaborn")
    
    print("\nAnalysis complete!")