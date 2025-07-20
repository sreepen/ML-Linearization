# macd_interval_comparison.py
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from abc import ABC, abstractmethod
import warnings
warnings.filterwarnings('ignore')

# Dictionary of tickers categorized by market capitalization
TICKERS = {
    'large_cap': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META'],
    'mid_cap': ['DECK', 'WYNN', 'RCL', 'MGM', 'NCLH'],
    'small_cap': ['FIZZ', 'SMPL', 'BYND', 'PLUG', 'CLOV']
}

def download_data_robust(ticker, period='7d', interval='1m', retries=3, include_premarket=True):
    """Download stock data with retry logic and basic cleaning"""
    for attempt in range(retries):
        try:
            data = yf.download(ticker, period=period, interval=interval, 
                             progress=False, auto_adjust=True, prepost=include_premarket)
            if data.empty:
                continue
                
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = [col[0].lower() if isinstance(col, tuple) else col.lower() 
                               for col in data.columns]
            else:
                data.columns = data.columns.str.lower()
            
            if not include_premarket and isinstance(data.index, pd.DatetimeIndex):
                try:
                    data = data.between_time('09:30', '16:00')
                except:
                    pass
            
            if len(data) > 100:
                return data
        except Exception as e:
            continue
    return None

class PredictionMethod(ABC):
    @abstractmethod
    def calculate_features(self, data: pd.DataFrame) -> pd.DataFrame:
        pass
    
    @abstractmethod
    def walk_forward_predict(self, features: pd.DataFrame, train_test_ratio=0.7) -> dict:
        pass

class MACDPredictionMethod(PredictionMethod):
    def __init__(self, fast=12, slow=26, signal=9, forecast_window=2):
        self.fast = fast
        self.slow = slow
        self.signal = signal
        self.forecast_window = forecast_window

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
            'price': close  # Add price for trading simulation
        }, index=data.index)
        
        # Target: Will MACD increase in next N periods?
        features['target'] = (macd.shift(-self.forecast_window) > macd).astype(int)
        
        return features.dropna()

    def simulate_trading(self, features, predictions, initial_capital=10000, transaction_cost=0.001):
        """Simulate trading based on predictions and return monetary performance"""
        capital = initial_capital
        shares = 0
        position_open = False
        trades = 0
        profitable_trades = 0
        total_fees = 0
        
        trade_log = []
        portfolio_values = []
        
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
                    'prediction': prediction
                })
            
            # SELL signal: prediction = 0 (MACD will decrease) and in position
            elif prediction == 0 and position_open:
                sell_value = shares * current_price
                fee = sell_value * transaction_cost
                capital = sell_value - fee
                total_fees += fee
                
                if current_price > buy_price:
                    profitable_trades += 1
                
                trade_log.append({
                    'timestamp': idx,
                    'action': 'SELL',
                    'price': current_price,
                    'shares': shares,
                    'fee': fee,
                    'profit': sell_value - (shares * buy_price) - fee,
                    'prediction': prediction
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
            'trade_log': trade_log
        }

    def walk_forward_predict(self, features, train_test_ratio=0.7):
        """Train models and simulate trading performance"""
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
        
        # Simulate trading performance
        rf_trading = self.simulate_trading(test, rf_pred)
        svm_trading = self.simulate_trading(test, svm_pred)
        
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

class TimeRangeMACDPredictionMethod(MACDPredictionMethod):
    def __init__(self, time_ranges=None, **kwargs):
        super().__init__(**kwargs)
        self.time_ranges = time_ranges or [
            (7, 9),     # Pre-Market
            (7, 11),    # Early session
            (9, 11),    # Early morning
            (11, 13),   # Midday
            (13, 15),   # Early afternoon
            (15, 16)    # Late afternoon
        ]
    
    def walk_forward_predict(self, features, train_test_ratio=0.7):
        """Perform time-range specific analysis with trading simulation"""
        results = {}
        
        for start_hour, end_hour in self.time_ranges:
            # Filter data for this time range
            time_mask = (features.index.time >= pd.to_datetime(f"{start_hour}:00").time()) & \
                        (features.index.time <= pd.to_datetime(f"{end_hour}:00").time())
            time_features = features[time_mask]
            
            if len(time_features) < 20:
                continue
            
            # Split chronologically
            split_idx = int(len(time_features) * train_test_ratio)
            train = time_features.iloc[:split_idx]
            test = time_features.iloc[split_idx:]
            
            if len(train) < 10 or len(test) < 5:
                continue
            
            # Feature selection
            feature_cols = [col for col in features.columns if col not in ['target', 'price']]
            X_train = train[feature_cols]
            y_train = train['target']
            X_test = test[feature_cols]
            y_test = test['target']
            
            # Train models
            rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
            rf.fit(X_train, y_train)
            rf_pred = rf.predict(X_test)
            rf_accuracy = accuracy_score(y_test, rf_pred)
            
            svm = SVC(kernel='rbf', C=1.0, gamma='scale', probability=True, random_state=42)
            svm.fit(X_train, y_train)
            svm_pred = svm.predict(X_test)
            svm_accuracy = accuracy_score(y_test, svm_pred)
            
            # Simulate trading
            rf_trading = self.simulate_trading(test, rf_pred)
            svm_trading = self.simulate_trading(test, svm_pred)
            
            time_key = f"{start_hour:02d}:00-{end_hour:02d}:00"
            results[time_key] = {
                'rf_accuracy': rf_accuracy,
                'svm_accuracy': svm_accuracy,
                'rf_trading': rf_trading,
                'svm_trading': svm_trading,
                'test_size': len(y_test),
                'train_size': len(y_train),
                'feature_importances': rf.feature_importances_
            }
        
        return results if results else None

def analyze_ticker(ticker, method_name):
    """Analyze a ticker with monetary performance tracking"""
    print(f"\n{'='*60}")
    print(f"ANALYZING: {ticker}")
    print(f"{'='*60}")
    
    data = download_data_robust(ticker, include_premarket=True)
    if data is None:
        print(f" Failed to download data for {ticker}")
        return None
    
    try:
        # Standard analysis
        standard_method = MACDPredictionMethod()
        features = standard_method.calculate_features(data)
        standard_results = standard_method.walk_forward_predict(features)
        
        # Time-range analysis
        time_method = TimeRangeMACDPredictionMethod()
        time_results = time_method.walk_forward_predict(features)
        
        if standard_results is None and time_results is None:
            print(f" Insufficient data for analysis")
            return None
        
        # Print overall performance
        if standard_results:
            print(f"\n OVERALL PERFORMANCE:")
            print(f"   Starting Capital: $10,000")
            
            rf_trading = standard_results['rf']['trading']
            svm_trading = standard_results['svm']['trading']
            
            print(f"\n Random Forest Model:")
            print(f"   Prediction Accuracy: {standard_results['rf']['accuracy']:.1%}")
            print(f"   Final Capital: ${rf_trading['final_capital']:.2f}")
            print(f"   Total Return: ${rf_trading['total_return']:+.2f} ({rf_trading['return_percentage']:+.2f}%)")
            print(f"   Buy & Hold Return: {rf_trading['buy_hold_return']:+.2f}%")
            print(f"   Alpha (Outperformance): {rf_trading['alpha']:+.2f}%")
            print(f"   Total Trades: {rf_trading['total_trades']}")
            print(f"   Win Rate: {rf_trading['win_rate']:.1%}")
            print(f"   Total Fees: ${rf_trading['total_fees']:.2f}")
            
            print(f"\n SVM Model:")
            print(f"   Prediction Accuracy: {standard_results['svm']['accuracy']:.1%}")
            print(f"   Final Capital: ${svm_trading['final_capital']:.2f}")
            print(f"   Total Return: ${svm_trading['total_return']:+.2f} ({svm_trading['return_percentage']:+.2f}%)")
            print(f"   Buy & Hold Return: {svm_trading['buy_hold_return']:+.2f}%")
            print(f"   Alpha (Outperformance): {svm_trading['alpha']:+.2f}%")
            print(f"   Total Trades: {svm_trading['total_trades']}")
            print(f"   Win Rate: {svm_trading['win_rate']:.1%}")
            print(f"   Total Fees: ${svm_trading['total_fees']:.2f}")
        
        # Print time-range performance
        if time_results:
            print(f"\n TIME RANGE PERFORMANCE:")
            print(f"{'Time Range':<12}{'Model':<6}{'Accuracy':<10}{'Final $':<12}{'Return':<10}{'Alpha':<8}{'Trades':<8}")
            print("-" * 70)
            
            for time_range, metrics in time_results.items():
                # RF results
                rf_trade = metrics['rf_trading']
                print(f"{time_range:<12}{'RF':<6}{metrics['rf_accuracy']:.1%}{'':<4}"
                      f"${rf_trade['final_capital']:.0f}{'':<4}{rf_trade['return_percentage']:+.1f}%{'':<4}"
                      f"{rf_trade['alpha']:+.1f}%{'':<2}{rf_trade['total_trades']:<8}")
                
                # SVM results
                svm_trade = metrics['svm_trading']
                print(f"{'':<12}{'SVM':<6}{metrics['svm_accuracy']:.1%}{'':<4}"
                      f"${svm_trade['final_capital']:.0f}{'':<4}{svm_trade['return_percentage']:+.1f}%{'':<4}"
                      f"{svm_trade['alpha']:+.1f}%{'':<2}{svm_trade['total_trades']:<8}")
                print()
        
        return {
            'ticker': ticker,
            'method': method_name,
            'rf_accuracy': standard_results['rf']['accuracy'] if standard_results else None,
            'svm_accuracy': standard_results['svm']['accuracy'] if standard_results else None,
            'rf_trading': standard_results['rf']['trading'] if standard_results else None,
            'svm_trading': standard_results['svm']['trading'] if standard_results else None,
            'test_size': standard_results['test_size'] if standard_results else None,
            'train_size': standard_results['train_size'] if standard_results else None,
            'time_results': time_results
        }
        
    except Exception as e:
        print(f" Error analyzing {ticker}: {str(e)}")
        return None

def analyze_all_tickers():
    """Analyze all tickers with monetary performance tracking"""
    results = []
    
    print(" MACD PREDICTION TRADING SYSTEM")
    print("=" * 70)
    print(" Strategy: Machine Learning MACD Direction Prediction")
    print(" Models: Random Forest vs SVM")
    print(" Starting Capital: $10,000")
    print(" Transaction Cost: 0.1%")
    print("=" * 70)
    
    # Get all tickers
    all_tickers = []
    for group_name, tickers in TICKERS.items():
        all_tickers.extend(tickers)
    
    # Process each ticker
    for ticker in all_tickers:
        result = analyze_ticker(ticker, "macd_ml")
        if result:
            results.append(result)
    
    return results

def print_monetary_summary(results):
    """Print comprehensive monetary performance summary"""
    print("\n" + "=" * 100)
    print(" COMPREHENSIVE MONETARY PERFORMANCE SUMMARY")
    print("=" * 100)
    
    if not results:
        print(" No valid results to display")
        return
    
    # Print header
    print(f"{'Ticker':<8}{'Model':<6}{'Accuracy':<10}{'Final Capital':<15}{'Return':<10}{'Alpha':<8}{'Trades':<8}{'Win Rate':<10}")
    print("-" * 100)
    
    # Accumulators
    rf_total_capital = 0
    svm_total_capital = 0
    rf_total_return = 0
    svm_total_return = 0
    rf_total_alpha = 0
    svm_total_alpha = 0
    rf_total_trades = 0
    svm_total_trades = 0
    rf_wins = 0
    svm_wins = 0
    count = 0
    
    # Print results for each ticker
    for result in results:
        ticker = result['ticker']
        
        if result['rf_trading'] and result['svm_trading']:
            rf_trade = result['rf_trading']
            svm_trade = result['svm_trading']
            
            # RF results
            print(f"{ticker:<8}{'RF':<6}{result['rf_accuracy']:.1%}{'':<4}"
                  f"${rf_trade['final_capital']:.2f}{'':<3}{rf_trade['return_percentage']:+.1f}%{'':<4}"
                  f"{rf_trade['alpha']:+.1f}%{'':<2}{rf_trade['total_trades']:<8}{rf_trade['win_rate']:.1%}")
            
            # SVM results
            print(f"{'':<8}{'SVM':<6}{result['svm_accuracy']:.1%}{'':<4}"
                  f"${svm_trade['final_capital']:.2f}{'':<3}{svm_trade['return_percentage']:+.1f}%{'':<4}"
                  f"{svm_trade['alpha']:+.1f}%{'':<2}{svm_trade['total_trades']:<8}{svm_trade['win_rate']:.1%}")
            
            # Accumulate totals
            rf_total_capital += rf_trade['final_capital']
            svm_total_capital += svm_trade['final_capital']
            rf_total_return += rf_trade['return_percentage']
            svm_total_return += svm_trade['return_percentage']
            rf_total_alpha += rf_trade['alpha']
            svm_total_alpha += svm_trade['alpha']
            rf_total_trades += rf_trade['total_trades']
            svm_total_trades += svm_trade['total_trades']
            
            # Count wins
            if rf_trade['final_capital'] > svm_trade['final_capital']:
                rf_wins += 1
            elif svm_trade['final_capital'] > rf_trade['final_capital']:
                svm_wins += 1
            
            count += 1
            print()
    
    print("-" * 100)
    
    if count > 0:
        # Calculate averages
        avg_rf_capital = rf_total_capital / count
        avg_svm_capital = svm_total_capital / count
        avg_rf_return = rf_total_return / count
        avg_svm_return = svm_total_return / count
        avg_rf_alpha = rf_total_alpha / count
        avg_svm_alpha = svm_total_alpha / count
        avg_rf_trades = rf_total_trades / count
        avg_svm_trades = svm_total_trades / count
        
        print(f"{'AVERAGE':<8}{'RF':<6}{'':<10}${avg_rf_capital:.2f}{'':<3}{avg_rf_return:+.1f}%{'':<4}"
              f"{avg_rf_alpha:+.1f}%{'':<2}{avg_rf_trades:.1f}")
        print(f"{'AVERAGE':<8}{'SVM':<6}{'':<10}${avg_svm_capital:.2f}{'':<3}{avg_svm_return:+.1f}%{'':<4}"
              f"{avg_svm_alpha:+.1f}%{'':<2}{avg_svm_trades:.1f}")
        
        # Print comprehensive summary
        print("\n" + "=" * 80)
        print(" FINAL INVESTMENT ANALYSIS")
        print("=" * 80)
        
        print(f" Portfolio Performance (Starting with $10,000):")
        print(f"   Random Forest Average: ${avg_rf_capital:.2f} ({avg_rf_return:+.2f}%)")
        print(f"   SVM Average: ${avg_svm_capital:.2f} ({avg_svm_return:+.2f}%)")
        print(f"   Average Profit Difference: ${abs(avg_rf_capital - avg_svm_capital):.2f}")
        
        print(f"\n Model Performance:")
        print(f"   Random Forest wins: {rf_wins}/{count} ({rf_wins/count:.1%})")
        print(f"   SVM wins: {svm_wins}/{count} ({svm_wins/count:.1%})")
        print(f"   Ties: {count - rf_wins - svm_wins}")
        
        print(f"\n Alpha (Outperformance vs Buy & Hold):")
        print(f"   Random Forest: {avg_rf_alpha:+.2f}% average")
        print(f"   SVM: {avg_svm_alpha:+.2f}% average")
        
        # Determine best model
        if avg_rf_capital > avg_svm_capital:
            profit_advantage = avg_rf_capital - avg_svm_capital
            print(f"\n WINNER: Random Forest")
            print(f"   Average profit advantage: ${profit_advantage:.2f}")
            print(f"   Over {count} stocks, RF would make ${profit_advantage * count:.2f} more")
        elif avg_svm_capital > avg_rf_capital:
            profit_advantage = avg_svm_capital - avg_rf_capital
            print(f"\n WINNER: SVM")
            print(f"   Average profit advantage: ${profit_advantage:.2f}")
            print(f"   Over {count} stocks, SVM would make ${profit_advantage * count:.2f} more")
        else:
            print(f"\n RESULT: Models perform equally on average")
        
        # Total portfolio value if invested across all stocks
        total_investment = 10000 * count
        total_rf_value = rf_total_capital
        total_svm_value = svm_total_capital
        
        print(f"\n💼 Total Portfolio Analysis:")
        print(f"   If you invested $10,000 in each of {count} stocks (${total_investment:,.2f} total):")
        print(f"   Random Forest total value: ${total_rf_value:,.2f}")
        print(f"   SVM total value: ${total_svm_value:,.2f}")
        print(f"   Difference: ${abs(total_rf_value - total_svm_value):,.2f}")

def main():
    """Main function with enhanced monetary analysis"""
    print(" Starting enhanced MACD prediction analysis with monetary tracking...")
    print(f" Analyzing {sum(len(tickers) for tickers in TICKERS.values())} tickers")
    print(" Comparing Random Forest vs SVM with actual trading simulation\n")
    
    # Perform analysis
    results = analyze_all_tickers()
    print_monetary_summary(results)
    
    print("\n" + "=" * 60)
    print(" ANALYSIS COMPLETE")
    print("=" * 60)
    print(" Key Insights:")
    print("   - Both models simulate actual trading with $10,000 starting capital")
    print("   - Transaction costs (0.1%) are included in all calculations")
    print("   - Alpha shows outperformance vs simple buy-and-hold strategy")
    print("   - Time range analysis shows optimal trading periods")

if __name__ == "__main__":
    main()
