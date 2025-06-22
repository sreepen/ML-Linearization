# stock_prediction_with_profit.py
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

def download_data_robust(ticker, period='7d', interval='1m', retries=3):
    """Download stock data with retry logic and basic cleaning"""
    for attempt in range(retries):
        try:
            data = yf.download(ticker, period=period, interval=interval, 
                             progress=False, auto_adjust=True, prepost=False)
            if data.empty:
                continue
                
            #clean column names 
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = [col[0].lower() if isinstance(col, tuple) else col.lower() 
                               for col in data.columns]
            else:
                data.columns = data.columns.str.lower()
            
            #filter to regular trading hours (9:30am-4pm)
            if isinstance(data.index, pd.DatetimeIndex):
                try:
                    data = data.between_time('09:30', '16:00')
                except:
                    pass
            
            #only return if we have sufficient data
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
        
        # Create features
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
            'price': close,  # Add actual price for profit calculation
        }, index=data.index)
        
        # Target: Will price increase in next N periods?
        features['target'] = (close.shift(-self.forecast_window) > close).astype(int)
        
        return features.dropna()

    def walk_forward_predict(self, features, train_test_ratio=0.7):
        """Train models and calculate trading profits"""
        split_idx = int(len(features) * train_test_ratio)
        train = features.iloc[:split_idx]
        test = features.iloc[split_idx:]
        
        if len(train) < 30 or len(test) < 10:
            return None
        
        # Feature selection (excluding target and price)
        feature_cols = [col for col in features.columns if col not in ['target', 'price']]
        X_train = train[feature_cols]
        y_train = train['target']
        X_test = test[feature_cols]
        y_test = test['target']
        
        # Get test prices for profit calculation
        test_prices = test['price'].values
        
        # Random Forest Model
        rf = RandomForestClassifier(
            n_estimators=100,
            max_depth=5,
            min_samples_split=5,
            random_state=42
        )
        rf.fit(X_train, y_train)
        rf_pred = rf.predict(X_test)
        rf_accuracy = accuracy_score(y_test, rf_pred)
        
        # SVM Model
        svm = SVC(
            kernel='rbf',
            C=1.0,
            gamma='scale',
            probability=True,
            random_state=42
        )
        svm.fit(X_train, y_train)
        svm_pred = svm.predict(X_test)
        svm_accuracy = accuracy_score(y_test, svm_pred)
        
        # Calculate trading profits
        rf_profits = self.calculate_trading_profit(rf_pred, test_prices, y_test.values)
        svm_profits = self.calculate_trading_profit(svm_pred, test_prices, y_test.values)
        buy_hold_profit = self.calculate_buy_hold_profit(test_prices)
        
        return {
            'rf': (rf_pred, y_test.values, rf_accuracy),
            'svm': (svm_pred, y_test.values, svm_accuracy),
            'rf_profits': rf_profits,
            'svm_profits': svm_profits,
            'buy_hold_profit': buy_hold_profit,
            'test_size': len(y_test),
            'train_size': len(y_train),
            'feature_importances': rf.feature_importances_,
            'test_prices': test_prices
        }
    
    def calculate_trading_profit(self, predictions, prices, actual_targets, 
                               initial_capital=10000, transaction_cost=0.001):
        """
        Calculate profit from trading based on model predictions
        
        Args:
            predictions: Model predictions (1=buy, 0=don't buy)
            prices: Stock prices during test period
            actual_targets: Actual price movements (for reference)
            initial_capital: Starting money
            transaction_cost: Trading fee as percentage (0.001 = 0.1%)
        
        Returns:
            dict: Trading results including total profit, number of trades, etc.
        """
        capital = initial_capital
        shares = 0
        trades = 0
        profitable_trades = 0
        total_fees = 0
        
        position_open = False
        buy_price = 0
        
        trade_log = []
        
        for i in range(len(predictions) - 1):  # -1 because we need next price
            current_price = prices[i]
            next_price = prices[i + 1] if i + 1 < len(prices) else current_price
            
            # If model predicts price will go up (1) and we're not already in position
            if predictions[i] == 1 and not position_open:
                # BUY
                shares_to_buy = capital / current_price
                fee = capital * transaction_cost
                shares = shares_to_buy
                buy_price = current_price
                capital = 0  # All money is now in stock
                total_fees += fee
                position_open = True
                trades += 1
                
                trade_log.append({
                    'action': 'BUY',
                    'price': current_price,
                    'shares': shares,
                    'fee': fee
                })
            
            # If model predicts price will go down (0) and we have a position, SELL
            elif predictions[i] == 0 and position_open:
                # SELL
                sell_value = shares * current_price
                fee = sell_value * transaction_cost
                capital = sell_value - fee
                total_fees += fee
                
                # Check if trade was profitable
                if current_price > buy_price:
                    profitable_trades += 1
                
                trade_log.append({
                    'action': 'SELL',
                    'price': current_price,
                    'shares': shares,
                    'fee': fee,
                    'profit': sell_value - (shares * buy_price) - fee
                })
                
                shares = 0
                position_open = False
        
        # Close any remaining position at the end
        if position_open and len(prices) > 0:
            final_price = prices[-1]
            sell_value = shares * final_price
            fee = sell_value * transaction_cost
            capital = sell_value - fee
            total_fees += fee
            
            if final_price > buy_price:
                profitable_trades += 1
            
            trade_log.append({
                'action': 'SELL (Final)',
                'price': final_price,
                'shares': shares,
                'fee': fee,
                'profit': sell_value - (shares * buy_price) - fee
            })
        
        total_return = capital - initial_capital
        return_percentage = (total_return / initial_capital) * 100
        
        return {
            'initial_capital': initial_capital,
            'final_capital': capital,
            'total_return': total_return,
            'return_percentage': return_percentage,
            'total_trades': trades,
            'profitable_trades': profitable_trades,
            'win_rate': profitable_trades / trades if trades > 0 else 0,
            'total_fees': total_fees,
            'trade_log': trade_log[-5:]  # Keep last 5 trades for review
        }
    
    def calculate_buy_hold_profit(self, prices, initial_capital=10000):
        """Calculate profit from simple buy and hold strategy"""
        if len(prices) < 2:
            return {'return_percentage': 0, 'total_return': 0}
        
        initial_price = prices[0]
        final_price = prices[-1]
        
        shares = initial_capital / initial_price
        final_value = shares * final_price
        total_return = final_value - initial_capital
        return_percentage = (total_return / initial_capital) * 100
        
        return {
            'initial_capital': initial_capital,
            'final_capital': final_value,
            'total_return': total_return,
            'return_percentage': return_percentage
        }

def analyze_ticker(ticker, method_name):
    """Analyze ticker with profit calculations"""
    data = download_data_robust(ticker)
    if data is None:
        return None
    
    try:
        method = MACDPredictionMethod()
        features = method.calculate_features(data)
        results = method.walk_forward_predict(features)
        
        if results is None:
            return None
            
        return {
            'ticker': ticker,
            'method': method_name,
            'rf_accuracy': results['rf'][2],
            'svm_accuracy': results['svm'][2],
            'rf_profit': results['rf_profits']['return_percentage'],
            'svm_profit': results['svm_profits']['return_percentage'],
            'buy_hold_profit': results['buy_hold_profit']['return_percentage'],
            'rf_trades': results['rf_profits']['total_trades'],
            'svm_trades': results['svm_profits']['total_trades'],
            'rf_win_rate': results['rf_profits']['win_rate'],
            'svm_win_rate': results['svm_profits']['win_rate'],
            'test_size': results['test_size'],
            'train_size': results['train_size'],
            'rf_trade_log': results['rf_profits']['trade_log'],
            'svm_trade_log': results['svm_profits']['trade_log']
        }
    except Exception as e:
        print(f"Error analyzing {ticker}: {str(e)}")
        return None

def analyze_all_tickers():
    """Analyze all tickers with profit calculations"""
    results = []
    
    print("STOCK PREDICTION MODEL COMPARISON WITH PROFIT ANALYSIS")
    print("=" * 80)
    print("Starting capital: $10,000 per strategy")
    print("Transaction cost: 0.1% per trade")
    print("=" * 80)
    
    all_tickers = []
    for group_name, tickers in TICKERS.items():
        all_tickers.extend(tickers)
    
    for ticker in all_tickers:
        print(f"\nProcessing {ticker}...")
        result = analyze_ticker(ticker, "macd")
        
        if result:
            results.append(result)
            print(f"  RF: {result['rf_accuracy']:.1%} accuracy, {result['rf_profit']:+.1f}% profit, {result['rf_trades']} trades")
            print(f"  SVM: {result['svm_accuracy']:.1%} accuracy, {result['svm_profit']:+.1f}% profit, {result['svm_trades']} trades")
            print(f"  Buy & Hold: {result['buy_hold_profit']:+.1f}% profit")
        else:
            print(f"  Failed to analyze {ticker}")
    
    return results

def print_profit_summary(results):
    """Print detailed profit analysis summary"""
    print("\n" + "=" * 100)
    print("PROFIT ANALYSIS SUMMARY")
    print("=" * 100)
    
    if not results:
        print("No valid results to display")
        return
    
    # Print header
    print(f"{'Ticker':<8}{'RF Profit':<12}{'SVM Profit':<12}{'Buy&Hold':<12}{'RF Trades':<10}{'SVM Trades':<11}{'Best Strategy':<15}")
    print("-" * 100)
    
    rf_total_profit = 0
    svm_total_profit = 0
    buy_hold_total = 0
    count = 0
    
    rf_wins = 0
    svm_wins = 0
    buy_hold_wins = 0
    
    for result in results:
        ticker = result['ticker']
        rf_profit = result['rf_profit']
        svm_profit = result['svm_profit']
        bh_profit = result['buy_hold_profit']
        rf_trades = result['rf_trades']
        svm_trades = result['svm_trades']
        
        # Determine best strategy for this ticker
        best_profit = max(rf_profit, svm_profit, bh_profit)
        if rf_profit == best_profit:
            best_strategy = "Random Forest"
            rf_wins += 1
        elif svm_profit == best_profit:
            best_strategy = "SVM"
            svm_wins += 1
        else:
            best_strategy = "Buy & Hold"
            buy_hold_wins += 1
        
        print(f"{ticker:<8}{rf_profit:+.1f}%{'':<5}{svm_profit:+.1f}%{'':<5}{bh_profit:+.1f}%{'':<5}{rf_trades:<10}{svm_trades:<11}{best_strategy:<15}")
        
        rf_total_profit += rf_profit
        svm_total_profit += svm_profit
        buy_hold_total += bh_profit
        count += 1
    
    print("-" * 100)
    if count > 0:
        avg_rf = rf_total_profit / count
        avg_svm = svm_total_profit / count
        avg_bh = buy_hold_total / count
        
        print(f"{'Average':<8}{avg_rf:+.1f}%{'':<5}{avg_svm:+.1f}%{'':<5}{avg_bh:+.1f}%")
        
        print("\n" + "=" * 60)
        print("STRATEGY PERFORMANCE SUMMARY")
        print("=" * 60)
        print(f"Random Forest wins: {rf_wins} stocks")
        print(f"SVM wins: {svm_wins} stocks")
        print(f"Buy & Hold wins: {buy_hold_wins} stocks")
        
        print(f"\nAverage Returns:")
        print(f"Random Forest: {avg_rf:+.2f}%")
        print(f"SVM: {avg_svm:+.2f}%")
        print(f"Buy & Hold: {avg_bh:+.2f}%")
        
        # Money analysis
        print(f"\nIf you invested $10,000 in each strategy:")
        print(f"Random Forest would give you: ${10000 * (1 + avg_rf/100):,.2f}")
        print(f"SVM would give you: ${10000 * (1 + avg_svm/100):,.2f}")
        print(f"Buy & Hold would give you: ${10000 * (1 + avg_bh/100):,.2f}")
        
        # Are you making money?
        print("\n" + "="*50)
        print("ARE YOU MAKING MONEY?")
        print("="*50)
        
        profitable_rf = sum(1 for r in results if r['rf_profit'] > 0)
        profitable_svm = sum(1 for r in results if r['svm_profit'] > 0)
        profitable_bh = sum(1 for r in results if r['buy_hold_profit'] > 0)
        
        print(f"Random Forest: {profitable_rf}/{count} stocks profitable ({profitable_rf/count*100:.1f}%)")
        print(f"SVM: {profitable_svm}/{count} stocks profitable ({profitable_svm/count*100:.1f}%)")
        print(f"Buy & Hold: {profitable_bh}/{count} stocks profitable ({profitable_bh/count*100:.1f}%)")
        
        if avg_rf > 0:
            print(f"\nRandom Forest: YES, making {avg_rf:.2f}% on average")
        else:
            print(f"\nRandom Forest: NO, losing {abs(avg_rf):.2f}% on average")
            
        if avg_svm > 0:
            print(f"SVM: YES, making {avg_svm:.2f}% on average")
        else:
            print(f"SVM: NO, losing {abs(avg_svm):.2f}% on average")
            
        if avg_bh > 0:
            print(f"Buy & Hold: YES, making {avg_bh:.2f}% on average")
        else:
            print(f"Buy & Hold: NO, losing {abs(avg_bh):.2f}% on average")

def main():
    """Main function to execute the profit analysis"""
    print("Starting stock prediction profit analysis...")
    print(f"Analyzing {sum(len(tickers) for tickers in TICKERS.values())} tickers")
    print("Comparing Random Forest vs SVM vs Buy & Hold strategies\n")
    
    results = analyze_all_tickers()
    print_profit_summary(results)
    
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)
    print("\nNote: Results are based on short-term intraday data.")
    print("Past performance does not guarantee future results.")
    print("This is for educational purposes only, not financial advice.")

if __name__ == "__main__":
    main()