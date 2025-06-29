# enhanced_stock_prediction_with_trailing_stop.py
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
        """Train models and calculate trading profits with multiple strategies"""
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
        
        # Calculate trading profits with different strategies
        rf_profits_basic = self.calculate_trading_profit(rf_pred, test_prices, y_test.values)
        svm_profits_basic = self.calculate_trading_profit(svm_pred, test_prices, y_test.values)
        
        # Calculate with trailing stop loss
        rf_profits_trailing = self.calculate_trading_profit_with_trailing_stop(
            rf_pred, test_prices, y_test.values, stop_loss_pct=0.05)
        svm_profits_trailing = self.calculate_trading_profit_with_trailing_stop(
            svm_pred, test_prices, y_test.values, stop_loss_pct=0.05)
        
        buy_hold_profit = self.calculate_buy_hold_profit(test_prices)
        
        return {
            'rf': (rf_pred, y_test.values, rf_accuracy),
            'svm': (svm_pred, y_test.values, svm_accuracy),
            'rf_profits_basic': rf_profits_basic,
            'svm_profits_basic': svm_profits_basic,
            'rf_profits_trailing': rf_profits_trailing,
            'svm_profits_trailing': svm_profits_trailing,
            'buy_hold_profit': buy_hold_profit,
            'test_size': len(y_test),
            'train_size': len(y_train),
            'feature_importances': rf.feature_importances_,
            'test_prices': test_prices
        }
    
    def calculate_trading_profit(self, predictions, prices, actual_targets, 
                               initial_capital=10000, transaction_cost=0.001):
        """Original trading profit calculation without stop loss"""
        capital = initial_capital
        shares = 0
        trades = 0
        profitable_trades = 0
        total_fees = 0
        
        position_open = False
        buy_price = 0
        
        trade_log = []
        loss_amounts = []  # Track individual loss amounts
        
        for i in range(len(predictions) - 1):
            current_price = prices[i]
            next_price = prices[i + 1] if i + 1 < len(prices) else current_price
            
            if predictions[i] == 1 and not position_open:
                # BUY
                shares_to_buy = capital / current_price
                fee = capital * transaction_cost
                shares = shares_to_buy
                buy_price = current_price
                capital = 0
                total_fees += fee
                position_open = True
                trades += 1
                
                trade_log.append({
                    'action': 'BUY',
                    'price': current_price,
                    'shares': shares,
                    'fee': fee
                })
            
            elif predictions[i] == 0 and position_open:
                # SELL
                sell_value = shares * current_price
                fee = sell_value * transaction_cost
                capital = sell_value - fee
                total_fees += fee
                
                # Calculate profit/loss for this trade
                trade_profit = sell_value - (shares * buy_price) - fee
                
                if current_price > buy_price:
                    profitable_trades += 1
                else:
                    # Track loss amounts
                    loss_amounts.append(abs(trade_profit))
                
                trade_log.append({
                    'action': 'SELL',
                    'price': current_price,
                    'shares': shares,
                    'fee': fee,
                    'profit': trade_profit
                })
                
                shares = 0
                position_open = False
        
        # Close any remaining position
        if position_open and len(prices) > 0:
            final_price = prices[-1]
            sell_value = shares * final_price
            fee = sell_value * transaction_cost
            capital = sell_value - fee
            total_fees += fee
            
            trade_profit = sell_value - (shares * buy_price) - fee
            
            if final_price > buy_price:
                profitable_trades += 1
            else:
                loss_amounts.append(abs(trade_profit))
            
            trade_log.append({
                'action': 'SELL (Final)',
                'price': final_price,
                'shares': shares,
                'fee': fee,
                'profit': trade_profit
            })
        
        total_return = capital - initial_capital
        return_percentage = (total_return / initial_capital) * 100
        
        # Analyze loss patterns
        avg_loss = np.mean(loss_amounts) if loss_amounts else 0
        max_loss = max(loss_amounts) if loss_amounts else 0
        
        return {
            'initial_capital': initial_capital,
            'final_capital': capital,
            'total_return': total_return,
            'return_percentage': return_percentage,
            'total_trades': trades,
            'profitable_trades': profitable_trades,
            'win_rate': profitable_trades / trades if trades > 0 else 0,
            'total_fees': total_fees,
            'trade_log': trade_log[-5:],
            'loss_amounts': loss_amounts,
            'avg_loss': avg_loss,
            'max_loss': max_loss,
            'strategy': 'Basic'
        }
    
    def calculate_trading_profit_with_trailing_stop(self, predictions, prices, actual_targets, 
                                                  initial_capital=10000, transaction_cost=0.001,
                                                  stop_loss_pct=0.05):
        """
        Calculate trading profit with trailing stop loss protection
        
        Args:
            stop_loss_pct: Maximum allowed loss from peak (e.g., 0.05 = 5% trailing stop)
        """
        capital = initial_capital
        shares = 0
        trades = 0
        profitable_trades = 0
        total_fees = 0
        
        position_open = False
        buy_price = 0
        highest_price_since_buy = 0  # Track the highest price since buying
        
        trade_log = []
        loss_amounts = []
        stop_loss_triggers = 0
        
        for i in range(len(predictions) - 1):
            current_price = prices[i]
            
            if predictions[i] == 1 and not position_open:
                # BUY
                shares_to_buy = capital / current_price
                fee = capital * transaction_cost
                shares = shares_to_buy
                buy_price = current_price
                highest_price_since_buy = current_price  # Initialize trailing stop
                capital = 0
                total_fees += fee
                position_open = True
                trades += 1
                
                trade_log.append({
                    'action': 'BUY',
                    'price': current_price,
                    'shares': shares,
                    'fee': fee
                })
            
            elif position_open:
                # Update trailing stop
                if current_price > highest_price_since_buy:
                    highest_price_since_buy = current_price
                
                # Calculate trailing stop price
                trailing_stop_price = highest_price_since_buy * (1 - stop_loss_pct)
                
                # Check for trailing stop trigger OR model sell signal
                should_sell = (predictions[i] == 0) or (current_price <= trailing_stop_price)
                
                if should_sell:
                    # SELL
                    sell_value = shares * current_price
                    fee = sell_value * transaction_cost
                    capital = sell_value - fee
                    total_fees += fee
                    
                    trade_profit = sell_value - (shares * buy_price) - fee
                    
                    # Determine reason for sale
                    if current_price <= trailing_stop_price:
                        stop_loss_triggers += 1
                        sell_reason = "TRAILING STOP"
                    else:
                        sell_reason = "MODEL SIGNAL"
                    
                    if current_price > buy_price:
                        profitable_trades += 1
                    else:
                        loss_amounts.append(abs(trade_profit))
                    
                    trade_log.append({
                        'action': f'SELL ({sell_reason})',
                        'price': current_price,
                        'shares': shares,
                        'fee': fee,
                        'profit': trade_profit,
                        'highest_since_buy': highest_price_since_buy,
                        'stop_price': trailing_stop_price
                    })
                    
                    shares = 0
                    position_open = False
                    highest_price_since_buy = 0
        
        # Close any remaining position
        if position_open and len(prices) > 0:
            final_price = prices[-1]
            sell_value = shares * final_price
            fee = sell_value * transaction_cost
            capital = sell_value - fee
            total_fees += fee
            
            trade_profit = sell_value - (shares * buy_price) - fee
            
            if final_price > buy_price:
                profitable_trades += 1
            else:
                loss_amounts.append(abs(trade_profit))
            
            trade_log.append({
                'action': 'SELL (Final)',
                'price': final_price,
                'shares': shares,
                'fee': fee,
                'profit': trade_profit
            })
        
        total_return = capital - initial_capital
        return_percentage = (total_return / initial_capital) * 100
        
        # Analyze loss patterns
        avg_loss = np.mean(loss_amounts) if loss_amounts else 0
        max_loss = max(loss_amounts) if loss_amounts else 0
        
        return {
            'initial_capital': initial_capital,
            'final_capital': capital,
            'total_return': total_return,
            'return_percentage': return_percentage,
            'total_trades': trades,
            'profitable_trades': profitable_trades,
            'win_rate': profitable_trades / trades if trades > 0 else 0,
            'total_fees': total_fees,
            'trade_log': trade_log[-5:],
            'loss_amounts': loss_amounts,
            'avg_loss': avg_loss,
            'max_loss': max_loss,
            'stop_loss_triggers': stop_loss_triggers,
            'stop_loss_pct': stop_loss_pct * 100,
            'strategy': f'Trailing Stop ({stop_loss_pct*100:.1f}%)'
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
            'return_percentage': return_percentage,
            'strategy': 'Buy & Hold'
        }

def analyze_ticker(ticker, method_name):
    """Analyze ticker with multiple trading strategies"""
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
            
            # Basic strategy results
            'rf_profit_basic': results['rf_profits_basic']['return_percentage'],
            'svm_profit_basic': results['svm_profits_basic']['return_percentage'],
            'rf_trades_basic': results['rf_profits_basic']['total_trades'],
            'svm_trades_basic': results['svm_profits_basic']['total_trades'],
            'rf_win_rate_basic': results['rf_profits_basic']['win_rate'],
            'svm_win_rate_basic': results['svm_profits_basic']['win_rate'],
            'rf_max_loss_basic': results['rf_profits_basic']['max_loss'],
            'svm_max_loss_basic': results['svm_profits_basic']['max_loss'],
            
            # Trailing stop results
            'rf_profit_trailing': results['rf_profits_trailing']['return_percentage'],
            'svm_profit_trailing': results['svm_profits_trailing']['return_percentage'],
            'rf_trades_trailing': results['rf_profits_trailing']['total_trades'],
            'svm_trades_trailing': results['svm_profits_trailing']['total_trades'],
            'rf_win_rate_trailing': results['rf_profits_trailing']['win_rate'],
            'svm_win_rate_trailing': results['svm_profits_trailing']['win_rate'],
            'rf_max_loss_trailing': results['rf_profits_trailing']['max_loss'],
            'svm_max_loss_trailing': results['svm_profits_trailing']['max_loss'],
            'rf_stop_triggers': results['rf_profits_trailing']['stop_loss_triggers'],
            'svm_stop_triggers': results['svm_profits_trailing']['stop_loss_triggers'],
            
            'buy_hold_profit': results['buy_hold_profit']['return_percentage'],
            'test_size': results['test_size'],
            'train_size': results['train_size']
        }
    except Exception as e:
        print(f"Error analyzing {ticker}: {str(e)}")
        return None

def analyze_all_tickers():
    """Analyze all tickers with win/loss analysis"""
    results = []
    
    print("ENHANCED STOCK PREDICTION WITH TRAILING STOP LOSS ANALYSIS")
    print("=" * 80)
    print("Starting capital: $10,000 per strategy")
    print("Transaction cost: 0.1% per trade")
    print("Trailing stop loss: 5% from peak")
    print("=" * 80)
    
    all_tickers = []
    for group_name, tickers in TICKERS.items():
        all_tickers.extend(tickers)
    
    for ticker in all_tickers:
        print(f"\nProcessing {ticker}...")
        result = analyze_ticker(ticker, "macd")
        
        if result:
            results.append(result)
            print(f"  RF Basic: {result['rf_profit_basic']:+.1f}% | RF Trailing: {result['rf_profit_trailing']:+.1f}%")
            print(f"  SVM Basic: {result['svm_profit_basic']:+.1f}% | SVM Trailing: {result['svm_profit_trailing']:+.1f}%")
            print(f"  Buy & Hold: {result['buy_hold_profit']:+.1f}%")
        else:
            print(f"  Failed to analyze {ticker}")
    
    return results

def print_comprehensive_analysis(results):
    """Print comprehensive win/loss and trailing stop analysis"""
    print("\n" + "=" * 120)
    print("COMPREHENSIVE WIN/LOSS AND TRAILING STOP ANALYSIS")
    print("=" * 120)
    
    if not results:
        print("No valid results to display")
        return
    
    # Print comparison table
    print(f"{'Ticker':<8}{'RF Basic':<10}{'RF Trail':<10}{'SVM Basic':<11}{'SVM Trail':<11}{'Buy&Hold':<10}{'Best Strategy':<20}")
    print("-" * 120)
    
    strategy_wins = {
        'RF Basic': 0, 'RF Trailing': 0, 'SVM Basic': 0, 
        'SVM Trailing': 0, 'Buy & Hold': 0
    }
    
    total_profits = {
        'RF Basic': 0, 'RF Trailing': 0, 'SVM Basic': 0,
        'SVM Trailing': 0, 'Buy & Hold': 0
    }
    
    improvement_count = {'RF': 0, 'SVM': 0}
    total_improvement = {'RF': 0, 'SVM': 0}
    
    for result in results:
        ticker = result['ticker']
        profits = {
            'RF Basic': result['rf_profit_basic'],
            'RF Trailing': result['rf_profit_trailing'],
            'SVM Basic': result['svm_profit_basic'],
            'SVM Trailing': result['svm_profit_trailing'],
            'Buy & Hold': result['buy_hold_profit']
        }
        
        # Find best strategy
        best_strategy = max(profits.keys(), key=lambda k: profits[k])
        strategy_wins[best_strategy] += 1
        
        # Track improvements from trailing stop
        rf_improvement = result['rf_profit_trailing'] - result['rf_profit_basic']
        svm_improvement = result['svm_profit_trailing'] - result['svm_profit_basic']
        
        if rf_improvement > 0:
            improvement_count['RF'] += 1
        if svm_improvement > 0:
            improvement_count['SVM'] += 1
        
        total_improvement['RF'] += rf_improvement
        total_improvement['SVM'] += svm_improvement
        
        # Add to totals
        for strategy, profit in profits.items():
            total_profits[strategy] += profit
        
        print(f"{ticker:<8}{profits['RF Basic']:+.1f}%{'':<3}{profits['RF Trailing']:+.1f}%{'':<3}"
              f"{profits['SVM Basic']:+.1f}%{'':<3}{profits['SVM Trailing']:+.1f}%{'':<3}"
              f"{profits['Buy & Hold']:+.1f}%{'':<3}{best_strategy:<20}")
    
    print("-" * 120)
    
    count = len(results)
    if count > 0:
        # Calculate averages
        avg_profits = {strategy: total / count for strategy, total in total_profits.items()}
        
        print(f"{'Average':<8}{avg_profits['RF Basic']:+.1f}%{'':<3}{avg_profits['RF Trailing']:+.1f}%{'':<3}"
              f"{avg_profits['SVM Basic']:+.1f}%{'':<3}{avg_profits['SVM Trailing']:+.1f}%{'':<3}"
              f"{avg_profits['Buy & Hold']:+.1f}%")
        
        print("\n" + "=" * 80)
        print("TRAILING STOP LOSS IMPACT ANALYSIS")
        print("=" * 80)
        
        rf_avg_improvement = total_improvement['RF'] / count
        svm_avg_improvement = total_improvement['SVM'] / count
        
        print(f"Random Forest:")
        print(f"  Improved by trailing stop: {improvement_count['RF']}/{count} stocks ({improvement_count['RF']/count*100:.1f}%)")
        print(f"  Average improvement: {rf_avg_improvement:+.2f}%")
        print(f"  Basic average: {avg_profits['RF Basic']:+.2f}%")
        print(f"  Trailing average: {avg_profits['RF Trailing']:+.2f}%")
        
        print(f"\nSVM:")
        print(f"  Improved by trailing stop: {improvement_count['SVM']}/{count} stocks ({improvement_count['SVM']/count*100:.1f}%)")
        print(f"  Average improvement: {svm_avg_improvement:+.2f}%")
        print(f"  Basic average: {avg_profits['SVM Basic']:+.2f}%")
        print(f"  Trailing average: {avg_profits['SVM Trailing']:+.2f}%")
        
        print("\n" + "=" * 80)
        print("STRATEGY PERFORMANCE RANKING")
        print("=" * 80)
        
        # Rank strategies by average performance
        ranked_strategies = sorted(avg_profits.items(), key=lambda x: x[1], reverse=True)
        
        for i, (strategy, avg_profit) in enumerate(ranked_strategies, 1):
            wins = strategy_wins[strategy]
            print(f"{i}. {strategy:<15}: {avg_profit:+.2f}% average, {wins} individual wins")
        
        print("\n" + "=" * 80)
        print("LOSS PROTECTION ANALYSIS")
        print("=" * 80)
        
        # Analyze max losses
        print("Maximum single trade losses (before vs after trailing stop):")
        for result in results:
            if result['rf_max_loss_basic'] > 0 or result['svm_max_loss_basic'] > 0:
                print(f"\n{result['ticker']}:")
                if result['rf_max_loss_basic'] > 0:
                    reduction = result['rf_max_loss_basic'] - result['rf_max_loss_trailing']
                    print(f"  RF: ${result['rf_max_loss_basic']:.0f} → ${result['rf_max_loss_trailing']:.0f} "
                          f"(saved ${reduction:.0f})")
                if result['svm_max_loss_basic'] > 0:
                    reduction = result['svm_max_loss_basic'] - result['svm_max_loss_trailing']
                    print(f"  SVM: ${result['svm_max_loss_basic']:.0f} → ${result['svm_max_loss_trailing']:.0f} "
                          f"(saved ${reduction:.0f})")
        
        print("\n" + "=" * 80)
        print("FINAL RECOMMENDATION")
        print("=" * 80)
        
        best_overall = ranked_strategies[0]
        print(f"Best performing strategy: {best_overall[0]} with {best_overall[1]:+.2f}% average return")
        
        if rf_avg_improvement > 0:
            print(f"✓ Trailing stop HELPS Random Forest (avg +{rf_avg_improvement:.2f}%)")
        else:
            print(f"✗ Trailing stop HURTS Random Forest (avg {rf_avg_improvement:.2f}%)")
            
        if svm_avg_improvement > 0:
            print(f"✓ Trailing stop HELPS SVM (avg +{svm_avg_improvement:.2f}%)")
        else:
            print(f"✗ Trailing stop HURTS SVM (avg {svm_avg_improvement:.2f}%)")
            
        # Money impact
        print(f"\nIf you invested $10,000 in the best strategy ({best_overall[0]}):")
        print(f"You would have: ${10000 * (1 + best_overall[1]/100):,.2f}")
        print(f"Profit/Loss: ${10000 * best_overall[1]/100:+,.2f}")

def main():
    """Main function to execute the comprehensive analysis"""
    print("Starting enhanced stock prediction analysis with trailing stop loss...")
    print(f"Analyzing {sum(len(tickers) for tickers in TICKERS.values())} tickers")
    print("Comparing: Basic vs Trailing Stop strategies for RF and SVM vs Buy & Hold\n")
    
    results = analyze_all_tickers()
    print_comprehensive_analysis(results)
    
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print("\nKey Insights:")
    print("• Trailing stop loss limits maximum loss per trade to ~5% from peak")
    print("• Compare 'Basic' vs 'Trailing' columns to see protection impact")
    print("• Results show if algorithmic protection beats raw model predictions")
    print("\nNote: Results based on short-term intraday data.")
    print("Past performance does not guarantee future results.")
    print("This is for educational purposes only, not financial advice.")

if __name__ == "__main__":
    main()