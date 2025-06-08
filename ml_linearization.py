import yfinance as yf # type: ignore
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from abc import ABC, abstractmethod
import warnings
warnings.filterwarnings('ignore')

TICKERS = {
    'large_cap': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META'],
    'mid_cap': ['DECK', 'WYNN', 'RCL', 'MGM', 'NCLH'],
    'small_cap': ['FIZZ', 'SMPL', 'BYND', 'PLUG', 'CLOV']
}

# DATA DOWNLOAD FUNCTION
def download_data_robust(ticker, period='7d', interval='1m', retries=3):
    """Download stock data with retry logic and basic cleaning"""
    for attempt in range(retries):
        try:
            data = yf.download(ticker, period=period, interval=interval, 
                              progress=False, auto_adjust=True, prepost=False)
            if data.empty:
                continue
            
            # Clean column names
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = [col[0].lower() if isinstance(col, tuple) else col.lower() 
                               for col in data.columns]
            else:
                data.columns = data.columns.str.lower()
            
            # Filter to regular trading hours if possible
            if isinstance(data.index, pd.DatetimeIndex):
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
    #transform raw price data into technical indicators
    def calculate_features(self, data: pd.DataFrame) -> pd.DataFrame:
        pass
    
    @abstractmethod
    #train a model and test it on out-of-sample data
    def walk_forward_predict(self, features: pd.DataFrame, train_test_ratio=0.7) -> tuple:
        pass

"""
MACD = momentum indicator that tracks the relationship 
between two moving averages of a stock's price

to find segments:
Bullish crossover: MACD > Signal (upward)
Bearish crossover: MACD < Signal (downward)

for each segment the following is calculated:
Trend: +1 (bullish) or -1 (bearish)
Slope: Linear regression slope of MACD during that segment
Intercept: Y-intercept of the linear fit

ML uses a Random Forest classifier with features such as:
slope: to see how steep the MACD trend is
intercept: starting value of the trend
trend: current bullish/bearish state

This works as MACD crossovers are classic buy/sell signals, 
and the slope indicates momentum strength
"""
class MACDMethod(PredictionMethod):
    def __init__(self, fast=12, slow=26, signal=9):
        self.fast = fast
        self.slow = slow
        self.signal = signal

    def calculate_features(self, data):
        close = data['close']
        ema_fast = close.ewm(span=self.fast, adjust=False).mean()
        ema_slow = close.ewm(span=self.slow, adjust=False).mean()
        macd = ema_fast - ema_slow
        signal = macd.ewm(span=self.signal, adjust=False).mean()
        
        segments = self._find_segments(macd, signal)
        features = pd.DataFrame({
            'macd': macd,
            'signal': signal,
            'trend': segments['trend'],
            'slope': segments['slope'],
            'intercept': segments['intercept'],
            'close': close
        }, index=data.index)
        
        return features

    def _find_segments(self, macd, signal):
        trend = np.zeros(len(macd))
        slope = np.zeros(len(macd))
        intercept = np.zeros(len(macd))
        
        current_trend = 0
        start_idx = 0
        
        for i in range(1, len(macd)):
            if (macd.iloc[i] > signal.iloc[i] and macd.iloc[i-1] <= signal.iloc[i-1]) or \
               (macd.iloc[i] < signal.iloc[i] and macd.iloc[i-1] >= signal.iloc[i-1]):
                
                x = np.arange(i - start_idx)
                y = macd.iloc[start_idx:i]
                if len(y) > 1:
                    s, intr = np.polyfit(x, y, 1)
                else:
                    s, intr = 0, y.iloc[0]
                
                trend[start_idx:i] = current_trend
                slope[start_idx:i] = s
                intercept[start_idx:i] = intr
                
                start_idx = i
                current_trend = 1 if macd.iloc[i] > signal.iloc[i] else -1
        
        if start_idx < len(macd):
            x = np.arange(len(macd) - start_idx)
            y = macd.iloc[start_idx:]
            if len(y) > 1:
                s, intr = np.polyfit(x, y, 1)
            else:
                s, intr = 0, y.iloc[0]
            
            trend[start_idx:] = current_trend
            slope[start_idx:] = s
            intercept[start_idx:] = intr
        
        return {'trend': trend, 'slope': slope, 'intercept': intercept}

    def walk_forward_predict(self, features, train_test_ratio=0.7):
        features['target'] = (features['close'].shift(-1) > features['close']).astype(int)
        features = features.dropna()
        
        split_idx = int(len(features) * train_test_ratio)
        train_features = features.iloc[:split_idx]
        test_features = features.iloc[split_idx:]
        
        if len(train_features) < 10 or len(test_features) < 5:
            return None, None, 0.0
        
        X_train = train_features[['slope', 'intercept', 'trend']]
        y_train = train_features['target']
        
        model = RandomForestClassifier(n_estimators=50, random_state=42)
        model.fit(X_train, y_train)
        
        X_test = test_features[['slope', 'intercept', 'trend']]
        y_test = test_features['target']
        
        pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, pred)
        
        return pred, y_test.values, accuracy

"""
Least Squares = fits a straight line through the last N price points using
linear regression, then uses the line's characteristics to predict future
direction

features:
slope: positive = uptrend, negative = downtrend
intercept: starting price of the fitted line
residual: sum of squared errors (shows how well the line fits)
trend: +1 if slope > 0, -1 if slope < 0

ML uses logistic regression with features:
slope: direction and strength of trend
residual: how "noisy" vs "smooth" the price movement is

This works because if prices are trending smoothly (low residuals)
in one direction (slope), they may continue that trend. 
"""
class LeastSquaresMethod(PredictionMethod):
    def __init__(self, window=20):
        self.window = window

    def calculate_features(self, data):
        close = data['close']
        slopes = np.zeros(len(close))
        intercepts = np.zeros(len(close))
        residuals = np.zeros(len(close))
        
        for i in range(self.window, len(close)):
            x = np.arange(self.window)
            y = close.iloc[i-self.window:i]
            slope, intercept = np.polyfit(x, y, 1)
            slopes[i] = slope
            intercepts[i] = intercept
            residuals[i] = np.sum((y - (slope * x + intercept))**2)
            
        return pd.DataFrame({
            'slope': slopes,
            'intercept': intercepts,
            'residual': residuals,
            'trend': np.where(slopes > 0, 1, -1),
            'close': close
        }, index=data.index)

    def walk_forward_predict(self, features, train_test_ratio=0.7):
        features['target'] = (features['close'].shift(-1) > features['close']).astype(int)
        features = features.dropna()
        
        split_idx = int(len(features) * train_test_ratio)
        train_features = features.iloc[:split_idx]
        test_features = features.iloc[split_idx:]
        
        if len(train_features) < 10 or len(test_features) < 5:
            return None, None, 0.0
        
        X_train = train_features[['slope', 'residual']]
        y_train = train_features['target']
        
        model = LogisticRegression(random_state=42)
        model.fit(X_train, y_train)
        
        X_test = test_features[['slope', 'residual']]
        y_test = test_features['target']
        
        pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, pred)
        
        return pred, y_test.values, accuracy

"""
RSI = Relative Strength Index, it measures the speed and magnitude of 
price changes, oscillating between 0-100 traditional interpretation:

RSI > 70: overbought (sell signal)
RSI < 30: oversold (buy signal)

rsi formula: 100 - (100 / (1 + relative strength))
relative strength = average gain/average loss

ML uses logistic regression with features:
rsi: current RSI value
trend: +1 if RSI > 50, -1 if RSI < 50

This works as RSI reversals often coincide with price reversals,
especially at extreme levels.
"""
class RSIMethod(PredictionMethod):
    def __init__(self, window=14, overbought=70, oversold=30):
        self.window = window
        self.overbought = overbought
        self.oversold = oversold

    def calculate_features(self, data):
        close = data['close']
        delta = close.diff()
        
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.rolling(self.window).mean()
        avg_loss = loss.rolling(self.window).mean()
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        signals = pd.Series(0, index=data.index)
        signals[(rsi > self.overbought) & (rsi.shift(1) <= self.overbought)] = -1
        signals[(rsi < self.oversold) & (rsi.shift(1) >= self.oversold)] = 1
        
        return pd.DataFrame({
            'rsi': rsi,
            'signal': signals,
            'trend': np.where(rsi > 50, 1, -1),
            'close': close
        })

    def walk_forward_predict(self, features, train_test_ratio=0.7):
        features['target'] = (features['close'].shift(-1) > features['close']).astype(int)
        features = features.dropna()
        
        split_idx = int(len(features) * train_test_ratio)
        train_features = features.iloc[:split_idx]
        test_features = features.iloc[split_idx:]
        
        if len(train_features) < 10 or len(test_features) < 5:
            return None, None, 0.0
        
        X_train = train_features[['rsi', 'trend']]
        y_train = train_features['target']
        
        model = LogisticRegression(random_state=42)
        model.fit(X_train, y_train)
        
        X_test = test_features[['rsi', 'trend']]
        y_test = test_features['target']
        
        pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, pred)
        
        return pred, y_test.values, accuracy

"""
Relative volume = compares current trading volume to historical 
average volume to identify unusual activity

relative volume formula: volume/average volume

features:
relative volume: how much higher/lower than normal volume is
trend: +1 if volume above average, -1 if below
spike: binary flag for extreme volume

ML uses logistic reasoning with features:
relative volume: magnitude of volume anomaly
trend: general volume trend

This works because unusual volume often precedes significant 
price moves. High volume can confirm price trends.
"""
class RelativeVolumeMethod(PredictionMethod):
    def __init__(self, window=20, threshold=2.0):
        self.window = window
        self.threshold = threshold

    def calculate_features(self, data):
        volume = data['volume']
        close = data['close']
        
        avg_volume = volume.rolling(self.window, min_periods=1).mean()
        rel_volume = volume / avg_volume
        spikes = (rel_volume > self.threshold).astype(int)
        
        return pd.DataFrame({
            'volume': volume,
            'avg_volume': avg_volume,
            'rel_volume': rel_volume,
            'spike': spikes,
            'trend': np.where(rel_volume > 1.0, 1, -1),
            'close': close
        })

    def walk_forward_predict(self, features, train_test_ratio=0.7):
        features['target'] = (features['close'].shift(-1) > features['close']).astype(int)
        features = features.dropna()
        
        split_idx = int(len(features) * train_test_ratio)
        train_features = features.iloc[:split_idx]
        test_features = features.iloc[split_idx:]
        
        if len(train_features) < 10 or len(test_features) < 5:
            return None, None, 0.0
        
        X_train = train_features[['rel_volume', 'trend']]
        y_train = train_features['target']
        
        model = LogisticRegression(random_state=42)
        model.fit(X_train, y_train)
        
        X_test = test_features[['rel_volume', 'trend']]
        y_test = test_features['target']
        
        pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, pred)
        
        return pred, y_test.values, accuracy

METHODS = {
    'macd': MACDMethod(),
    'least_squares': LeastSquaresMethod(),
    'rsi': RSIMethod(),
    'relative_volume': RelativeVolumeMethod()
}

def analyze_ticker(ticker, method_name):
    """Analyze a single ticker with walk-forward validation"""
    data = download_data_robust(ticker)
    if data is None:
        return None
    
    try:
        method = METHODS[method_name]
        features = method.calculate_features(data)
        pred, actual, accuracy = method.walk_forward_predict(features)
        
        if accuracy is None:
            return None
            
        return {
            'ticker': ticker,
            'method': method_name,
            'accuracy': accuracy,
            'test_size': len(actual) if actual is not None else 0,
            'train_size': len(features) - (len(actual) if actual is not None else 0)
        }
    except Exception as e:
        print(f"Error analyzing {ticker} with {method_name}: {str(e)}")
        return None

def analyze_all_tickers_all_methods():
    """Analyze all tickers with all methods using walk-forward validation"""
    results = {}
    
    print("STOCK PREDICTION ACCURACY ANALYSIS (WALK-FORWARD VALIDATION)")
    print("=" * 70)
    
    for method_name in METHODS.keys():
        results[method_name] = {}
    
    all_tickers = []
    for group_name, tickers in TICKERS.items():
        all_tickers.extend(tickers)
    
    total_analyses = len(all_tickers) * len(METHODS)
    current = 0
    
    for ticker in all_tickers:
        print(f"\nProcessing {ticker}...")
        
        for method_name in METHODS.keys():
            current += 1
            print(f"  [{current}/{total_analyses}] {method_name.upper()}...", end=" ")
            
            result = analyze_ticker(ticker, method_name)
            
            if result and result['accuracy'] is not None:
                results[method_name][ticker] = result
                print(f"✓ Train: {result['train_size']} Test: {result['test_size']} Acc: {result['accuracy']:.1%}")
            else:
                results[method_name][ticker] = None
                print("✗ Failed")
    
    return results

def print_accuracy_summary(results):
    """Print comprehensive accuracy summary with walk-forward validation results"""
    print("\n" + "=" * 80)
    print("WALK-FORWARD VALIDATION SUMMARY: ALL METHODS ACROSS ALL TICKERS")
    print("=" * 80)
    
    all_tickers = set()
    for method_results in results.values():
        all_tickers.update(method_results.keys())
    all_tickers = sorted(list(all_tickers))
    
    print(f"{'TICKER':<8}", end="")
    for method in METHODS.keys():
        print(f"{method.upper():<18}", end="")
    print("AVG")
    print("-" * 95)
    
    ticker_averages = {}
    for ticker in all_tickers:
        print(f"{ticker:<8}", end="")
        ticker_accuracies = []
        
        for method in METHODS.keys():
            result = results[method].get(ticker)
            if result and result['accuracy'] is not None:
                acc = result['accuracy']
                print(f"{acc:.1%} (T:{result['test_size']}) ", end="")
                ticker_accuracies.append(acc)
            else:
                print("FAILED            ", end="")
        
        if ticker_accuracies:
            avg_acc = sum(ticker_accuracies) / len(ticker_accuracies)
            ticker_averages[ticker] = avg_acc
            print(f"{avg_acc:.1%}")
        else:
            print("N/A")
    
    print("-" * 95)
    print(f"{'METHOD AVG':<8}", end="")
    method_averages = {}
    
    for method in METHODS.keys():
        accuracies = [r['accuracy'] for r in results[method].values() 
                     if r and r['accuracy'] is not None]
        test_sizes = [r['test_size'] for r in results[method].values()
                     if r and r['test_size'] is not None]
        
        if accuracies:
            avg = sum(accuracies) / len(accuracies)
            avg_test = sum(test_sizes) / len(test_sizes)
            method_averages[method] = avg
            print(f"{avg:.1%} (T:{avg_test:.0f}) ", end="")
        else:
            print("N/A               ", end="")
    
    if ticker_averages:
        overall_avg = sum(ticker_averages.values()) / len(ticker_averages)
        print(f"{overall_avg:.1%}")
    else:
        print("N/A")
    
    print("\n" + "=" * 50)
    print("BEST PERFORMING COMBINATIONS (OUT-OF-SAMPLE)")
    print("=" * 50)
    
    best_combinations = []
    for ticker in all_tickers:
        for method in METHODS.keys():
            result = results[method].get(ticker)
            if result and result['accuracy'] is not None:
                best_combinations.append((
                    result['accuracy'], 
                    result['test_size'],
                    ticker, 
                    method
                ))
    
    best_combinations.sort(key=lambda x: (-x[0], -x[1]))
    
    print("Top 10 Ticker-Method Combinations:")
    for i, (acc, test_size, ticker, method) in enumerate(best_combinations[:10], 1):
        print(f"{i:2d}. {ticker} + {method.upper():<15} = {acc:.1%} (Test size: {test_size})")
    
    if method_averages:
        print(f"\nBest Method Overall: {max(method_averages, key=method_averages.get).upper()} ({method_averages[max(method_averages, key=method_averages.get)]:.1%})")
    
    if ticker_averages:
        best_ticker = max(ticker_averages, key=ticker_averages.get)
        print(f"Best Ticker Overall: {best_ticker} ({ticker_averages[best_ticker]:.1%})")

def main():
    """Main execution function"""
    print("Starting walk-forward validation stock analysis...")
    print(f"Analyzing {sum(len(tickers) for tickers in TICKERS.values())} tickers")
    print(f"Using {len(METHODS)} methods: {', '.join(METHODS.keys())}")
    print()
    
    results = analyze_all_tickers_all_methods()
    print_accuracy_summary(results)
    
    print("\n" + "=" * 50)
    print("ANALYSIS COMPLETE")
    print("=" * 50)

if __name__ == "__main__":
    main()