# stock_prediction_comparison.py
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from abc import ABC, abstractmethod
import warnings
warnings.filterwarnings('ignore')  # Suppress warnings for cleaner output

# Dictionary of tickers categorized by market capitalization
TICKERS = {
    'large_cap': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META'],  # Large capitalization stocks
    'mid_cap': ['DECK', 'WYNN', 'RCL', 'MGM', 'NCLH'],       # Medium capitalization stocks
    'small_cap': ['FIZZ', 'SMPL', 'BYND', 'PLUG', 'CLOV']    # Small capitalization stocks
}

def download_data_robust(ticker, period='7d', interval='1m', retries=3):
    """
    Download stock data with retry logic and basic cleaning
    
    Args:
        ticker (str): Stock symbol to download
        period (str): Time period to download (default: '7d' for 7 days)
        interval (str): Data frequency (default: '1m' for 1-minute intervals)
        retries (int): Number of download attempts
        
    Returns:
        pd.DataFrame: Cleaned stock data or None if download fails
    """
    for attempt in range(retries):
        try:
            data = yf.download(ticker, period=period, interval=interval, 
                             progress=False, auto_adjust=True, prepost=False)
            if data.empty:
                continue
                
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = [col[0].lower() if isinstance(col, tuple) else col.lower() 
                               for col in data.columns]
            else:
                data.columns = data.columns.str.lower()
            
            if isinstance(data.index, pd.DatetimeIndex):
                try:
                    data = data.between_time('09:30', '16:00')
                except:
                    pass
            
            if len(data) > 100:
                return data
        except Exception as e:
            continue
    return None  # Return None if all retries fail

# Abstract base class defining the interface for prediction methods
class PredictionMethod(ABC):
    @abstractmethod
    def calculate_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate features from raw stock data"""
        pass
    
    @abstractmethod
    def walk_forward_predict(self, features: pd.DataFrame, train_test_ratio=0.7) -> dict:
        """Perform walk-forward validation and return predictions"""
        pass

# Implementation of improved MACD prediction method
class MACDPredictionMethod(PredictionMethod):
    def __init__(self, fast=12, slow=26, signal=9, forecast_window=2):
        self.fast = fast
        self.slow = slow
        self.signal = signal
        self.forecast_window = forecast_window  # How many periods ahead to predict

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
            'macd_lag1': macd.shift(1),  # Previous MACD value
            'macd_lag2': macd.shift(2),   # Two periods back
            'macd_lag3': macd.shift(3),   # Three periods back
            'signal_line': signal_line,
            'histogram': histogram,
            'macd_slope': macd.diff(),    # Instantaneous slope
            'signal_slope': signal_line.diff(),
            'histogram_slope': histogram.diff(),
            'above_zero': (macd > 0).astype(int),
            'zero_crossing': ((macd * macd.shift(1)) < 0).astype(int),  # Changed sign
            'high_volume': (volume > volume.rolling(20).mean() * 1.5).astype(int),
            'volatility': close.rolling(14).std()
        }, index=data.index)
        
        # Target: Will MACD increase in next N periods? (1=yes, 0=no)
        features['target'] = (macd.shift(-self.forecast_window) > macd).astype(int)
        
        return features.dropna()

    def walk_forward_predict(self, features, train_test_ratio=0.7):
        """Train models to predict future MACD direction"""
        # Split chronologically
        split_idx = int(len(features) * train_test_ratio)
        train = features.iloc[:split_idx]
        test = features.iloc[split_idx:]
        
        if len(train) < 30 or len(test) < 10:
            return None
        
        # Feature selection (excluding target)
        feature_cols = [col for col in features.columns if col != 'target']
        X_train = train[feature_cols]
        y_train = train['target']
        X_test = test[feature_cols]
        y_test = test['target']
        
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
        
        return {
            'rf': (rf_pred, y_test.values, rf_accuracy),
            'svm': (svm_pred, y_test.values, svm_accuracy),
            'test_size': len(y_test),
            'train_size': len(y_train),
            'feature_importances': rf.feature_importances_  # Show which features mattered most
        }

def analyze_ticker(ticker, method_name):
    """(Modified for MACD prediction)"""
    data = download_data_robust(ticker)
    if data is None:
        return None
    
    try:
        method = MACDPredictionMethod()
        features = method.calculate_features(data)
        results = method.walk_forward_predict(features)
        
        if results is None:
            return None
            
        # Print feature importances
        print(f"\nFeature Importances for {ticker}:")
        for col, imp in sorted(zip(features.columns[:-1], results['feature_importances']), 
                              key=lambda x: x[1], reverse=True):
            print(f"  {col:<15}: {imp:.2f}")
        
        return {
            'ticker': ticker,
            'method': method_name,
            'rf_accuracy': results['rf'][2],
            'svm_accuracy': results['svm'][2],
            'test_size': results['test_size'],
            'train_size': results['train_size']
        }
    except Exception as e:
        print(f"Error analyzing {ticker}: {str(e)}")
        return None


def analyze_all_tickers():
    """
    Analyze all tickers defined in the TICKERS dictionary
    
    Returns:
        list: List of analysis results for all successful analyses
    """
    results = []
    
    print("STOCK PREDICTION MODEL COMPARISON (Random Forest vs SVM)")
    print("=" * 70)
    
    # Get all tickers from all categories
    all_tickers = []
    for group_name, tickers in TICKERS.items():
        all_tickers.extend(tickers)
    
    # Process each ticker
    for ticker in all_tickers:
        print(f"\nProcessing {ticker}...")
        result = analyze_ticker(ticker, "macd")
        
        if result:
            results.append(result)
            print(f"  RF Accuracy: {result['rf_accuracy']:.1%}")
            print(f"  SVM Accuracy: {result['svm_accuracy']:.1%}")
            print(f"  Train Size: {result['train_size']}  Test Size: {result['test_size']}")
        else:
            print(f"  Failed to analyze {ticker}")
    
    return results

def print_comparison_summary(results):
    """
    Print a formatted summary of the model comparison results
    
    Args:
        results (list): List of analysis results from analyze_all_tickers()
    """
    print("\n" + "=" * 80)
    print("MODEL COMPARISON SUMMARY")
    print("=" * 80)
    
    if not results:
        print("No valid results to display")
        return
    
    # Print header
    print(f"{'Ticker':<8}{'RF Accuracy':<15}{'SVM Accuracy':<15}{'Difference':<15}{'Test Size':<10}")
    print("-" * 60)
    
    rf_total = 0  # Accumulator for RF accuracy
    svm_total = 0  # Accumulator for SVM accuracy
    count = 0      # Count of valid results
    
    # Print results for each ticker
    for result in results:
        ticker = result['ticker']
        rf_acc = result['rf_accuracy']
        svm_acc = result['svm_accuracy']
        diff = svm_acc - rf_acc  # Difference in accuracy
        test_size = result['test_size']
        
        print(f"{ticker:<8}{rf_acc:.1%}{'':<7}{svm_acc:.1%}{'':<7}{diff:+.2%}{'':<7}{test_size:<10}")
        
        rf_total += rf_acc
        svm_total += svm_acc
        count += 1
    
    print("-" * 60)
    if count > 0:
        # Calculate and print averages
        avg_rf = rf_total / count
        avg_svm = svm_total / count
        avg_diff = avg_svm - avg_rf
        
        print(f"{'Average':<8}{avg_rf:.1%}{'':<7}{avg_svm:.1%}{'':<7}{avg_diff:+.2%}")
        
        # Determine which model performed better
        print("\n" + "=" * 50)
        print("BEST PERFORMING MODEL BY TICKER")
        print("=" * 50)
        
        rf_wins = 0  # Count of tickers where RF was better
        svm_wins = 0  # Count of tickers where SVM was better
        ties = 0      # Count of ties
        
        for result in results:
            if result['rf_accuracy'] > result['svm_accuracy']:
                rf_wins += 1
            elif result['svm_accuracy'] > result['rf_accuracy']:
                svm_wins += 1
            else:
                ties += 1
        
        print(f"Random Forest wins: {rf_wins}")
        print(f"SVM wins: {svm_wins}")
        print(f"Ties: {ties}")
        
        # Print overall conclusion
        if rf_wins > svm_wins:
            print("\nOverall best model: Random Forest")
        elif svm_wins > rf_wins:
            print("\nOverall best model: SVM")
        else:
            print("\nModels are equally effective on average")

def main():
    """Main function to execute the analysis"""
    print("Starting model comparison analysis...")
    print(f"Analyzing {sum(len(tickers) for tickers in TICKERS.values())} tickers")
    print("Comparing Random Forest vs SVM on MACD features\n")
    
    # Perform analysis and print results
    results = analyze_all_tickers()
    print_comparison_summary(results)
    
    print("\n" + "=" * 50)
    print("ANALYSIS COMPLETE")
    print("=" * 50)

if __name__ == "__main__":
    main()  # Run the main function when executed directly