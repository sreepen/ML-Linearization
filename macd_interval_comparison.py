# macd_interval_comparison.py
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

def download_data_robust(ticker, period='7d', interval='1m', retries=3, include_premarket=True):
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

class TimeRangeMACDPredictionMethod(MACDPredictionMethod):
    def __init__(self, time_ranges=None, **kwargs):
        """
        Initialize with specific time ranges to analyze
        
        Args:
            time_ranges (list): List of (start_hour, end_hour) tuples (24h format)
            kwargs: Arguments for parent MACDPredictionMethod
        """
        super().__init__(**kwargs)
        self.time_ranges = time_ranges or [
            (7, 9),     # Pre-Market (7am-9am)
            (7, 11),    # analysis period (7am - 11am)
            (9, 11),    # Early morning (9am-11am)
            (11, 13),   # Midday (11am-1pm)
            (13, 15),   # Early afternoon (1pm-3pm)
            (15, 16)    # Late afternoon (3pm-4pm)
        ]
    
    def filter_by_time(self, data, start_hour, end_hour):
        """Filter DataFrame to only include data within specified hours"""
        if start_hour < 9:
            return data[data.index.time < pd.to_datetime(f"{end_hour}:00").time()]
        
        else:
            return data.between_time(f"{start_hour}:00", f"{end_hour}:00")
    
    def walk_forward_predict(self, features, train_test_ratio=0.7):
        """Perform time-range specific walk-forward validation"""
        results = {}
        
        for start_hour, end_hour in self.time_ranges:
            # Filter data for this time range
            time_mask = (features.index.time >= pd.to_datetime(f"{start_hour}:00").time()) & \
                        (features.index.time <= pd.to_datetime(f"{end_hour}:00").time())
            time_features = features[time_mask]
            
            if len(time_features) < 20:  # Skip if not enough data
                continue
                
            # Split chronologically
            split_idx = int(len(time_features) * train_test_ratio)
            train = time_features.iloc[:split_idx]
            test = time_features.iloc[split_idx:]
            
            if len(train) < 10 or len(test) < 5:
                continue
            
            # Feature selection
            feature_cols = [col for col in features.columns if col != 'target']
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
            
            time_key = f"{start_hour:02d}:00-{end_hour:02d}:00"
            results[time_key] = {
                'rf_accuracy': rf_accuracy,
                'svm_accuracy': svm_accuracy,
                'test_size': len(y_test),
                'train_size': len(y_train),
                'feature_importances': rf.feature_importances_
            }
        
        return results if results else None

def analyze_ticker(ticker, method_name):
    """Analyze a ticker with both overall and time-range specific analysis"""
    data = download_data_robust(ticker, include_premarket=True)
    if data is None:
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
            return None
            
        # Print feature importances
        print(f"\nFeature Importances for {ticker}:")
        if standard_results:
            for col, imp in sorted(zip(features.columns[:-1], standard_results['feature_importances']), 
                                  key=lambda x: x[1], reverse=True):
                print(f"  {col:<15}: {imp:.2f}")
        
        # Print time-range specific results
        if time_results:
            print("\nTime Range Performance:")
            print(f"{'Time Range':<12}{'RF Acc':<10}{'SVM Acc':<10}{'Samples':<10}")
            for time_range, metrics in time_results.items():
                print(f"{time_range:<12}{metrics['rf_accuracy']:.1%}{'':<4}{metrics['svm_accuracy']:.1%}{'':<4}{metrics['test_size']:<10}")
        
        return {
            'ticker': ticker,
            'method': method_name,
            'rf_accuracy': standard_results['rf'][2] if standard_results else None,
            'svm_accuracy': standard_results['svm'][2] if standard_results else None,
            'test_size': standard_results['test_size'] if standard_results else None,
            'train_size': standard_results['train_size'] if standard_results else None,
            'time_results': time_results
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
            if result['rf_accuracy'] is not None:
                print(f"  Overall RF Accuracy: {result['rf_accuracy']:.1%}")
                print(f"  Overall SVM Accuracy: {result['svm_accuracy']:.1%}")
                print(f"  Overall Test Size: {result['test_size']}")
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
        if rf_acc is None or svm_acc is None:
            continue
            
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
            if result['rf_accuracy'] is None or result['svm_accuracy'] is None:
                continue
                
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

def print_time_range_summary(results):
    """Print detailed summary of performance by time range across all tickers"""
    if not results:
        return
    
    print("\n" + "=" * 80)
    print("TIME RANGE PERFORMANCE DETAILED SUMMARY")
    print("=" * 80)
    
    # Collect all time ranges from all results
    all_time_ranges = set()
    for result in results:
        if result.get('time_results'):
            all_time_ranges.update(result['time_results'].keys())
    
    if not all_time_ranges:
        print("No time range data available")
        return
    
    # Initialize accumulators for time range statistics
    time_stats = {
        tr: {
            'rf_total': 0,
            'svm_total': 0,
            'count': 0,
            'tickers': set(),
            'rf_best_count': 0,
            'svm_best_count': 0,
            'tie_count': 0
        }
        for tr in sorted(all_time_ranges)
    }
    
    # Initialize for best/worst performance tracking
    best_rf_time = {'range': None, 'accuracy': 0, 'count': 0}
    best_svm_time = {'range': None, 'accuracy': 0, 'count': 0}
    worst_rf_time = {'range': None, 'accuracy': 1, 'count': 0}
    worst_svm_time = {'range': None, 'accuracy': 1, 'count': 0}
    
    # Aggregate results
    for result in results:
        if not result.get('time_results'):
            continue
            
        ticker = result['ticker']
        for time_range, metrics in result['time_results'].items():
            time_stats[time_range]['rf_total'] += metrics['rf_accuracy']
            time_stats[time_range]['svm_total'] += metrics['svm_accuracy']
            time_stats[time_range]['count'] += 1
            time_stats[time_range]['tickers'].add(ticker)
            
            # Track which model performed better in this time range for this ticker
            if metrics['rf_accuracy'] > metrics['svm_accuracy']:
                time_stats[time_range]['rf_best_count'] += 1
            elif metrics['svm_accuracy'] > metrics['rf_accuracy']:
                time_stats[time_range]['svm_best_count'] += 1
            else:
                time_stats[time_range]['tie_count'] += 1
    
    # Print overall statistics header
    print("\nOVERALL TIME RANGE STATISTICS:")
    print(f"{'Time Range':<12}{'Avg RF Acc':<12}{'Avg SVM Acc':<12}{'Diff':<8}{'Tickers':<10}{'RF Wins':<8}{'SVM Wins':<8}{'Ties':<6}")
    print("-" * 80)
    
    # Print each time range's performance with additional statistics
    for time_range, stats in time_stats.items():
        if stats['count'] == 0:
            continue
            
        avg_rf = stats['rf_total'] / stats['count']
        avg_svm = stats['svm_total'] / stats['count']
        diff = avg_svm - avg_rf
        
        # Track best/worst performing time ranges
        if avg_rf > best_rf_time['accuracy'] or (avg_rf == best_rf_time['accuracy'] and stats['count'] > best_rf_time['count']):
            best_rf_time = {'range': time_range, 'accuracy': avg_rf, 'count': stats['count']}
        if avg_rf < worst_rf_time['accuracy'] or (avg_rf == worst_rf_time['accuracy'] and stats['count'] > worst_rf_time['count']):
            worst_rf_time = {'range': time_range, 'accuracy': avg_rf, 'count': stats['count']}
            
        if avg_svm > best_svm_time['accuracy'] or (avg_svm == best_svm_time['accuracy'] and stats['count'] > best_svm_time['count']):
            best_svm_time = {'range': time_range, 'accuracy': avg_svm, 'count': stats['count']}
        if avg_svm < worst_svm_time['accuracy'] or (avg_svm == worst_svm_time['accuracy'] and stats['count'] > worst_svm_time['count']):
            worst_svm_time = {'range': time_range, 'accuracy': avg_svm, 'count': stats['count']}
        
        print(f"{time_range:<12}{avg_rf:.1%}{'':<4}{avg_svm:.1%}{'':<4}{diff:+.2%}{'':<4}"
              f"{len(stats['tickers']):<10}{stats['rf_best_count']:<8}{stats['svm_best_count']:<8}{stats['tie_count']:<6}")
    
    # Calculate combined performance metric (average of RF and SVM accuracy)
    combined_performance = [
        (tr, (stats['rf_total'] + stats['svm_total']) / (2 * stats['count']) if stats['count'] > 0 else 0)
        for tr, stats in time_stats.items()
    ]
    best_combined = max(combined_performance, key=lambda x: x[1])
    worst_combined = min(combined_performance, key=lambda x: x[1])
    
    # Print performance extremes
    print("\nPERFORMANCE EXTREMES:")
    print(f"Best RF Time:    {best_rf_time['range']} (Accuracy: {best_rf_time['accuracy']:.1%}, Samples: {best_rf_time['count']})")
    print(f"Worst RF Time:   {worst_rf_time['range']} (Accuracy: {worst_rf_time['accuracy']:.1%}, Samples: {worst_rf_time['count']})")
    print(f"Best SVM Time:   {best_svm_time['range']} (Accuracy: {best_svm_time['accuracy']:.1%}, Samples: {best_svm_time['count']})")
    print(f"Worst SVM Time:  {worst_svm_time['range']} (Accuracy: {worst_svm_time['accuracy']:.1%}, Samples: {worst_svm_time['count']})")
    print(f"Best Combined:   {best_combined[0]} (Score: {best_combined[1]:.1%})")
    print(f"Worst Combined:  {worst_combined[0]} (Score: {worst_combined[1]:.1%})")
    
    # Print model dominance by time range
    print("\nMODEL DOMINANCE BY TIME RANGE:")
    for time_range, stats in time_stats.items():
        if stats['count'] == 0:
            continue
        
        dominance = ""
        total_comparisons = stats['rf_best_count'] + stats['svm_best_count'] + stats['tie_count']
        rf_percent = stats['rf_best_count'] / total_comparisons * 100
        svm_percent = stats['svm_best_count'] / total_comparisons * 100
        
        if rf_percent > 60:
            dominance = f"RF Dominant ({rf_percent:.0f}%)"
        elif svm_percent > 60:
            dominance = f"SVM Dominant ({svm_percent:.0f}%)"
        else:
            dominance = f"Balanced (RF: {rf_percent:.0f}%, SVM: {svm_percent:.0f}%)"
        
        print(f"{time_range:<12}: {dominance}")

def main():
    """Main function to execute the analysis"""
    print("Starting model comparison analysis...")
    print(f"Analyzing {sum(len(tickers) for tickers in TICKERS.values())} tickers")
    print("Comparing Random Forest vs SVM on MACD features with time range analysis\n")
    
    # Perform analysis and print results
    results = analyze_all_tickers()
    print_comparison_summary(results)
    print_time_range_summary(results)
    
    print("\n" + "=" * 50)
    print("ANALYSIS COMPLETE")
    print("=" * 50)

if __name__ == "__main__":
    main()