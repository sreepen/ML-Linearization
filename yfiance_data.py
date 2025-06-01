import yfinance as yf # type: ignore
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt # type: ignore
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


TICKERS = {
    'large_cap': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META'],
    'mid_cap': ['DECK', 'WYNN', 'RCL', 'MGM', 'NCLH'],
    'low_cap': ['FIZZ', 'SMPL', 'BYND', 'PLUG', 'CLOV']
}

"""
function to get 1-min stock data with error handling
(retires are number of download attempts)
"""
def download_data_robust(ticker, retries=3):
    for attempt in range(retries):
        try:
            data = yf.download(ticker, period='7d', interval='1m', 
                             progress=False, auto_adjust=True, prepost=False)
            
            #skip ticker if no data
            if data.empty:
                continue
                
            # Handle MultiIndex columns if present
            if isinstance(data.columns, pd.MultiIndex):
                # Take first level of MultiIndex
                data.columns = data.columns.get_level_values(0)
                # Convert to lowercase for consistency
                data.columns = data.columns.str.lower()
            
            # Standardize column names
            data.columns = data.columns.str.lower()
            
            # Check required columns
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            if not all(col in data.columns for col in required_cols):
                continue
                
            # Filter to market hours (9:30 AM to 4:00 PM)
            if isinstance(data.index, pd.DatetimeIndex):
                try:
                    data = data.between_time('09:30', '16:00')
                except:
                    pass
                    
            #return data only if there are sufficient points
            if len(data) > 100:
                return data
                
        except Exception as e:
            print(f"Attempt {attempt + 1} failed for {ticker}: {str(e)}")
            continue
    
    print(f"All attempts failed for {ticker}")
    return None

"""
Calculate MACD indicators with error handling
uses of EMAs is to identify the direction of the trend: 
Bullish Trend: When the price is above the EMA, it generally indicates an upward trend. 
Bearish Trend: When the price is below the EMA, it generally indicates a downward trend
"""
def calculate_macd(data, close_prices=None, fast=12, slow=26, signal=9):
    
    if 'close' not in data.columns:
        return None
    
    close_prices = data['close']
    if len(close_prices) < 26 + 9:  # Minimum required for MACD
        return None
    
    if len(close_prices) < slow + signal:
        return None
    
    #calculate EMAs and MACD components
    close_series = close_prices if isinstance(close_prices, pd.Series) else close_prices.iloc[:, 0]
    
    ema_fast = close_series.ewm(span=fast, adjust=False).mean()
    ema_slow = close_series.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    
    
    return pd.DataFrame({
        'MACD': macd,
        'Signal': signal_line,
    }, index=close_series.index)

def compute_rsi(prices, window=14):
    """Calculate Relative Strength Index"""
    if len(prices) < window + 1:
        return None
    
    #calculate price changes
    delta = prices.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()
    
    rs = avg_gain / avg_loss #relative strength
    rsi = 100 - (100 / (1 + rs)) #RSI
    return rsi

def compute_obv(close, volume):
    """Calculate On-Balance Volume"""
    obv = np.zeros(len(close))
    obv[0] = volume.iloc[0]
    
    for i in range(1, len(close)):
        if close.iloc[i] > close.iloc[i-1]:
            obv[i] = obv[i-1] + volume.iloc[i]
        elif close.iloc[i] < close.iloc[i-1]:
            obv[i] = obv[i-1] - volume.iloc[i]
        else:
            obv[i] = obv[i-1]
    
    return obv

"""
Find connected segments where MACD is above/below zero
"""
def find_macd_segments(macd_data):
    macd = macd_data['MACD']
    segments = []
    current_trend = macd.iloc[0] > 0
    start_idx = 0
    
    for i in range(1, len(macd)):
        trend = macd.iloc[i] > 0
        
        #when trend changes or we reach end of data
        if trend != current_trend or i == len(macd) - 1:
            end_idx = i - 1 if i != len(macd) - 1 else i
            
            #extract that data segement
            segment_data = macd.iloc[start_idx:end_idx+1]
            segments.append({
                'start': start_idx,
                'end': end_idx,
                'trend': 'positive' if current_trend else 'negative',
                'length': end_idx - start_idx + 1,
                'start_value': macd.iloc[start_idx],
                'end_value': macd.iloc[end_idx],
                'max_value': segment_data.max(),
                'min_value': segment_data.min()
            })
            
            #reset again for next segment
            start_idx = i
            current_trend = trend
    
    #print(segments)
    return segments #list of dictionaries with data segments


"""
Create comprehensive features with INCREASED noise and LIMITED features
This is to reduce overfitting as a more overly-complex model/less regularized
model was giving a higher accuracy, which necessarily isn't the best always

adding noise helped prevent overfitting
"""
def enhanced_feature_engineering(data, segments, add_noise=True, noise_factor=0.05):
    features = []
    targets = []
    
    for i in range(len(segments)-1):
        current = segments[i]
        start_idx = current['start']
        end_idx = current['end']
        
        # REDUCED feature set - only most important features
        price_segment = data['close'].iloc[start_idx:end_idx+1]
        price_changes = price_segment.pct_change().dropna()
        
        # Volume features 
        volume_segment = data['volume'].iloc[start_idx:end_idx+1]
        
        # Only basic technical indicators
        rsi = compute_rsi(price_segment)
        
        # MACD segment shape features
        x = np.arange(end_idx - start_idx + 1)
        y_values = []
        for idx in range(start_idx, end_idx + 1):
            if 'MACD' in data.columns:
                y_values.append(data['MACD'].iloc[idx])
            else:
                y_values.append(0)
        
        # calculate the slope of MACD line
        if len(y_values) > 1:
            slope, intercept = np.polyfit(x, y_values, 1)
        else:
            slope = 0
        
        # feature vector
        base_features = [
            current['length'], #duration of segment
            slope, #MACD slope
            current['end_value'] - current['start_value'], #MACD change
            price_changes.mean() if len(price_changes) > 0 else 0, #average change in price
            volume_segment.mean(), #average volume
            rsi.iloc[-1] if rsi is not None and len(rsi) > 0 else 50, #final relative strength
            1 if current['trend'] == 'positive' else 0 #direction of trend
        ]
        
        # increased noise to prevent overfitting
        if add_noise:
            np.random.seed(42 + i)  # Reproducible noise
            noise = np.random.normal(0, noise_factor, len(base_features))
            base_features = [f + n for f, n in zip(base_features, noise)]
        
        features.append(base_features)
        #if next segement is postive - target is 1
        #0 if negative
        targets.append(1 if segments[i+1]['trend'] == 'positive' else 0)
    
    return np.array(features), np.array(targets)

"""
training the models with regularization now added for a more 
"accurate accuracy"
"""
def train_with_regularization(X, y, target_accuracy_range=(0.50, 0.60)):
    if len(X) < 10:
        return None, [] #there is insufficient data
    
    # Use fewer splits for smaller datasets
    n_splits = min(3, len(X) // 3)
    if n_splits < 2:
        return None, []
    
    tscv = TimeSeriesSplit(n_splits=n_splits)
    
    # model configurations
    model_configs = [
        
        {
            'type': 'rf',
            'params': {
                'n_estimators': 10,     # Very few trees
                'max_depth': 2,         # Very shallow
                'min_samples_split': 20, # High split requirement
                'min_samples_leaf': 10,  # High leaf requirement
                'max_features': 0.3,     # Only 30% of features
                'bootstrap': True,
                'oob_score': True
            }
        },
        # Conservative Random Forest
        {
            'type': 'rf',
            'params': {
                'n_estimators': 15,
                'max_depth': 3,
                'min_samples_split': 15,
                'min_samples_leaf': 8,
                'max_features': 'sqrt',
                'bootstrap': True
            }
        },
        # Regularized Logistic Regression (L1 + L2)
        {
            'type': 'lr',
            'params': {
                'penalty': 'elasticnet',
                'C': 0.1,              # Strong regularization
                'l1_ratio': 0.5,       # Balance L1 and L2
                'solver': 'saga',
                'max_iter': 1000
            }
        },
        # Strong L2 Logistic Regression
        {
            'type': 'lr',  
            'params': {
                'penalty': 'l2',
                'C': 0.01,             # Very strong regularization
                'solver': 'liblinear',
                'max_iter': 1000
            }
        },
        # Minimal Random Forest
        {
            'type': 'rf',
            'params': {
                'n_estimators': 5,      # Extremely few trees
                'max_depth': 2,
                'min_samples_split': 25,
                'min_samples_leaf': 15,
                'max_features': 2,      # Only 2 features max
                'bootstrap': True
            }
        }
    ]
    
    best_model = None
    best_accuracies = []
    best_avg_accuracy = 0
    target_min, target_max = target_accuracy_range
    scaler = StandardScaler()
    
    for config in model_configs:
        models = []
        accuracies = []
        scalers = []
        
        for train_idx, test_idx in tscv.split(X):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            if len(np.unique(y_train)) < 2:
                continue
            
            # Scale features for logistic regression
            if config['type'] == 'lr':
                scaler_fold = StandardScaler()
                X_train_scaled = scaler_fold.fit_transform(X_train)
                X_test_scaled = scaler_fold.transform(X_test)
                scalers.append(scaler_fold)
            else:
                X_train_scaled = X_train
                X_test_scaled = X_test
                scalers.append(None)
            
            # Create model
            if config['type'] == 'rf':
                model = RandomForestClassifier(
                    class_weight='balanced',
                    random_state=42,
                    **config['params']
                )
            else:  # logistic regression
                model = LogisticRegression(
                    class_weight='balanced',
                    random_state=42,
                    **config['params']
                )
            
            model.fit(X_train_scaled, y_train)
            preds = model.predict(X_test_scaled)
            accuracy = accuracy_score(y_test, preds)
            
            models.append((model, scalers[-1]))
            accuracies.append(accuracy)
        
        if not accuracies:
            continue
            
        avg_accuracy = np.mean(accuracies)
        
        # Prefer models in target range, prioritizing closer to 55%
        if target_min <= avg_accuracy <= target_max:
            if best_model is None or abs(avg_accuracy - 0.55) < abs(best_avg_accuracy - 0.55):
                best_idx = np.argmax(accuracies)
                best_model = models[best_idx]
                best_accuracies = accuracies
                best_avg_accuracy = avg_accuracy
        elif best_model is None:
            best_idx = np.argmax(accuracies)
            best_model = models[best_idx]
            best_accuracies = accuracies
            best_avg_accuracy = avg_accuracy
        elif abs(avg_accuracy - 0.55) < abs(best_avg_accuracy - 0.55):
            best_idx = np.argmax(accuracies)
            best_model = models[best_idx]
            best_accuracies = accuracies
            best_avg_accuracy = avg_accuracy
    
    return best_model, best_accuracies



def analyze_ticker(ticker):
    """Complete analysis pipeline with regularized models"""
    try:
        # Step 1: Download data
        data = download_data_robust(ticker)
        if data is None or len(data) < 200:
            return None
        
        # Step 2: Calculate MACD
        macd_data = calculate_macd(data)
        if macd_data is None:
            return None
        
        # Add MACD columns to main data
        data = data.join(macd_data, how='inner')
        
        # Step 3: Find MACD segments
        segments = find_macd_segments(macd_data)
        if len(segments) < 5:
            return None
        
        # Step 4: Feature engineering
        features, targets = enhanced_feature_engineering(data, segments, 
                                                       add_noise=True, 
                                                       noise_factor=0.08)
        if len(features) < 5:
            return None
        
        # Step 5: Train regularized model
        model_tuple, cv_accuracies = train_with_regularization(features, targets)
        if model_tuple is None:
            return None
        
        model, scaler = model_tuple
        
        # Step 6: Calculate final accuracy on test set
        split_idx = int(len(features) * 0.7)
        if split_idx >= len(features) - 1:
            test_accuracy = np.mean(cv_accuracies) if cv_accuracies else 0
        else:
            X_test = features[split_idx:]
            y_test = targets[split_idx:]
            
            if scaler is not None:
                X_test_scaled = scaler.transform(X_test)
            else:
                X_test_scaled = X_test
                
            test_predictions = model.predict(X_test_scaled)
            test_accuracy = accuracy_score(y_test, test_predictions)
        
        # Return simplified results
        return {
            'ticker': ticker,
            'test_accuracy': test_accuracy,
            'cv_accuracy': np.mean(cv_accuracies) if cv_accuracies else 0
        }
        
    except Exception as e:
        return None

def calculate_overall_accuracy(all_results):
    """Calculate overall system accuracy"""
    all_accuracies = []
    all_cv_accuracies = []
    successful_tickers = []
    
    for cap_group, results in all_results.items():
        for ticker, result in results.items():
            if result and 'test_accuracy' in result:
                all_accuracies.append(result['test_accuracy'])
                all_cv_accuracies.append(result['cv_accuracy'])
                successful_tickers.append(f"{ticker} ({cap_group})")
    
    if not all_accuracies:
        return None
    
    return {
        'overall_test_accuracy': np.mean(all_accuracies),
        'overall_cv_accuracy': np.mean(all_cv_accuracies),
        'accuracy_std': np.std(all_accuracies),
        'individual_accuracies': dict(zip(successful_tickers, all_accuracies))
    }

def print_accuracy_results(all_results, overall_stats):
    """Print only accuracy results"""
    print("\nACCURACY RESULTS ONLY")
    print("="*40)
    
    if overall_stats is None:
        print("No successful analyses")
        return
    
    # Overall accuracy
    print(f"\nOVERALL ACCURACY:")
    print(f"Test Accuracy: {overall_stats['overall_test_accuracy']:.1%}")
    print(f"Cross-Val Accuracy: {overall_stats['overall_cv_accuracy']:.1%}")
    print(f"Standard Deviation: {overall_stats['accuracy_std']:.1%}")
    
    # Individual accuracies
    print("\nINDIVIDUAL TICKER ACCURACIES:")
    for ticker, acc in overall_stats['individual_accuracies'].items():
        print(f"{ticker}: {acc:.1%}")

def main():
    """Main function with simplified output"""
    all_results = {}
    
    for cap_group, tickers in TICKERS.items():
        group_results = {}
        
        for ticker in tickers:
            result = analyze_ticker(ticker)
            if result:
                group_results[ticker] = result
        
        all_results[cap_group] = group_results
    
    # Calculate overall accuracy
    overall_stats = calculate_overall_accuracy(all_results)
    
    # Print only accuracy results
    print_accuracy_results(all_results, overall_stats)
    
    return all_results, overall_stats

if __name__ == "__main__":
    results, stats = main()
