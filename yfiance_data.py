import yfinance as yf # type: ignore
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pytz
import warnings
from concurrent.futures import ThreadPoolExecutor
warnings.filterwarnings('ignore')


TICKERS_BY_CAP = {
    'large_cap': ['MSFT', 'AAPL', 'GOOG', 'AMZN', 'META'],
    'mid_cap': ['DECK', 'MTCH', 'WYNN', 'CZR', 'MGM'],
    'low_cap': ['FIZZ', 'SMPL', 'LANC', 'NATR', 'SATS']
}

class StockAccuracyEvaluator:
    def __init__(self):
        self.models = {}
        self.predictions = {}
        self.metrics = {}
    
    def create_features(self, data, lookback_windows=[5, 10, 20]):
        df = data.copy()
        df['Returns'] = df['Close'].pct_change()
        df['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1))

        for window in lookback_windows:
            df[f'MA_{window}'] = df['Close'].rolling(window=window).mean()
            df[f'MA_Ratio_{window}'] = df['Close'] / df[f'MA_{window}']
            df[f'Volatility_{window}'] = df['Returns'].rolling(window=window).std()

        df['RSI'] = self.calculate_rsi(df['Close'])
        df['BB_Upper'], df['BB_Lower'] = self.calculate_bollinger_bands(df['Close'])
        df['BB_Position'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])

        df['Volume_MA'] = df['Volume'].rolling(window=20).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_MA']
        df['High_Low_Ratio'] = df['High'] / df['Low']
        df['Open_Close_Ratio'] = df['Open'] / df['Close']

        for lag in [1, 2, 3, 5]:
            df[f'Close_Lag_{lag}'] = df['Close'].shift(lag)
            df[f'Returns_Lag_{lag}'] = df['Returns'].shift(lag)
            df[f'Volume_Lag_{lag}'] = df['Volume'].shift(lag)
        return df.dropna()

    def calculate_rsi(self, prices, window=14):
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain/loss
        return 100 - (100/(1+rs))

    def calculate_bollinger_bands(self, prices, window=20, num_std=2):
        ma = prices.rolling(window=window).mean()
        std = prices.rolling(window=window).std()
        upper = ma + (std*num_std)
        lower = ma - (std * num_std)
        return upper, lower

    def prepare_ml_data(self, df, target_column='Close', prediction_horizon=1):
        df[f'Target_{prediction_horizon}'] = df[target_column].shift(-prediction_horizon)
        feature_cols = [col for col in df.columns if col not in [
            f'Target_{prediction_horizon}', 'Dividends', 'Stock Splits'
        ] and df[col].dtype in ['float64', 'int64']]

        clean_df = df[feature_cols + [f'Target_{prediction_horizon}']].dropna()
        X = clean_df[feature_cols]
        y = clean_df[f'Target_{prediction_horizon}']
        return X, y, clean_df.index
    
    def create_baseline_predictions(self, train_data, val_data, test_data):
        baselines = {}
        if len(train_data) == 0:
            return baselines

        last_close = train_data['Close'].dropna().iloc[-1] if 'Close' in train_data.columns else np.nan

        baselines['naive'] = {
            'train': np.full(len(train_data), last_close) if len(train_data) > 0 else [],
            'val': np.full(len(val_data), last_close) if len(val_data) > 0 else [],
            'test': np.full(len(test_data), last_close) if len(test_data) > 0 else []
        }

        if len(train_data) >= 20:
            ma_20 = train_data['Close'].rolling(window=20).mean().dropna().iloc[-1]
        else:
            ma_20 = last_close
        
        baselines['moving_average'] = {
            'train': np.full(len(train_data), ma_20) if len(train_data) > 0 else [],
            'val': np.full(len(val_data), ma_20) if len(val_data) > 0 else [],
            'test': np.full(len(test_data), ma_20) if len(test_data) > 0 else []
        }

        if len(train_data) >= 2:
            train_prices = train_data['Close'].dropna().values
            time_index = np.arange(len(train_prices))
            slope = np.polyfit(time_index, train_prices, 1)[0]
            last_price = train_prices[-1]

            baselines['linear_trend'] = {
                'train': [last_price + slope*i for i in range(len(train_data))] if len(train_data) > 0 else [],
                'val': [last_price + slope * (len(train_data)+i) for i in range(len(val_data))] if len(val_data) > 0 else [],
                'test': [last_price + slope * (len(train_data) + len(val_data)+i) for i in range(len(test_data))] if len(test_data) > 0 else []
            }
        else:
            baselines['linear_trend'] = {
                'train': np.full(len(train_data), last_close) if len(train_data) > 0 else [],
                'val': np.full(len(val_data), last_close) if len(val_data) > 0 else [],
                'test': np.full(len(test_data), last_close) if len(test_data) > 0 else []
            }
        return baselines
    
    def calculate_metrics(self, y_true, y_pred, set_name=""):
        if len(y_true) == 0 or len(y_pred) == 0 or len(y_true) != len(y_pred):
            return {}
        
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        mask = np.isfinite(y_true) & np.isfinite(y_pred)
        y_true = y_true[mask]
        y_pred = y_pred[mask]

        if len(y_true) == 0 or len(y_pred) == 0:
            return {}
        
        metrics = {}
        metrics['MSE'] = mean_squared_error(y_true, y_pred)
        metrics['RMSE'] = np.sqrt(metrics['MSE'])
        metrics['MAE'] = mean_absolute_error(y_true, y_pred)
        metrics['R2'] = r2_score(y_true, y_pred)
        metrics['MAPE'] = np.mean(np.abs((y_true - y_pred)/y_true)) * 100
        metrics['MPE'] = np.mean((y_true - y_pred)/y_true) * 100

        if len(y_true) > 1:
            true_direction = np.diff(y_true) > 0
            pred_direction = np.diff(y_pred) > 0
            metrics['Direction_Accuracy'] = np.mean(true_direction == pred_direction) * 100
        
        mean_price = np.mean(y_true)
        metrics['RMSE_Percentage'] = (metrics['RMSE']/mean_price)*100
        metrics['MAE_Percentage'] = (metrics['MAE']/mean_price)*100

        return metrics
    
    def walk_forward_validation(self, data, model, n_splits=5):
        tscv = TimeSeriesSplit(n_splits=n_splits)
        X, y, timestamps = self.prepare_ml_data(data)
        wf_scores = [] 

        for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            fold_metrics = self.calculate_metrics(y_test, y_pred, f"Fold_{fold+1}")
            fold_metrics['fold'] = fold + 1
            fold_metrics['test_start'] = timestamps[test_idx[0]]
            fold_metrics['test_end'] = timestamps[test_idx[-1]]
            wf_scores.append(fold_metrics)
        return wf_scores
    
    def comprehensive_evaluation(self, train_data, val_data, test_data):
        train_features = self.create_features(train_data)
        val_features = self.create_features(val_data)
        test_features = self.create_features(test_data)

        X_train, y_train, train_idx = self.prepare_ml_data(train_features)
        X_val, y_val, val_idx = self.prepare_ml_data(val_features)
        X_test, y_test, test_idx = self.prepare_ml_data(test_features)

        models = {
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'Linear Regression': LinearRegression()
        }
        results = {}

        for model_name, model in models.items():
            model.fit(X_train, y_train)
            train_pred = model.predict(X_train)
            val_pred = model.predict(X_val) if len(X_val) > 0 else []
            test_pred = model.predict(X_test) if len(X_test) > 0 else []

            results[model_name] = {
                'train': self.calculate_metrics(y_train, train_pred, "Train"),
                'val': self.calculate_metrics(y_val, val_pred, "Validation") if len(y_val) > 0 else {},
                'test': self.calculate_metrics(y_test, test_pred, "Test") if len(y_test) > 0 else {}
            }
        
        baselines = self.create_baseline_predictions(train_data, val_data, test_data)
        baseline_results = {}
        for baseline_name, baseline_preds in baselines.items():
            baseline_results[baseline_name] = {
                'train': self.calculate_metrics(train_data['Close'].values, baseline_preds['train']),
                'val': self.calculate_metrics(val_data['Close'].values, baseline_preds['val']) if len(val_data) > 0 else {},
                'test': self.calculate_metrics(test_data['Close'].values, baseline_preds['test']) if len(test_data) > 0 else {}
            }
        
        all_data = pd.concat([train_data, val_data, test_data])
        all_features = self.create_features(all_data)
        wf_results = {}
        for model_name, model in models.items():
            wf_results[model_name] = self.walk_forward_validation(all_features, model)

        return results, baseline_results, wf_results

def get_1min_data(ticker, days=7):
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    try:
        data = yf.download(
            tickers=ticker,
            interval="1m",
            start=start_date,
            end=end_date,
            prepost=False,
            auto_adjust=True,
            threads=True
        )
        
        if data.empty:
            print(f"No data returned for {ticker}")
            return None
            
        if data.index.tz is None:
            data = data.tz_localize('UTC')
        data = data.tz_convert('America/New_York')
        
        data = data[data['Volume'] > 0]
        data = data.between_time('09:30', '16:00')
        
        if data.empty:
            print("No data remaining after market hours filtering")
            return None
            
        filename = f"{ticker}_1min_clean_{start_date.date()}_to_{end_date.date()}.csv"
        data.to_csv(filename, index=True)
        print(f"Saved {len(data)} market minutes to {filename}")
        return data
    
    except Exception as e:
        print(f"Error fetching data for {ticker}: {e}")
        return None

def fetch_ticker_data(ticker):
    """Wrapper function for parallel fetching"""
    return (ticker, get_1min_data(ticker))

def get_multiple_tickers_data(days=7):
    """Fetch data for all tickers in parallel"""
    all_data = {}
    
    for cap_type, tickers in TICKERS_BY_CAP.items():
        print(f"\nFetching {cap_type} tickers...")
        cap_data = {}
        
        # Use ThreadPoolExecutor to fetch data in parallel
        with ThreadPoolExecutor(max_workers=5) as executor:
            results = list(executor.map(fetch_ticker_data, tickers))
        
        for ticker, data in results:
            if data is not None and not data.empty:
                cap_data[ticker] = data
            else:
                print(f"Failed to get data for {ticker}")
        
        all_data[cap_type] = cap_data
    
    return all_data

def split_by_dates(data, train_ratio=0.6, val_ratio=0.2):
    try:
        trading_days = pd.Series(data.index.date).unique()
        if len(trading_days) < 2:
            raise ValueError("Not enough trading days to split")
            
        train_end = int(len(trading_days) * train_ratio)
        val_end = train_end + int(len(trading_days) * val_ratio)
        
        train_end = max(1, train_end)
        val_end = max(train_end + 1, val_end)
        
        train_end_date = trading_days[train_end-1]
        val_end_date = trading_days[val_end-1] if val_end < len(trading_days) else trading_days[-1]
        
        train = data[data.index.date <= train_end_date]
        val = data[(data.index.date > train_end_date) & 
                   (data.index.date <= val_end_date)] if val_end < len(trading_days) else pd.DataFrame()
        test = data[data.index.date > val_end_date] if val_end < len(trading_days) else pd.DataFrame()
        
        return train, val, test
        
    except Exception as e:
        print(f"Error splitting data: {e}")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

def process_ticker_data(ticker, data):
    """Process a single ticker's data through the evaluation pipeline"""
    print(f"\nProcessing {ticker}...")
    evaluator = StockAccuracyEvaluator()
    
    # Split data
    train, val, test = split_by_dates(data)
    
    # Convert any remaining MultiIndex DataFrames
    for df in [train, val, test]:
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
    
    # Run evaluation
    try:
        ml_results, baseline_results, wf_results = evaluator.comprehensive_evaluation(train, val, test)
        return {
            'ticker': ticker,
            'ml_results': ml_results,
            'baseline_results': baseline_results,
            'wf_results': wf_results,
            'success': True
        }
    except Exception as e:
        print(f"Error processing {ticker}: {e}")
        return {
            'ticker': ticker,
            'error': str(e),
            'success': False
        }

def analyze_results_by_market_cap(all_results):
    """Analyze and compare results across market capitalizations"""
    print("\n" + "=" * 60)
    print("Market Capitalization Comparison Summary")
    print("=" * 60)
    
    metrics_by_cap = {
        'large_cap': {'RMSE': [], 'MAPE': [], 'Direction_Accuracy': [], 'R2': []},
        'mid_cap': {'RMSE': [], 'MAPE': [], 'Direction_Accuracy': [], 'R2': []},
        'low_cap': {'RMSE': [], 'MAPE': [], 'Direction_Accuracy': [], 'R2': []}
    }
    
    for cap_type, ticker_results in all_results.items():
        for ticker, results in ticker_results.items():
            if results is None or not results.get('success', False):
                continue
                
            rf_test = results['ml_results'].get('Random Forest', {}).get('test', {})
            
            if rf_test:
                metrics_by_cap[cap_type]['RMSE'].append(rf_test.get('RMSE', np.nan))
                metrics_by_cap[cap_type]['MAPE'].append(rf_test.get('MAPE', np.nan))
                metrics_by_cap[cap_type]['Direction_Accuracy'].append(
                    rf_test.get('Direction_Accuracy', np.nan))
                metrics_by_cap[cap_type]['R2'].append(rf_test.get('R2', np.nan))
    
    for cap_type, metrics in metrics_by_cap.items():
        print(f"\n{cap_type.upper().replace('_', ' ')} PERFORMANCE (N={len(metrics['RMSE'])})")
        
        for metric_name, values in metrics.items():
            valid_values = [v for v in values if not np.isnan(v)]
            if valid_values:
                avg = np.mean(valid_values)
                std = np.std(valid_values)
                if metric_name in ['RMSE']:
                    print(f"  {metric_name}: ${avg:.2f} ± {std:.2f}")
                elif metric_name in ['MAPE', 'Direction_Accuracy']:
                    print(f"  {metric_name}: {avg:.1f}% ± {std:.1f}%")
                else:
                    print(f"  {metric_name}: {avg:.3f} ± {std:.3f}")
            else:
                print(f"  {metric_name}: No valid data")

def save_results_to_csv(all_results, filename="stock_prediction_results.csv"):
    """Save all results to a CSV file for further analysis"""
    rows = []
    
    for cap_type, ticker_results in all_results.items():
        for ticker, results in ticker_results.items():
            if results is None or not results.get('success', False):
                continue
                
            rf_test = results['ml_results'].get('Random Forest', {}).get('test', {})
            if rf_test:
                row = {
                    'Ticker': ticker,
                    'Market_Cap': cap_type,
                    'RMSE': rf_test.get('RMSE', np.nan),
                    'MAPE': rf_test.get('MAPE', np.nan),
                    'Direction_Accuracy': rf_test.get('Direction_Accuracy', np.nan),
                    'R2': rf_test.get('R2', np.nan)
                }
                rows.append(row)
    
    if rows:
        results_df = pd.DataFrame(rows)
        results_df.to_csv(filename, index=False)
        print(f"\nSaved results to {filename}")
    else:
        print("\nNo valid results to save")

if __name__ == "__main__":
    print("Starting analysis of multiple stocks with 1-minute data over 7 days...")
    
    # 1. Get clean data for all tickers
    print("\nFetching data for all tickers...")
    all_data = get_multiple_tickers_data(days=7)
    
    # 2. Process all tickers through the evaluation pipeline
    print("\nProcessing all tickers...")
    all_results = {}
    
    for cap_type, ticker_data in all_data.items():
        cap_results = {}
        
        # Process each ticker in this market cap category
        for ticker, data in ticker_data.items():
            result = process_ticker_data(ticker, data)
            cap_results[ticker] = result
        
        all_results[cap_type] = cap_results
    
    # 3. Analyze and compare results by market cap
    analyze_results_by_market_cap(all_results)
    
    # 4. Save results to CSV for further analysis
    save_results_to_csv(all_results)
    
    print("\nAnalysis complete!")
