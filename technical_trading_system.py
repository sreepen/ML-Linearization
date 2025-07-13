# technical_trading_system.py
import yfinance as yf
import pandas as pd
import numpy as np
import warnings
from datetime import time, datetime, timedelta
warnings.filterwarnings('ignore')

# Dictionary of tickers categorized by market capitalization
TICKERS = {
    'large_cap': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META'],
    'mid_cap': ['DECK', 'WYNN', 'RCL', 'MGM', 'NCLH'],
    'small_cap': ['FIZZ', 'SMPL', 'BYND', 'PLUG', 'CLOV']
}

def download_data_robust(ticker, period='7d', interval='1m', start_time='9:30', end_time='16:00', retries=3):
    """Download stock data with retry logic and basic cleaning"""
    for attempt in range(retries):
        try:
            data = yf.download(ticker, period=period, interval=interval, 
                             progress=False, auto_adjust=True, prepost=True)
            if data.empty:
                continue
                
            # Clean column names 
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = [col[0].lower() if isinstance(col, tuple) else col.lower() 
                               for col in data.columns]
            else:
                data.columns = data.columns.str.lower()
            
            # Filter to regular trading hours (9:30am-4pm)
            if isinstance(data.index, pd.DatetimeIndex):
                try:
                    #convert start_time and end_time to time objects
                    start = time(*map(int, start_time.split(':')))
                    end = time(*map(int, end_time.split(':')))

                    #for overnight sessions (like pre-market to regular hours)
                    if start > end:
                        data = data[((data.index.time >= start) | (data.index.time <= end))]

                    else: data = data[(data.index.time >= start) & (data.index.time <= end)]

                    #data = data.between_time('09:30', '16:00')
                except Exception as e:
                    print(f"Time filtering error: {e}")
                    pass
            
            # Only return if we have sufficient data
            if len(data) > 200:  # Need more data for RSI calculation
                return data
        except Exception as e:
            continue
    return None

class TechnicalTradingSystem:
    def __init__(self, macd_fast=12, macd_slow=26, macd_signal=9, 
                 rsi_period=14, rsi_overbought=70, rsi_oversold=30,
                 trailing_stop_pct=0.05):
        """
        Initialize technical trading system
        
        Args:
            macd_fast: Fast EMA period for MACD
            macd_slow: Slow EMA period for MACD
            macd_signal: Signal line EMA period
            rsi_period: RSI calculation period
            rsi_overbought: RSI level considered overbought (sell signal)
            rsi_oversold: RSI level considered oversold (buy signal)
            trailing_stop_pct: Trailing stop loss percentage
        """
        self.macd_fast = macd_fast
        self.macd_slow = macd_slow
        self.macd_signal = macd_signal
        self.rsi_period = rsi_period
        self.rsi_overbought = rsi_overbought
        self.rsi_oversold = rsi_oversold
        self.trailing_stop_pct = trailing_stop_pct
    
    def calculate_macd(self, prices):
        """Calculate MACD line, signal line, and histogram"""
        ema_fast = prices.ewm(span=self.macd_fast, adjust=False).mean()
        ema_slow = prices.ewm(span=self.macd_slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=self.macd_signal, adjust=False).mean()
        histogram = macd_line - signal_line
        
        return macd_line, signal_line, histogram
    
    def calculate_rsi(self, prices, period=None):
        """Calculate RSI (Relative Strength Index)"""
        if period is None:
            period = self.rsi_period
            
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def generate_signals(self, data):
        """Generate buy/sell signals based on MACD and RSI"""
        close_prices = data['close']
        
        # Calculate technical indicators
        macd_line, signal_line, histogram = self.calculate_macd(close_prices)
        rsi = self.calculate_rsi(close_prices)
        
        # Create signals dataframe
        signals = pd.DataFrame({
            'price': close_prices,
            'macd': macd_line,
            'signal': signal_line,
            'histogram': histogram,
            'rsi': rsi,
            'volume': data['volume']
        })
        
        # MACD signals
        signals['macd_bullish'] = (signals['macd'] > signals['signal']) & (signals['macd'].shift(1) <= signals['signal'].shift(1))
        signals['macd_bearish'] = (signals['macd'] < signals['signal']) & (signals['macd'].shift(1) >= signals['signal'].shift(1))
        
        # RSI signals
        signals['rsi_oversold'] = signals['rsi'] < self.rsi_oversold
        signals['rsi_overbought'] = signals['rsi'] > self.rsi_overbought
        
        # Combined signals
        # BUY: MACD bullish crossover AND RSI oversold (or recently oversold)
        signals['rsi_buy_condition'] = (signals['rsi'] < self.rsi_oversold) | (signals['rsi'].shift(1) < self.rsi_oversold)
        signals['buy_signal'] = signals['macd_bullish'] & signals['rsi_buy_condition']
        
        # SELL: MACD bearish crossover OR RSI overbought
        signals['sell_signal'] = signals['macd_bearish'] | signals['rsi_overbought']
        
        return signals.dropna()
    
    def backtest_strategy(self, signals, initial_capital=10000, transaction_cost=0.001):
        """
        Backtest the technical trading strategy with trailing stop
        
        Args:
            signals: DataFrame with trading signals
            initial_capital: Starting capital
            transaction_cost: Transaction cost as percentage
            
        Returns:
            Dictionary with backtest results
        """
        capital = initial_capital
        shares = 0
        trades = 0
        profitable_trades = 0
        total_fees = 0
        
        position_open = False
        buy_price = 0
        highest_price_since_buy = 0
        
        trade_log = []
        portfolio_value = []
        
        for i, row in signals.iterrows():
            current_price = row['price']
            
            # Calculate current portfolio value
            if position_open:
                current_portfolio_value = shares * current_price
            else:
                current_portfolio_value = capital
            portfolio_value.append(current_portfolio_value)
            
            # BUY LOGIC
            if row['buy_signal'] and not position_open:
                shares_to_buy = capital / current_price
                fee = capital * transaction_cost
                shares = shares_to_buy
                buy_price = current_price
                highest_price_since_buy = current_price
                capital = 0
                total_fees += fee
                position_open = True
                trades += 1
                
                trade_log.append({
                    'timestamp': i,
                    'action': 'BUY',
                    'price': current_price,
                    'shares': shares,
                    'fee': fee,
                    'reason': f"MACD: {row['macd']:.4f}, RSI: {row['rsi']:.1f}",
                    'macd_signal': row['macd_bullish'],
                    'rsi_oversold': row['rsi_buy_condition']
                })
            
            # SELL LOGIC (including trailing stop)
            elif position_open:
                # Update trailing stop
                if current_price > highest_price_since_buy:
                    highest_price_since_buy = current_price
                
                # Calculate trailing stop price
                trailing_stop_price = highest_price_since_buy * (1 - self.trailing_stop_pct)
                
                # Determine sell conditions
                trailing_stop_triggered = current_price <= trailing_stop_price
                signal_sell = row['sell_signal']
                
                should_sell = trailing_stop_triggered or signal_sell
                
                if should_sell:
                    # Execute sell
                    sell_value = shares * current_price
                    fee = sell_value * transaction_cost
                    capital = sell_value - fee
                    total_fees += fee
                    
                    trade_profit = sell_value - (shares * buy_price) - fee
                    
                    # Determine sell reason
                    if trailing_stop_triggered:
                        sell_reason = "TRAILING STOP"
                    elif row['macd_bearish']:
                        sell_reason = "MACD BEARISH"
                    elif row['rsi_overbought']:
                        sell_reason = "RSI OVERBOUGHT"
                    else:
                        sell_reason = "SIGNAL"
                    
                    if current_price > buy_price:
                        profitable_trades += 1
                    
                    trade_log.append({
                        'timestamp': i,
                        'action': f'SELL ({sell_reason})',
                        'price': current_price,
                        'shares': shares,
                        'fee': fee,
                        'profit': trade_profit,
                        'reason': f"MACD: {row['macd']:.4f}, RSI: {row['rsi']:.1f}",
                        'highest_since_buy': highest_price_since_buy,
                        'stop_price': trailing_stop_price,
                        'macd_bearish': row['macd_bearish'],
                        'rsi_overbought': row['rsi_overbought']
                    })
                    
                    shares = 0
                    position_open = False
                    highest_price_since_buy = 0
        
        # Close any remaining position
        if position_open:
            final_price = signals.iloc[-1]['price']
            sell_value = shares * final_price
            fee = sell_value * transaction_cost
            capital = sell_value - fee
            total_fees += fee
            
            trade_profit = sell_value - (shares * buy_price) - fee
            
            if final_price > buy_price:
                profitable_trades += 1
            
            trade_log.append({
                'timestamp': signals.index[-1],
                'action': 'SELL (Final)',
                'price': final_price,
                'shares': shares,
                'fee': fee,
                'profit': trade_profit
            })
        
        # Calculate performance metrics
        total_return = capital - initial_capital
        return_percentage = (total_return / initial_capital) * 100
        
        # Calculate buy and hold performance
        buy_hold_return = self.calculate_buy_hold_return(signals, initial_capital)
        
        # Calculate maximum drawdown
        portfolio_series = pd.Series(portfolio_value)
        running_max = portfolio_series.expanding().max()
        drawdown = (portfolio_series - running_max) / running_max
        max_drawdown = drawdown.min() * 100
        
        return {
            'initial_capital': initial_capital,
            'final_capital': capital,
            'total_return': total_return,
            'return_percentage': return_percentage,
            'total_trades': trades,
            'profitable_trades': profitable_trades,
            'win_rate': profitable_trades / trades if trades > 0 else 0,
            'total_fees': total_fees,
            'max_drawdown': max_drawdown,
            'buy_hold_return': buy_hold_return,
            'trade_log': trade_log,
            'portfolio_value': portfolio_value,
            'strategy': f'MACD+RSI+TrailingStop({self.trailing_stop_pct*100:.1f}%)'
        }
    
    def calculate_buy_hold_return(self, signals, initial_capital):
        """Calculate buy and hold return for comparison"""
        initial_price = signals.iloc[0]['price']
        final_price = signals.iloc[-1]['price']
        
        shares = initial_capital / initial_price
        final_value = shares * final_price
        return_percentage = ((final_value - initial_capital) / initial_capital) * 100
        
        return return_percentage
    
    def analyze_signals(self, signals):
        """Analyze the quality and frequency of generated signals"""
        total_periods = len(signals)
        buy_signals = signals['buy_signal'].sum()
        sell_signals = signals['sell_signal'].sum()
        
        # MACD analysis
        macd_bullish = signals['macd_bullish'].sum()
        macd_bearish = signals['macd_bearish'].sum()
        
        # RSI analysis
        rsi_oversold = signals['rsi_oversold'].sum()
        rsi_overbought = signals['rsi_overbought'].sum()
        
        avg_rsi = signals['rsi'].mean()
        
        return {
            'total_periods': total_periods,
            'buy_signals': buy_signals,
            'sell_signals': sell_signals,
            'macd_bullish_crossovers': macd_bullish,
            'macd_bearish_crossovers': macd_bearish,
            'rsi_oversold_periods': rsi_oversold,
            'rsi_overbought_periods': rsi_overbought,
            'avg_rsi': avg_rsi,
            'signal_frequency': (buy_signals + sell_signals) / total_periods * 100
        }
    
    def backtest_time_windows(self, ticker, time_windows, initial_capital=10000):
        """
        Backtest strategy across different time windows

        Args:
            ticker: Stock ticker symbol
            time_windows: List of tuples with (start_time, end_time)
            initial_capital: Starting capital
        
        Returns:
            DataFrame with results for each time window
        """
        results = []

        for window in time_windows:
            start_time, end_time = window

            # Download data for this time window
            data = download_data_robust(ticker, start_time=start_time, end_time=end_time)
            if data is None or len(data) < 50:
                continue

            # Generate signals and backtest
            signals = self.generate_signals(data)
            backtest = self.backtest_strategy(signals, initial_capital)

            results.append({
                'ticker': ticker,
                'time_window': f"{start_time}-{end_time}",
                'strategy_return': backtest['return_percentage'],
                'buy_hold_return': backtest['buy_hold_return'],
                'alpha': backtest['return_percentage'] - backtest['buy_hold_return'],
                'total_trades': backtest['total_trades'],
                'win_rate': backtest['win_rate'],
                'max_drawdown': backtest['max_drawdown']
            })
        
        return pd.DataFrame(results)

def analyze_time_intervals():
    # Analyze performance across different time intervals
    print("\nTime Interval Analysis")
    print("=" * 80)
    print("Testing different trading windows to find optimal times")

    #Define time windows to test
    time_windows = [
        ('06:30', '16:00'), #early pre-market to close
        ('07:00', '16:00'), #mind pre-market to close
        ('08:00', '16:00'), #late pre-market to close
        ('09:30', '16:00'),  # Regular trading hours only
        ('09:30', '15:30'),  # Regular hours minus last 30 mins
        ('10:00', '15:00'),  # Core trading hours
        ('06:30', '12:00'),  # Morning session only
        ('07:00', '11:00'), #7-11 am
        ('12:00', '16:00')   # Afternoon session only
    ]

    #initialize trading system
    system = TechnicalTradingSystem(
        macd_fast=12,
        macd_slow=26,
        macd_signal=9,
        rsi_period=14,
        rsi_overbought=70,
        rsi_oversold=30,
        trailing_stop_pct=0.05
    )

        # Test on a sample of tickers
    test_tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'DECK', 'WYNN', 'RCL', 'MGM', 'NCLH']
    
    all_results = []
    
    for ticker in test_tickers:
        print(f"\nAnalyzing {ticker} across time windows...")
        results = system.backtest_time_windows(ticker, time_windows)
        if not results.empty:
            all_results.append(results)
    
    if not all_results:
        print("No valid results obtained")
        return
    
    combined_results = pd.concat(all_results)
    
    # Aggregate results by time window
    summary = combined_results.groupby('time_window').agg({
        'alpha': 'mean',
        'strategy_return': 'mean',
        'buy_hold_return': 'mean',
        'win_rate': 'mean',
        'total_trades': 'mean'
    }).sort_values('alpha', ascending=False)
    
    print("\nTIME WINDOW PERFORMANCE SUMMARY")
    print("=" * 80)
    print(summary.round(2))
    
    # Find best performing window
    best_window = summary.iloc[0].name
    avg_alpha = summary.iloc[0]['alpha']
    
    print(f"\nBEST PERFORMING TIME WINDOW: {best_window} (Avg Alpha: {avg_alpha:.2f}%)")
    
    # Additional analysis
    early_windows = [w for w in time_windows if w[0] <= '08:00']
    early_results = combined_results[combined_results['time_window'].isin([f"{s}-{e}" for s,e in early_windows])]
    
    if not early_results.empty:
        early_alpha = early_results['alpha'].mean()
        print(f"\nEARLY MORNING SESSIONS (6:30-8:00 start):")
        print(f"  Average Alpha: {early_alpha:.2f}%")
        print(f"  Average Win Rate: {early_results['win_rate'].mean():.1%}")
        print(f"  Average Trades: {early_results['total_trades'].mean():.1f}")
    
    print("\nRECOMMENDATIONS:")
    print("1. Early morning sessions often show different volatility patterns")
    print("2. Pre-market moves can sometimes predict regular session trends")
    print("3. Liquidity is lower in pre-market, affecting execution")
    print("4. The optimal window depends on your strategy and risk tolerance")

def analyze_ticker_technical(ticker, system):
    """Analyze a single ticker using technical trading system"""
    print(f"\n{'='*60}")
    print(f"ANALYZING: {ticker}")
    print(f"{'='*60}")
    
    data = download_data_robust(ticker)
    if data is None:
        print(f" Failed to download data for {ticker}")
        return None
    
    try:
        # Generate signals
        signals = system.generate_signals(data)
        
        if len(signals) < 50:
            print(f" Insufficient data for {ticker}")
            return None
        
        # Analyze signals
        signal_analysis = system.analyze_signals(signals)
        
        # Backtest strategy
        backtest_results = system.backtest_strategy(signals)
        
        # Print detailed results
        print(f" Signal Analysis:")
        print(f"   Total periods: {signal_analysis['total_periods']}")
        print(f"   Buy signals: {signal_analysis['buy_signals']}")
        print(f"   Sell signals: {signal_analysis['sell_signals']}")
        print(f"   MACD bullish crossovers: {signal_analysis['macd_bullish_crossovers']}")
        print(f"   MACD bearish crossovers: {signal_analysis['macd_bearish_crossovers']}")
        print(f"   RSI oversold periods: {signal_analysis['rsi_oversold_periods']}")
        print(f"   RSI overbought periods: {signal_analysis['rsi_overbought_periods']}")
        print(f"   Average RSI: {signal_analysis['avg_rsi']:.1f}")
        print(f"   Signal frequency: {signal_analysis['signal_frequency']:.1f}%")
        
        print(f"\n Trading Results:")
        print(f"   Strategy return: {backtest_results['return_percentage']:+.2f}%")
        print(f"   Buy & Hold return: {backtest_results['buy_hold_return']:+.2f}%")
        print(f"   Alpha (outperformance): {backtest_results['return_percentage'] - backtest_results['buy_hold_return']:+.2f}%")
        print(f"   Total trades: {backtest_results['total_trades']}")
        print(f"   Win rate: {backtest_results['win_rate']:.1%}")
        print(f"   Max drawdown: {backtest_results['max_drawdown']:.2f}%")
        print(f"   Total fees: ${backtest_results['total_fees']:.2f}")
        
        # Show recent trades
        if backtest_results['trade_log']:
            print(f"\n Recent Trades:")
            for trade in backtest_results['trade_log'][-3:]:
                print(f"   {trade['action']}: ${trade['price']:.2f} - {trade['reason']}")
        
        return {
            'ticker': ticker,
            'strategy_return': backtest_results['return_percentage'],
            'buy_hold_return': backtest_results['buy_hold_return'],
            'alpha': backtest_results['return_percentage'] - backtest_results['buy_hold_return'],
            'total_trades': backtest_results['total_trades'],
            'win_rate': backtest_results['win_rate'],
            'max_drawdown': backtest_results['max_drawdown'],
            'signal_frequency': signal_analysis['signal_frequency'],
            'avg_rsi': signal_analysis['avg_rsi']
        }
    
    except Exception as e:
        print(f" Error analyzing {ticker}: {str(e)}")
        return None

def run_comprehensive_analysis():
    """Run comprehensive analysis across all tickers"""
    print(" TECHNICAL ANALYSIS TRADING SYSTEM")
    print("=" * 80)
    print("Strategy: MACD Crossover + RSI + Trailing Stop Loss")
    print("MACD: 12/26/9 periods")
    print("RSI: 14 periods (Buy: <30, Sell: >70)")
    print("Trailing Stop: 5% from peak")
    print("Starting Capital: $10,000")
    print("Transaction Cost: 0.1%")
    print("=" * 80)
    
    # Initialize trading system
    system = TechnicalTradingSystem(
        macd_fast=12,
        macd_slow=26,
        macd_signal=9,
        rsi_period=14,
        rsi_overbought=70,
        rsi_oversold=30,
        trailing_stop_pct=0.05
    )
    
    results = []
    
    # Analyze all tickers
    for category, tickers in TICKERS.items():
        print(f"\n  {category.upper()} STOCKS:")
        for ticker in tickers:
            result = analyze_ticker_technical(ticker, system)
            if result:
                results.append(result)
    
    # Print summary
    print("\n" + "=" * 100)
    print(" COMPREHENSIVE RESULTS SUMMARY")
    print("=" * 100)
    
    if not results:
        print(" No valid results to display")
        return
    
    # Summary table
    print(f"{'Ticker':<8}{'Strategy':<12}{'Buy&Hold':<12}{'Alpha':<10}{'Trades':<8}{'Win%':<8}{'MaxDD%':<8}")
    print("-" * 100)
    
    total_strategy_return = 0
    total_buy_hold_return = 0
    total_alpha = 0
    positive_alpha_count = 0
    
    for result in results:
        total_strategy_return += result['strategy_return']
        total_buy_hold_return += result['buy_hold_return']
        total_alpha += result['alpha']
        
        if result['alpha'] > 0:
            positive_alpha_count += 1
        
        print(f"{result['ticker']:<8}{result['strategy_return']:+.2f}%{'':<4}"
              f"{result['buy_hold_return']:+.2f}%{'':<4}{result['alpha']:+.2f}%{'':<4}"
              f"{result['total_trades']:<8}{result['win_rate']:.1%}{'':<3}"
              f"{result['max_drawdown']:.2f}%")
    
    count = len(results)
    avg_strategy = total_strategy_return / count
    avg_buy_hold = total_buy_hold_return / count
    avg_alpha = total_alpha / count
    
    print("-" * 100)
    print(f"{'AVERAGE':<8}{avg_strategy:+.2f}%{'':<4}{avg_buy_hold:+.2f}%{'':<4}{avg_alpha:+.2f}%")
    
    print(f"\n FINAL ANALYSIS:")
    print(f"   Stocks analyzed: {count}")
    print(f"   Average strategy return: {avg_strategy:+.2f}%")
    print(f"   Average buy & hold return: {avg_buy_hold:+.2f}%")
    print(f"   Average alpha (outperformance): {avg_alpha:+.2f}%")
    print(f"   Stocks with positive alpha: {positive_alpha_count}/{count} ({positive_alpha_count/count:.1%})")
    
    if avg_alpha > 0:
        print(f"    Strategy OUTPERFORMED buy & hold by {avg_alpha:.2f}%")
    else:
        print(f"    Strategy UNDERPERFORMED buy & hold by {abs(avg_alpha):.2f}%")
    


if __name__ == "__main__":
    run_comprehensive_analysis()
    analyze_time_intervals()