#!/usr/bin/env python3
"""
generate_best_params.py - Optimize ROC+SL/TP parameters for all Kraken USD pairs
"""

import os
import json
import time
import numpy as np
import pandas as pd
import ccxt
from datetime import datetime, timedelta
from itertools import product
from tqdm import tqdm
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
API_KEY = os.getenv('KRAKEN_APIKEY')
API_SECRET = os.getenv('KRAKEN_SECRET')

# Configuration
TIMEFRAME = '1h'
START_TS = datetime.utcnow() - timedelta(days=365)  # 1 year back
END_TS = datetime.utcnow()
NOTIONAL = 10000  # For position sizing
FEE_RATE = 0.001  # 0.1% fee
SLIPPAGE_PCT = 0.0005  # 0.05% slippage

# Parameter search space
PARAM_GRID = {
    'roc_period': [1, 2, 3, 5, 8, 13],
    'threshold': [0.005, 0.01, 0.015, 0.02, 0.025],
    'sl_pct': [0.005, 0.008, 0.01, 0.015, 0.02],
    'tp_pct': [0.01, 0.015, 0.02, 0.025, 0.03, 0.04]
}

# Exchange setup
exchange = ccxt.kraken({
    'enableRateLimit': True,
    'apiKey': API_KEY,
    'secret': API_SECRET
})
exchange.load_markets()

def get_usd_pairs():
    """Get all active USD trading pairs on Kraken"""
    usd_pairs = []
    for symbol in exchange.symbols:
        if symbol.endswith('/USD') and exchange.markets[symbol].get('active'):
            usd_pairs.append(symbol)
    return sorted(usd_pairs)

def fetch_ohlcv(symbol, start, end):
    """Fetch OHLCV data between start and end timestamps"""
    since = exchange.parse8601(start.isoformat() + 'Z')
    end_ts = exchange.parse8601(end.isoformat() + 'Z')
    all_bars = []
    
    while since < end_ts:
        try:
            chunk = exchange.fetch_ohlcv(symbol, TIMEFRAME, since, limit=500)
            if not chunk:
                break
            all_bars += chunk
            since = chunk[-1][0] + 1
            time.sleep(exchange.rateLimit / 1000)
        except Exception as e:
            print(f"Error fetching {symbol}: {e}")
            break
    
    if not all_bars:
        return None
    
    df = pd.DataFrame(all_bars, columns=['ts', 'open', 'high', 'low', 'close', 'volume'])
    df['dt'] = pd.to_datetime(df['ts'], unit='ms')
    df = df.set_index('dt').sort_index()
    return df[df.index <= end]

def calculate_indicators(df):
    """Calculate technical indicators"""
    if df.empty:
        return df
    
    # True Range and ATR
    df['prev_close'] = df['close'].shift(1)
    df['tr'] = np.maximum.reduce([
        df['high'] - df['low'],
        (df['high'] - df['prev_close']).abs(),
        (df['low'] - df['prev_close']).abs()
    ])
    df['atr20'] = df['tr'].rolling(20).mean()
    
    # Other indicators
    df['rv20'] = np.log(df['close']).diff().rolling(20).std()
    df['ma10'] = df['close'].rolling(10).mean()
    df['ma50'] = df['close'].rolling(50).mean()
    df['ma10_50'] = df['ma10'] - df['ma50']
    
    # RSI
    delta = df['close'].diff()
    up, down = delta.clip(lower=0), -delta.clip(upper=0)
    df['rsi14'] = 100 - 100/(1 + up.rolling(14).mean() / down.rolling(14).mean())
    
    # Volume spike
    df['vol_spike'] = df['volume'] / df['volume'].rolling(20).mean()
    
    return df.dropna()

def backtest_sl_tp(df, params):
    """Backtest ROC strategy with SL/TP exits"""
    roc_period = params['roc_period']
    threshold = params['threshold']
    sl_pct = params['sl_pct']
    tp_pct = params['tp_pct']
    
    trades = []
    in_position = False
    entry_price = None
    entry_idx = None
    
    for i in range(roc_period, len(df)):
        if in_position:
            current_price = df['close'].iloc[i]
            
            # Check for TP/SL
            if current_price >= entry_price * (1 + tp_pct):
                exit_type = 'TP'
                exit_price = entry_price * (1 + tp_pct)
            elif current_price <= entry_price * (1 - sl_pct):
                exit_type = 'SL'
                exit_price = entry_price * (1 - sl_pct)
            else:
                continue
            
            # Calculate PnL with fees and slippage
            exit_price_adj = exit_price * (1 - FEE_RATE - SLIPPAGE_PCT/2)
            qty = NOTIONAL / (entry_price * (1 + FEE_RATE + SLIPPAGE_PCT/2))
            pnl = (exit_price_adj - entry_price) * qty
            pnl_pct = (exit_price_adj - entry_price) / entry_price
            hold_hours = (df.index[i] - df.index[entry_idx]).total_seconds() / 3600
            
            trades.append({
                'entry_dt': df.index[entry_idx],
                'exit_dt': df.index[i],
                'entry_price': entry_price,
                'exit_price': exit_price_adj,
                'pnl': pnl,
                'pnl_pct': pnl_pct,
                'exit_type': exit_type,
                'hold_hours': hold_hours,
                'roc': (df['close'].iloc[i] / df['close'].iloc[i-roc_period]) - 1,
                **params
            })
            
            in_position = False
            
        # Check for new entry
        else:
            roc = (df['close'].iloc[i] / df['close'].iloc[i-roc_period]) - 1
            if roc > threshold:
                in_position = True
                entry_price = df['close'].iloc[i]
                entry_idx = i
    
    return pd.DataFrame(trades)

def evaluate_params(trades_df):
    """Evaluate parameter set based on trades"""
    if trades_df.empty or len(trades_df) < 5:
        return {
            'sharpe': -np.inf,
            'profit_factor': 0,
            'win_rate': 0,
            'avg_trade': 0,
            'n_trades': 0
        }
    
    returns = trades_df['pnl_pct']
    wins = trades_df[trades_df['pnl'] > 0]
    losses = trades_df[trades_df['pnl'] <= 0]
    
    sharpe = returns.mean() / (returns.std() + 1e-9)
    profit_factor = wins['pnl'].sum() / (abs(losses['pnl'].sum()) + 1e-9)
    win_rate = len(wins) / len(trades_df)
    avg_trade = trades_df['pnl'].mean()
    
    return {
        'sharpe': sharpe,
        'profit_factor': profit_factor,
        'win_rate': win_rate,
        'avg_trade': avg_trade,
        'n_trades': len(trades_df)
    }

def optimize_symbol(symbol):
    """Find best parameters for a single symbol"""
    print(f"\nOptimizing {symbol}...")
    
    # Fetch and prepare data
    df = fetch_ohlcv(symbol, START_TS, END_TS)
    if df is None or len(df) < 100:
        print(f"⚠️ Insufficient data for {symbol}")
        return None
    
    df = calculate_indicators(df)
    
    # Generate all parameter combinations
    param_names = list(PARAM_GRID.keys())
    param_values = list(PARAM_GRID.values())
    all_combinations = list(product(*param_values))
    
    best_score = -np.inf
    best_params = None
    best_metrics = None
    
    # Brute-force search
    for combo in tqdm(all_combinations, desc=f"Testing {symbol}"):
        params = dict(zip(param_names, combo))
        
        # Skip if TP <= SL
        if params['tp_pct'] <= params['sl_pct']:
            continue
        
        trades = backtest_sl_tp(df, params)
        metrics = evaluate_params(trades)
        
        # Use Sharpe ratio as primary metric
        if metrics['sharpe'] > best_score and metrics['n_trades'] >= 5:
            best_score = metrics['sharpe']
            best_params = params
            best_metrics = metrics
    
    if best_params:
        print(f"✅ Best params for {symbol}:")
        print(f"   ROC Period: {best_params['roc_period']}")
        print(f"   Threshold: {best_params['threshold']:.3f}")
        print(f"   SL: {best_params['sl_pct']:.3f}, TP: {best_params['tp_pct']:.3f}")
        print(f"   Sharpe: {best_metrics['sharpe']:.2f}, Win Rate: {best_metrics['win_rate']:.1%}")
        print(f"   Trades: {best_metrics['n_trades']}, Avg Trade: {best_metrics['avg_trade']:.2f}")
        
        return {
            'params': best_params,
            'metrics': best_metrics
        }
    else:
        print(f"⚠️ No valid parameters found for {symbol}")
        return None

def main():
    """Main optimization routine"""
    # Get all active USD pairs
    usd_pairs = get_usd_pairs()
    print(f"Found {len(usd_pairs)} USD pairs: {usd_pairs}")
    
    best_params = {}
    
    for symbol in usd_pairs:
        result = optimize_symbol(symbol)
        if result:
            best_params[symbol] = result['params']
    
    # Save results
    with open('best_params.json', 'w') as f:
        json.dump(best_params, f, indent=2)
    print(f"\nSaved best parameters for {len(best_params)} pairs to best_params.json")

if __name__ == '__main__':
    main()
