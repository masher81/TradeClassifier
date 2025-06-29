#!/usr/bin/env python3
"""
trade_roc_sltp.py - Optimized Version

Key Improvements:
1. Parallel symbol processing with ThreadPoolExecutor
2. Two-level caching (raw data + processed DataFrames)
3. Dynamic rate limiting
4. Pre-filtering of illiquid symbols
5. Optimized indicator calculations
"""

import os
import sys
import time
import csv
import json
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import pandas as pd
import ccxt
import joblib
from tqdm import tqdm
from dotenv import load_dotenv
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# ─────────────── CONFIG ────────────────────────────────────────────────
load_dotenv()
API_KEY = os.getenv('KRAKEN_APIKEY')
API_SECRET = os.getenv('KRAKEN_SECRET')
NOTIONAL = float(os.getenv('NOTIONAL', 10_000))
FEE_RATE = float(os.getenv('FEE_RATE', 0.001))
SLIPPAGE_PCT = float(os.getenv('SLIPPAGE_PCT', 0.0005))
DRY_RUN = os.getenv('DRY_RUN', 'false').lower() in ('1','true')
CLASSIFIER_THRESHOLD = float(os.getenv('CLASSIFIER_THRESHOLD', 0.40))
LOG_FILE = os.getenv('TRADE_LOG_FILE', 'trade_log.csv')
TIMEFRAME = '1h'
MIN_DATA_BARS = 100
HISTORY_HOURS = 168  # 7 days
MIN_AVG_VOLUME = 100  # Skip symbols with avg volume < this
MAX_WORKERS = 5  # Thread count for parallel processing

# Load parameters
with open('best_params.json') as f:
    BEST = json.load(f)
SYMBOLS = list(BEST.keys())

# ─────────────── EXCHANGE SETUP ────────────────────────────────────────
exchange = ccxt.kraken({
    'enableRateLimit': True,
    'apiKey': API_KEY,
    'secret': API_SECRET,
})
exchange.load_markets()

# ─────────────── CACHING SYSTEM ────────────────────────────────────────
RAW_CACHE = {}
PROCESSED_CACHE = {}
CACHE_EXPIRY = timedelta(minutes=15)
last_request_time = 0

# ─────────────── RATE LIMIT HANDLING ───────────────────────────────────
def rate_limited_request():
    """Ensure we don't exceed Kraken's rate limits"""
    global last_request_time
    now = time.time()
    elapsed = now - last_request_time
    if elapsed < 1.0:  # Kraken's 1req/sec limit
        time.sleep(1.0 - elapsed)
    last_request_time = time.time()

# ─────────────── DATA LOADING ──────────────────────────────────────────
def fetch_raw_data(symbol, hours=HISTORY_HOURS, max_retries=3):
    """Fetch raw OHLCV data with retries and caching"""
    if symbol in RAW_CACHE:
        cached_time, data = RAW_CACHE[symbol]
        if datetime.utcnow() - cached_time < CACHE_EXPIRY:
            return data
    
    since = exchange.milliseconds() - int(hours * 3600 * 1000)
    all_bars = []
    retries = 0
    
    while retries < max_retries and len(all_bars) < MIN_DATA_BARS * 2:
        try:
            rate_limited_request()
            chunk = exchange.fetch_ohlcv(symbol, TIMEFRAME, since, limit=500)
            if not chunk:
                retries += 1
                time.sleep(2 ** retries)  # Exponential backoff
                continue
                
            all_bars += chunk
            since = chunk[-1][0] + 1
        except Exception as e:
            print(f"Error loading {symbol} (attempt {retries+1}): {e}")
            retries += 1
            time.sleep(5)
    
    if all_bars:
        RAW_CACHE[symbol] = (datetime.utcnow(), all_bars)
    return all_bars if len(all_bars) >= 50 else None

def compute_indicators(raw_bars):
    """Efficiently compute all indicators from raw data"""
    df = pd.DataFrame(raw_bars, columns=['ts','open','high','low','close','vol'])
    df['dt'] = pd.to_datetime(df['ts'], unit='ms')
    df = df.set_index('dt').sort_index()
    
    # Compute indicators in bulk
    df['prev_close'] = df['close'].shift(1)
    df['tr'] = np.maximum.reduce([
        df['high'] - df['low'],
        abs(df['high'] - df['prev_close']),
        abs(df['low'] - df['prev_close'])
    ])
    
    # Rolling calculations
    df['atr20'] = df['tr'].rolling(20).mean()
    df['rv20'] = np.log(df['close']).diff().rolling(20).std()
    df['ma10'] = df['close'].rolling(10).mean()
    df['ma50'] = df['close'].rolling(50).mean()
    df['ma10_50'] = df['ma10'] - df['ma50']
    
    # RSI calculation
    delta = df['close'].diff()
    up, down = delta.clip(lower=0), -delta.clip(upper=0)
    df['rsi14'] = 100 - 100/(1 + up.rolling(14).mean() / down.rolling(14).mean())
    
    # Volume spike
    df['vol_spike'] = df['vol'] / df['vol'].rolling(20).mean()
    
    return df.dropna()

def load_history(symbol):
    """Load and process data with caching"""
    if symbol in PROCESSED_CACHE:
        cached_time, df = PROCESSED_CACHE[symbol]
        if datetime.utcnow() - cached_time < CACHE_EXPIRY:
            return df
    
    raw_data = fetch_raw_data(symbol)
    if not raw_data:
        return None
    
    df = compute_indicators(raw_data)
    if df is not None:
        PROCESSED_CACHE[symbol] = (datetime.utcnow(), df)
    return df

# ─────────────── TRADING LOGIC ────────────────────────────────────────
def backtest_sltp(df, params):
    """Optimized backtesting function"""
    roc_period = params['roc_period']
    threshold = params['threshold']
    sl_pct = params['sl_pct']
    tp_pct = params['tp_pct']
    
    trades = []
    positions = {}
    
    closes = df['close'].values
    times = df.index
    
    for i in range(roc_period, len(df)):
        current_time = times[i]
        close = closes[i]
        roc = close / closes[i-roc_period] - 1
        
        # Check exits
        for sym in list(positions.keys()):
            entry = positions[sym]
            exit_type = None
            
            if close >= entry['price'] * (1 + tp_pct):
                exit_price = entry['price'] * (1 + tp_pct)
                exit_type = 'TP'
            elif close <= entry['price'] * (1 - sl_pct):
                exit_price = entry['price'] * (1 - sl_pct)
                exit_type = 'SL'
                
            if exit_type:
                exit_adj = exit_price * (1 - FEE_RATE - SLIPPAGE_PCT/2)
                entry_adj = entry['price'] * (1 + FEE_RATE + SLIPPAGE_PCT/2)
                pnl = (exit_adj - entry_adj) * entry['qty']
                hold_hours = (current_time - entry['time']).total_seconds() / 3600
                
                trades.append({
                    'symbol': sym,
                    'entry_time': entry['time'],
                    'exit_time': current_time,
                    'entry_price': entry['price'],
                    'exit_price': exit_price,
                    'exit_type': exit_type,
                    'roc': roc,
                    'pnl': pnl,
                    'hold_hours': hold_hours
                })
                del positions[sym]
        
        # Check entries
        if roc > threshold and symbol not in positions:
            entry_adj = close * (1 + FEE_RATE + SLIPPAGE_PCT/2)
            positions[symbol] = {
                'time': current_time,
                'price': close,
                'qty': NOTIONAL / entry_adj
            }
    
    return pd.DataFrame(trades)

# ─────────────── LOGGING ──────────────────────────────────────────────
def log_trade(row):
    """Optimized trade logging"""
    with open(LOG_FILE, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            row.get('symbol'),
            row.get('action'),
            row.get('timestamp'),
            row.get('price'),
            row.get('qty'),
            row.get('roc', ''),
            row.get('entry_time', ''),
            row.get('entry_price', ''),
            row.get('exit_time', ''),
            row.get('exit_price', ''),
            row.get('exit_type', ''),
            row.get('pnl', ''),
            row.get('hold_hours', ''),
        ])

# ─────────────── MAIN TRAINING ────────────────────────────────────────
print(f"→ Fetching historical data for {len(SYMBOLS)} symbols...")

def train_model():
    """Train the classifier with parallel data loading"""
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(load_history, sym): sym for sym in SYMBOLS}
        history = {}
        
        for future in tqdm(as_completed(futures), total=len(SYMBOLS), desc="Loading data"):
            sym = futures[future]
            try:
                df = future.result()
                if df is not None and len(df) >= MIN_DATA_BARS and df['vol'].mean() >= MIN_AVG_VOLUME:
                    history[sym] = df
            except Exception as e:
                print(f"Error processing {sym}: {e}")
    
    # Backtest all symbols
    all_trades = []
    for sym, df in tqdm(history.items(), desc="Backtesting"):
        trades = backtest_sltp(df, BEST[sym])
        if not trades.empty:
            all_trades.append(trades)
    
    if not all_trades:
        raise ValueError("No trades generated during backtesting")
    
    # Prepare features
    feat_rows = []
    for trade in pd.concat(all_trades).itertuples():
        bar = history[trade.symbol].loc[trade.entry_time]
        feat_rows.append({
            'roc': trade.roc,
            'atr20': bar['atr20'],
            'rv20': bar['rv20'],
            'ma10_50': bar['ma10_50'],
            'rsi14': bar['rsi14'],
            'vol_spike': bar['vol_spike'],
            'hold_hours': trade.hold_hours,
            'hour': trade.entry_time.hour,
            'weekend': int(trade.entry_time.weekday() >= 5),
            'sl_pct': BEST[trade.symbol]['sl_pct'],
            'tp_pct': BEST[trade.symbol]['tp_pct'],
            'y': int(trade.pnl > 0),
        })
    
    # Train classifier
    feat_df = pd.DataFrame(feat_rows).dropna()
    X_train, X_test, y_train, y_test = train_test_split(
        feat_df.drop(columns=['y']), feat_df['y'], 
        test_size=0.2, stratify=feat_df['y'], random_state=42
    )
    
    scaler = StandardScaler().fit(X_train)
    model = RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42)
    model.fit(scaler.transform(X_train), y_train)
    
    print("\n=== Classification Report ===")
    print(classification_report(y_test, model.predict(scaler.transform(X_test))))
    
    # Save models
    joblib.dump(scaler, 'scaler.pkl')
    joblib.dump(model, 'classifier.pkl')
    return scaler, model

scaler, model = train_model()
print("\n→ Training complete. Starting live trading...")

# ─────────────── LIVE TRADING ─────────────────────────────────────────
def process_symbol(symbol, positions):
    """Process a single symbol in parallel"""
    try:
        df = load_history(symbol)
        if df is None or len(df) < MIN_DATA_BARS or df['vol'].mean() < MIN_AVG_VOLUME:
            return None
        
        params = BEST[symbol]
        roc_period = params['roc_period']
        if len(df) < roc_period + 1:
            return None
            
        now = datetime.utcnow()
        close = df['close'].iloc[-1]
        roc = close / df['close'].iloc[-1-roc_period] - 1
        
        # Exit checks
        if symbol in positions:
            pos = positions[symbol]
            exit_type = None
            
            if close >= pos['entry_price'] * (1 + params['tp_pct']):
                exit_price = pos['entry_price'] * (1 + params['tp_pct'])
                exit_type = 'TP'
            elif close <= pos['entry_price'] * (1 - params['sl_pct']):
                exit_price = pos['entry_price'] * (1 - params['sl_pct'])
                exit_type = 'SL'
                
            if exit_type:
                exit_adj = exit_price * (1 - FEE_RATE - SLIPPAGE_PCT/2)
                pnl = (exit_adj - pos['entry_price']) * pos['qty']
                hold_hours = (now - pos['entry_time']).total_seconds() / 3600
                
                if not DRY_RUN:
                    try:
                        rate_limited_request()
                        exchange.create_order(symbol, 'market', 'sell', pos['qty'])
                    except Exception as e:
                        print(f"Exit order failed: {e}")
                        return None
                
                log_trade({
                    'symbol': symbol,
                    'action': 'EXIT',
                    'timestamp': now.isoformat(),
                    'price': f"{exit_price:.4f}",
                    'qty': f"{pos['qty']:.6f}",
                    'exit_type': exit_type,
                    'entry_time': pos['entry_time'].isoformat(),
                    'entry_price': f"{pos['entry_price']:.4f}",
                    'exit_time': now.isoformat(),
                    'exit_price': f"{exit_price:.4f}",
                    'pnl': f"{pnl:.2f}",
                    'hold_hours': f"{hold_hours:.2f}"
                })
                del positions[symbol]
                return exit_type
        
        # Entry checks
        if symbol not in positions and roc > params['threshold']:
            feat = {
                'roc': roc,
                'atr20': df['atr20'].iloc[-1],
                'rv20': df['rv20'].iloc[-1],
                'ma10_50': df['ma10_50'].iloc[-1],
                'rsi14': df['rsi14'].iloc[-1],
                'vol_spike': df['vol_spike'].iloc[-1],
                'hold_hours': 0,
                'hour': now.hour,
                'weekend': int(now.weekday() >= 5),
                'sl_pct': params['sl_pct'],
                'tp_pct': params['tp_pct'],
            }
            
            proba = model.predict_proba(scaler.transform(pd.DataFrame([feat]))[0, 1]
            if proba >= CLASSIFIER_THRESHOLD:
                entry_adj = close * (1 + FEE_RATE + SLIPPAGE_PCT/2)
                qty = NOTIONAL / entry_adj
                
                if not DRY_RUN:
                    try:
                        rate_limited_request()
                        exchange.create_order(symbol, 'market', 'buy', qty)
                    except Exception as e:
                        print(f"Entry order failed: {e}")
                        return None
                
                positions[symbol] = {
                    'entry_time': now,
                    'entry_price': close,
                    'qty': qty
                }
                
                log_trade({
                    'symbol': symbol,
                    'action': 'ENTER',
                    'timestamp': now.isoformat(),
                    'price': f"{close:.4f}",
                    'qty': f"{qty:.6f}",
                    'roc': f"{roc:.4f}",
                    'entry_time': now.isoformat(),
                    'entry_price': f"{close:.4f}",
                })
                return "ENTER"
    except Exception as e:
        print(f"Error processing {symbol}: {e}")
    return None

def live_trading_loop():
    """Optimized main trading loop"""
    positions = {}
    print(f"\n▶️ Starting live trader (SL/TP version) - Dry run={DRY_RUN}")
    
    try:
        while True:
            cycle_start = time.time()
            print(f"\n[{datetime.utcnow()}] Open positions: {len(positions)}")
            
            # Refresh cache every 15 minutes
            if datetime.utcnow().minute % 15 == 0:
                RAW_CACHE.clear()
                PROCESSED_CACHE.clear()
            
            # Process all symbols in parallel
            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                futures = [executor.submit(process_symbol, sym, positions) for sym in SYMBOLS]
                for future in as_completed(futures):
                    future.result()  # Just wait for completion
            
            # Sleep until next candle
            elapsed = time.time() - cycle_start
            sleep_time = max(60 - elapsed, 1)
            time.sleep(sleep_time)
            
    except KeyboardInterrupt:
        print("\n🛑 Stopped cleanly")

if __name__ == "__main__":
    live_trading_loop()
