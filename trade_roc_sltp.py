#!/usr/bin/env python3
"""
trade_roc_sltp.py

1) Fetch historical data, backtest ROC strategy with SL/TP, build features, train classifier
2) Run live ROC-based trader with SL/TP exits, filtered through trained classifier
"""

import os
import sys
import time
import csv
import json
from datetime import datetime, timedelta
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

# Load best parameters including SL/TP values
with open('best_params.json') as f:
    BEST = json.load(f)
SYMBOLS = list(BEST.keys())

# ─────────────── EXCHANGE SETUP ─────────────────────────────────────────
exchange = ccxt.kraken({
    'enableRateLimit': True,
    'apiKey': API_KEY,
    'secret': API_SECRET,
})
exchange.load_markets()

# ─────────────── LOG FILE SETUP ────────────────────────────────────────
if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'symbol', 'action', 'timestamp', 'price', 'qty', 'roc',
            'entry_time', 'entry_price', 'exit_time', 'exit_price', 
            'exit_type', 'pnl', 'hold_hours'
        ])

def load_history(symbol, hours=720):
    """Fetch historical data and compute indicators"""
    since = exchange.milliseconds() - int(hours * 3600 * 1000)
    all_bars = []
    
    while True:
        try:
            chunk = exchange.fetch_ohlcv(symbol, TIMEFRAME, since, limit=500)
            if not chunk:
                break
            all_bars += chunk
            since = chunk[-1][0] + 1
            time.sleep(exchange.rateLimit / 1000)
        except Exception as e:
            print(f"Error loading {symbol}: {e}")
            break
    
    if not all_bars:
        return None
    
    df = pd.DataFrame(all_bars, columns=['ts','open','high','low','close','vol'])
    df['dt'] = pd.to_datetime(df['ts'], unit='ms')
    df = df.set_index('dt').sort_index()
    
    # Calculate indicators
    df['prev_close'] = df['close'].shift(1)
    df['tr'] = np.maximum.reduce([
        df['high'] - df['low'],
        (df['high'] - df['prev_close']).abs(),
        (df['low'] - df['prev_close']).abs()
    ])
    df['atr20'] = df['tr'].rolling(20).mean()
    df['rv20'] = np.log(df['close']).diff().rolling(20).std()
    df['ma10'] = df['close'].rolling(10).mean()
    df['ma50'] = df['close'].rolling(50).mean()
    df['ma10_50'] = df['ma10'] - df['ma50']
    
    delta = df['close'].diff()
    up, down = delta.clip(lower=0), -delta.clip(upper=0)
    df['rsi14'] = 100 - 100/(1 + up.rolling(14).mean() / down.rolling(14).mean())
    df['vol_spike'] = df['vol'] / df['vol'].rolling(20).mean()
    
    return df.dropna()

def backtest_sltp(df, params):
    """Backtest ROC strategy with SL/TP exits"""
    roc_period = params['roc_period']
    threshold = params['threshold']
    sl_pct = params['sl_pct']
    tp_pct = params['tp_pct']
    
    trades = []
    positions = {}
    
    for i in range(roc_period, len(df)):
        current_time = df.index[i]
        close = df['close'].iloc[i]
        roc = close / df['close'].iloc[i-roc_period] - 1
        
        # Check for exits first
        for sym in list(positions.keys()):
            pos = positions[sym]
            entry_price = pos['entry_price']
            
            # Check TP
            if close >= entry_price * (1 + tp_pct):
                exit_price = entry_price * (1 + tp_pct)
                exit_type = 'TP'
            # Check SL
            elif close <= entry_price * (1 - sl_pct):
                exit_price = entry_price * (1 - sl_pct)
                exit_type = 'SL'
            else:
                continue
                
            # Calculate PnL with fees and slippage
            exit_price_adj = exit_price * (1 - FEE_RATE - SLIPPAGE_PCT/2)
            entry_price_adj = pos['entry_price'] * (1 + FEE_RATE + SLIPPAGE_PCT/2)
            qty = NOTIONAL / entry_price_adj
            pnl = (exit_price_adj - entry_price_adj) * qty
            hold_hours = (current_time - pos['entry_time']).total_seconds() / 3600
            
            trades.append({
                'symbol': sym,
                'entry_time': pos['entry_time'],
                'exit_time': current_time,
                'entry_price': pos['entry_price'],
                'exit_price': exit_price,
                'exit_type': exit_type,
                'roc': roc,
                'pnl': pnl,
                'hold_hours': hold_hours
            })
            del positions[sym]
        
        # Check for new entries
        if roc > threshold:
            entry_price_adj = close * (1 + FEE_RATE + SLIPPAGE_PCT/2)
            positions[symbol] = {
                'entry_time': current_time,
                'entry_price': close,
                'qty': NOTIONAL / entry_price_adj
            }
    
    return pd.DataFrame(trades)

# ─────────────── MAIN TRAINING LOGIC ────────────────────────────────────
print(f"→ Fetching historical data for {len(SYMBOLS)} symbols...")
history = {}
for sym in tqdm(SYMBOLS, desc='Loading data'):
    try:
        # Load ~30 days of data for training (720 hours)
        history[sym] = load_history(sym, hours=720)  
    except Exception as e:
        print(f"⚠️ Failed {sym}: {e}")

# Backtest all symbols with SL/TP to collect trades
all_trades = []
for symbol in tqdm(SYMBOLS, desc='Backtesting'):
    if symbol not in history or history[symbol] is None:
        continue
    
    params = BEST[symbol]
    trades = backtest_sltp(history[symbol], params)
    if not trades.empty:
        all_trades.append(trades)

if not all_trades:
    print("⚠️ No trades generated - check your parameters or data")
    sys.exit(1)

trades_df = pd.concat(all_trades)

# Build feature matrix
feat_rows = []
for _, trade in trades_df.iterrows():
    sym = trade['symbol']
    t0 = trade['entry_time']
    bar = history[sym].loc[t0]
    
    feat_rows.append({
        'roc': trade['roc'],
        'atr20': bar['atr20'],
        'rv20': bar['rv20'],
        'ma10_50': bar['ma10_50'],
        'rsi14': bar['rsi14'],
        'vol_spike': bar['vol_spike'],
        'hold_hours': trade['hold_hours'],
        'hour': t0.hour,
        'weekend': int(t0.weekday() >= 5),
        'sl_pct': BEST[sym]['sl_pct'],
        'tp_pct': BEST[sym]['tp_pct'],
        'y': int(trade['pnl'] > 0),  # 1 if profitable, 0 otherwise
    })

feat_df = pd.DataFrame(feat_rows).dropna()
X = feat_df.drop(columns=['y'])
y = feat_df['y']

# Train/test split and scale
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
scaler = StandardScaler().fit(X_train)
Xtr = scaler.transform(X_train)
Xte = scaler.transform(X_test)

# Train classifier
model = RandomForestClassifier(
    n_estimators=200,
    class_weight='balanced',
    random_state=42
)
model.fit(Xtr, y_train)
y_pred = model.predict(Xte)

print("\n=== Classification Report ===")
print(classification_report(y_test, y_pred))

# Persist models
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(model, 'classifier.pkl')
print("\n→ Saved scaler.pkl and classifier.pkl")

# ─────────────── LIVE TRADING LOOP (WITH DEBUG OUTPUT) ────────────────────────────────────
def log_trade(row):
    """Log trade to CSV file"""
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

def sleep_till_next():
    """Sleep until next candle"""
    now = datetime.utcnow()
    to_sleep = 60 - now.second
    time.sleep(max(to_sleep, 1))

print(f"\n▶️ Starting live trader (SL/TP version) - Dry run={DRY_RUN}")
positions = {}  # symbol -> {entry_time, entry_price, qty}

try:
    while True:
        now = datetime.utcnow()
        print(f"\n[{now}] Open positions: {len(positions)}")
        
        for symbol in tqdm(SYMBOLS, desc='Checking symbols'):
            params = BEST.get(symbol)
            if not params:
                continue
            
            # Fetch recent data
            try:
                df = load_history(symbol, hours=72)  # Enough for indicators
                if df is None or len(df) < 50:
                    print(f"⚠️ {symbol}: Insufficient data ({len(df) if df is not None else 0} bars)")
                    continue
            except Exception as e:
                print(f"Error fetching {symbol}: {e}")
                continue
            
            # Current price and ROC calculation
            roc_period = params['roc_period']
            if len(df) < roc_period + 1:
                continue
                
            close = df['close'].iloc[-1]
            roc = close / df['close'].iloc[-1-roc_period] - 1
            
            # Print detailed debug information
            print(f"\n{symbol} at {now}:")
            print(f"• Last ROC: {roc:.2%} (Threshold: {params['threshold']:.2%})")
            print(f"• Price: {close:.2f} | MA10: {df['ma10'].iloc[-1]:.2f} | RSI: {df['rsi14'].iloc[-1]:.2f}")
            print(f"• Volatility: ATR20={df['atr20'].iloc[-1]:.2f} RV20={df['rv20'].iloc[-1]:.4f}")
            
            # Check for exits first
            if symbol in positions:
                pos = positions[symbol]
                entry_price = pos['entry_price']
                exit_type = None
                exit_price = None
                
                # Check TP
                if close >= entry_price * (1 + params['tp_pct']):
                    exit_type = 'TP'
                    exit_price = entry_price * (1 + params['tp_pct'])
                # Check SL
                elif close <= entry_price * (1 - params['sl_pct']):
                    exit_type = 'SL'
                    exit_price = entry_price * (1 - params['sl_pct'])
                
                if exit_type:
                    # Calculate PnL with fees and slippage
                    exit_price_adj = exit_price * (1 - FEE_RATE - SLIPPAGE_PCT/2)
                    pnl = (exit_price_adj - pos['entry_price']) * pos['qty']
                    hold_hours = (now - pos['entry_time']).total_seconds() / 3600
                    
                    print(f"{now} ← EXIT {symbol} ({exit_type}) @ {exit_price:.4f} "
                          f"PNL: {pnl:.2f} ({hold_hours:.1f}h)")
                    
                    if not DRY_RUN:
                        try:
                            exchange.create_order(symbol, 'market', 'sell', pos['qty'])
                        except Exception as e:
                            print(f"Exit order failed: {e}")
                    
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
            
            # Check for new entries (if no position)
            if symbol not in positions and roc > params['threshold']:
                # Build feature vector
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
                
                # Predict probability of profit
                X = pd.DataFrame([feat])
                Xs = scaler.transform(X)
                proba = model.predict_proba(Xs)[0, 1]
                
                print(f"• Prob: {proba:.2f} (Threshold: {CLASSIFIER_THRESHOLD:.2f})")
                
                if proba >= CLASSIFIER_THRESHOLD:
                    entry_price_adj = close * (1 + FEE_RATE + SLIPPAGE_PCT/2)
                    qty = NOTIONAL / entry_price_adj
                    
                    print(f"✅ ENTERING TRADE @ {close:.2f} (Qty: {qty:.4f})")
                    
                    if not DRY_RUN:
                        try:
                            exchange.create_order(symbol, 'market', 'buy', qty)
                        except Exception as e:
                            print(f"Entry order failed: {e}")
                            continue
                    
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
                else:
                    print("⛔ Probability below threshold")
            else:
                if roc <= params['threshold']:
                    print(f"⛔ ROC below threshold ({roc:.2%} <= {params['threshold']:.2%})")
            
        sleep_till_next()

except KeyboardInterrupt:
    print("\n🛑 Stopped cleanly")
