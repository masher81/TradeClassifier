#!/usr/bin/env python3
"""
trade_roc.py

1) Fetch historical data, backtest your ROC strategy, build features, train classifier.
2) Run a live ROC-based trader, filtering new signals through the trained classifier.
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
from sklearn.ensemble    import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# ─────────────── CONFIG ────────────────────────────────────────────────
load_dotenv()
API_KEY      = os.getenv('KRAKEN_APIKEY')
API_SECRET   = os.getenv('KRAKEN_SECRET')
NOTIONAL     = float(os.getenv('NOTIONAL', 10_000))
FEE_RATE     = float(os.getenv('FEE_RATE', 0.001))
SLIPPAGE_PCT = float(os.getenv('SLIPPAGE_PCT', 0.0005))
DRY_RUN      = os.getenv('DRY_RUN', 'false').lower() in ('1','true')
CLASSIFIER_THRESHOLD = float(os.getenv('CLASSIFIER_THRESHOLD', 0.40))
LOG_FILE     = os.getenv('TRADE_LOG_FILE', 'trade_log.csv')

TIMEFRAME    = '1h'
START_TS     = '2024-06-26T00:00:00Z'
END_TS       = '2025-06-26T00:00:00Z'

# ─────────────── LOAD BEST PARAMS ───────────────────────────────────────
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
            'symbol','action','timestamp','price','qty','roc',
            'entry_time','entry_price','exit_time','exit_price','pnl'
        ])

def load_history(symbol):
    """Fetch full history between START_TS and END_TS and compute indicators."""
    since = exchange.parse8601(START_TS)
    end   = exchange.parse8601(END_TS)
    all_bars = []
    while since < end:
        chunk = exchange.fetch_ohlcv(symbol, TIMEFRAME, since, limit=500)
        if not chunk:
            break
        all_bars += chunk
        since = chunk[-1][0] + 1
    df = pd.DataFrame(all_bars, columns=['ts','open','high','low','close','vol'])
    df['dt'] = pd.to_datetime(df['ts'], unit='ms')
    df = df.set_index('dt').sort_index()

    # True range for ATR
    df['prev_close'] = df['close'].shift(1)
    df['tr'] = np.maximum.reduce([
        df['high'] - df['low'],
        (df['high'] - df['prev_close']).abs(),
        (df['low']  - df['prev_close']).abs(),
    ])
    # rolling indicators
    df['atr20']   = df['tr'].rolling(20).mean()
    df['rv20']    = np.log(df['close']).diff().rolling(20).std()
    df['ma10']    = df['close'].rolling(10).mean()
    df['ma50']    = df['close'].rolling(50).mean()
    df['ma10_50'] = df['ma10'] - df['ma50']
    # RSI(14)
    delta = df['close'].diff()
    up, down = delta.clip(lower=0), -delta.clip(upper=0)
    df['rsi14'] = 100 - 100/(1 + up.rolling(14).mean() / down.rolling(14).mean())
    # volume spike
    df['vol_spike'] = df['vol'] / df['vol'].rolling(20).mean()

    return df

# ─────────────── BACKTEST & BUILD FEATURE DATAFRAME ────────────────────
print(f"→ fetching 1h bars for {len(SYMBOLS)} symbols…")
history = {}
for sym in tqdm(SYMBOLS, desc='symbols'):
    try:
        history[sym] = load_history(sym)
    except Exception as e:
        print(f"⚠️ failed {sym}: {e}")

# backtest your ROC strategy to collect all trades
positions = {}
trades = []
master_idx = history[next(iter(history))].index

for now in tqdm(master_idx, desc='backtesting'):
    for sym in SYMBOLS:
        df = history[sym]
        if now not in df.index:
            continue

        params = BEST[sym]
        roc_p   = int(params.get('roc_period', params.get('roc', 1)))
        thr     = float(params.get('threshold', params.get('thr', 0)))
        hold_h  = int(params.get('hold_bars', params.get('hold', 1)))

        close = df.at[now, 'close']
        idx   = df.index.get_loc(now)
        if idx < roc_p:
            continue
        past = df['close'].iat[idx-roc_p]
        roc  = close/past - 1
        pos  = positions.get(sym)

        # EXIT
        if pos and now >= pos['exit_dt']:
            entry_p = pos['entry_price']
            qty     = pos['qty']
            exit_p  = close * (1 - FEE_RATE - SLIPPAGE_PCT/2)
            pnl     = (exit_p - entry_p)*qty
            trades.append({
                'symbol':   sym,
                'entry_dt': pos['entry_dt'],
                'exit_dt':  now,
                'roc':      roc,
                'pnl':      pnl,
                'hold_hrs': (now - pos['entry_dt']).total_seconds()/3600
            })
            del positions[sym]

        # ENTER
        elif not pos and roc > thr:
            entry_p = close * (1 + FEE_RATE + SLIPPAGE_PCT/2)
            qty     = NOTIONAL/entry_p
            exit_dt = now + timedelta(hours=hold_h)
            positions[sym] = {
                'entry_price': entry_p,
                'qty':         qty,
                'entry_dt':    now,
                'exit_dt':     exit_dt
            }

# build feature dataframe
trades_df = pd.DataFrame(trades).dropna(subset=['pnl'])
feat_rows = []
for _, row in trades_df.iterrows():
    sym  = row['symbol']
    t0   = row['entry_dt']
    bar  = history[sym].loc[t0]
    feat_rows.append({
        'roc':       row['roc'],
        'atr20':     bar['atr20'],
        'rv20':      bar['rv20'],
        'ma10_50':   bar['ma10_50'],
        'rsi14':     bar['rsi14'],
        'vol_spike': bar['vol_spike'],
        'hold_hrs':  row['hold_hrs'],
        'hour':      t0.hour,
        'weekend':   int(t0.weekday() >= 5),
        'y':         int(row['pnl'] > 0),
    })
feat_df = pd.DataFrame(feat_rows).dropna()
X = feat_df.drop(columns=['y'])
y = feat_df['y']

# ─────────────── TRAIN / EVALUATE ───────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
scaler = StandardScaler().fit(X_train)
Xtr = scaler.transform(X_train)
Xte = scaler.transform(X_test)

model = RandomForestClassifier(
    n_estimators=200,
    class_weight='balanced',
    random_state=42
)
model.fit(Xtr, y_train)
y_pred = model.predict(Xte)

print("\n=== Classification Report ===")
print(classification_report(y_test, y_pred))

# persist for later inspection
joblib.dump(scaler,    'scaler.pkl')
joblib.dump(model,     'classifier.pkl')
print("\n→ Saved scaler.pkl and classifier.pkl\n")

# ─────────────── LIVE TRADING LOOP ────────────────────────────────────
SCALER     = scaler
CLASSIFIER = model
positions = {}   # symbol → {entry_time, entry_price, qty, exit_time}

def fetch_bars(symbol, hours):
    since = exchange.milliseconds() - int((hours+2)*3600*1000)
    bars  = exchange.fetch_ohlcv(symbol, TIMEFRAME, since, limit=hours+2)
    df    = pd.DataFrame(bars, columns=['ts','open','high','low','close','vol'])
    df['dt'] = pd.to_datetime(df['ts'], unit='ms')
    return df.set_index('dt')

def sleep_till_next():
    now = datetime.utcnow()
    to_sleep = 60 - now.second
    time.sleep(max(to_sleep,1))

def log_trade(row):
    with open(LOG_FILE,'a',newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            row.get('symbol'),
            row.get('action'),
            row.get('timestamp'),
            row.get('price'),
            row.get('qty'),
            row.get('roc',''),
            row.get('entry_time',''),
            row.get('entry_price',''),
            row.get('exit_time',''),
            row.get('exit_price',''),
            row.get('pnl',''),
        ])

print(f"▶️  Starting live trader… Dry run={DRY_RUN}, classifier threshold={CLASSIFIER_THRESHOLD}")
try:
    while True:
        now = datetime.utcnow()
        sys.stdout.write(f"\n[{now.isoformat()}] open positions: {len(positions)}\n")
        sys.stdout.flush()

        for sym, params in tqdm(BEST.items(), desc='symbols', leave=False):
            # read ROC params
            roc_p  = int(params.get('roc_period', params.get('roc',1)))
            thr    = float(params.get('threshold', params.get('thr',0)))
            hold_h = int(params.get('hold_bars', params.get('hold',1)))

            # fetch enough bars for all indicators
            lookback = max(roc_p,50,20,14) + hold_h
            df = fetch_bars(sym, lookback)
            if len(df) < lookback:
                continue

            # recompute indicators on the fly
            df['prev_close'] = df['close'].shift(1)
            df['tr'] = np.maximum.reduce([
                df['high'] - df['low'],
                (df['high'] - df['prev_close']).abs(),
                (df['low']  - df['prev_close']).abs(),
            ])
            df['atr20']   = df['tr'].rolling(20).mean()
            df['rv20']    = np.log(df['close']).diff().rolling(20).std()
            df['ma10']    = df['close'].rolling(10).mean()
            df['ma50']    = df['close'].rolling(50).mean()
            df['ma10_50'] = df['ma10'] - df['ma50']
            delta = df['close'].diff()
            up, down = delta.clip(lower=0), -delta.clip(upper=0)
            df['rsi14']   = 100 - 100/(1 + up.rolling(14).mean()/down.rolling(14).mean())
            df['vol_spike'] = df['vol']/df['vol'].rolling(20).mean()

            t0    = df.index[-1]
            close = df['close'].iat[-1]
            roc   = close/df['close'].iat[-1-roc_p] - 1
            pos   = positions.get(sym)

            # EXIT
            if pos and now >= pos['exit_time']:
                qty       = pos['qty']
                exit_p    = close * (1 - FEE_RATE - SLIPPAGE_PCT/2)
                pnl       = (exit_p - pos['entry_price'])*qty
                ts        = now.isoformat()
                print(f"{ts} ← EXIT {sym:8}@{exit_p:.4f}  pnl={pnl:.2f}")
                if not DRY_RUN:
                    exchange.create_order(sym,'market','sell',qty)
                log_trade({
                    'symbol':     sym,
                    'action':     'EXIT',
                    'timestamp':  ts,
                    'price':      f"{exit_p:.4f}",
                    'qty':        f"{qty:.6f}",
                    'roc':        '',
                    'entry_time': pos['entry_time'],
                    'entry_price':f"{pos['entry_price']:.4f}",
                    'exit_time':  ts,
                    'exit_price': f"{exit_p:.4f}",
                    'pnl':        f"{pnl:.2f}",
                })
                del positions[sym]

            # ENTER
            elif not pos and roc > thr:
                # build feature row
                feat = pd.DataFrame([{
                    'roc':       roc,
                    'atr20':     df['atr20'].iat[-1],
                    'rv20':      df['rv20'].iat[-1],
                    'ma10_50':   df['ma10_50'].iat[-1],
                    'rsi14':     df['rsi14'].iat[-1],
                    'vol_spike': df['vol_spike'].iat[-1],
                    'hold_hrs':  hold_h,
                    'hour':      t0.hour,
                    'weekend':   int(t0.weekday()>=5),
                }])
                Xs    = SCALER.transform(feat)
                p_prof= CLASSIFIER.predict_proba(Xs)[0,1]
                if p_prof < CLASSIFIER_THRESHOLD:
                    # skip low-confidence signals
                    continue

                entry_p     = close * (1 + FEE_RATE + SLIPPAGE_PCT/2)
                qty         = NOTIONAL/entry_p
                exit_time   = t0 + timedelta(hours=hold_h)
                ts          = now.isoformat()
                print(f"{ts} → ENTER {sym:8}@{entry_p:.4f}  roc={roc:.4f}  P={p_prof:.2f}")
                if not DRY_RUN:
                    exchange.create_order(sym,'market','buy',qty)

                positions[sym] = {
                    'entry_time':  ts,
                    'entry_price': entry_p,
                    'qty':          qty,
                    'exit_time':    exit_time
                }
                log_trade({
                    'symbol':     sym,
                    'action':     'ENTER',
                    'timestamp':  ts,
                    'price':      f"{entry_p:.4f}",
                    'qty':        f"{qty:.6f}",
                    'roc':        f"{roc:.4f}",
                    'entry_time': ts,
                    'entry_price':f"{entry_p:.4f}",
                    'exit_time':  '',
                    'exit_price': '',
                    'pnl':        '',
                })

        sys.stdout.write(f"[{datetime.utcnow().isoformat()}] scan complete.\n")
        sys.stdout.flush()
        sleep_till_next()

except KeyboardInterrupt:
    print("\n📴  Stopped cleanly.")






