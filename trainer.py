#!/usr/bin/env python3
import os
import json
from datetime import datetime, timedelta

import ccxt
import joblib
import numpy as np
import pandas as pd
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

TIMEFRAME    = '1h'
HISTORY_DAYS = 365   # fetch one year of data
HISTORY_HOURS= HISTORY_DAYS * 24

# ─────────────── LOAD BEST PARAMS ───────────────────────────────────────
BEST_PARAMS_FILE = 'best_params.json'
with open(BEST_PARAMS_FILE) as f:
    BEST = json.load(f)
SYMBOLS = list(BEST.keys())

# ─────────────── EXCHANGE SETUP ─────────────────────────────────────────
exchange = ccxt.kraken({
    'enableRateLimit': True,
    'apiKey':          API_KEY,
    'secret':          API_SECRET,
})
exchange.load_markets()

# ─────────────── FETCH & INDICATORS ────────────────────────────────────
def fetch_ohlcv(symbol, since_ms):
    all_bars = []
    while True:
        chunk = exchange.fetch_ohlcv(symbol, TIMEFRAME, since_ms, limit=500)
        if not chunk:
            break
        all_bars += chunk
        since_ms = chunk[-1][0] + 1
        # stop once we've covered HISTORY_HOURS+2 bars
        if len(all_bars) >= HISTORY_HOURS + 2:
            break
    return all_bars

def compute_indicators(df):
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
    df['vol_spike'] = df['volume'] / df['volume'].rolling(20).mean()
    return df.dropna()

# ─────────────── BACKTEST SL/TP ────────────────────────────────────────
def backtest_sltp(df, params, symbol):
    roc_p   = params['roc_period']
    thr     = params['threshold']
    sl_pct  = params['sl_pct']
    tp_pct  = params['tp_pct']

    trades = []
    in_pos = False
    entry = {}

    closes = df['close'].values
    times  = df.index

    for i in range(roc_p, len(df)):
        t = times[i]
        price = closes[i]
        roc = price / closes[i-roc_p] - 1

        # exit if in position
        if in_pos:
            etype = None
            if price >= entry['price']*(1+tp_pct):
                exit_price = entry['price']*(1+tp_pct)
                etype = 'TP'
            elif price <= entry['price']*(1-sl_pct):
                exit_price = entry['price']*(1-sl_pct)
                etype = 'SL'

            if etype:
                # adjust for fees+slippage
                exit_adj = exit_price*(1 - FEE_RATE - SLIPPAGE_PCT/2)
                entry_adj= entry['price']*(1 + FEE_RATE + SLIPPAGE_PCT/2)
                qty      = entry['qty']
                pnl      = (exit_adj - entry_adj)*qty
                hold_h   = (t - entry['time']).total_seconds()/3600
                trades.append({
                    'symbol':     symbol,
                    'entry_time': entry['time'],
                    'exit_time':  t,
                    'entry_price':entry['price'],
                    'exit_price': exit_price,
                    'exit_type':  etype,
                    'roc':        roc,
                    'pnl':        pnl,
                    'hold_hours': hold_h
                })
                in_pos = False

        # entry logic
        if (not in_pos) and (roc > thr):
            entry = {
                'time':  t,
                'price': price,
                'qty':   NOTIONAL / (price*(1 + FEE_RATE + SLIPPAGE_PCT/2))
            }
            in_pos = True

    return pd.DataFrame(trades)

# ─────────────── MAIN TRAINING PIPELINE ────────────────────────────────
def main():
    history = {}
    all_trades = []

    print(f"→ Fetching + processing history for {len(SYMBOLS)} symbols…")
    for sym in tqdm(SYMBOLS, desc="symbols"):
        since = exchange.milliseconds() - HISTORY_HOURS*3600*1000
        raw = fetch_ohlcv(sym, since)
        if not raw:
            continue
        df = pd.DataFrame(raw, columns=['ts','open','high','low','close','volume'])
        df['dt'] = pd.to_datetime(df['ts'], unit='ms')
        df.set_index('dt', inplace=True)
        df = compute_indicators(df)
        if df.empty:
            continue

        history[sym] = df
        trades = backtest_sltp(df, BEST[sym], sym)
        if not trades.empty:
            all_trades.append(trades)

    if not all_trades:
        raise RuntimeError("⚠️  No trades generated in backtest – check your params/data")

    trades_df = pd.concat(all_trades, ignore_index=True)

    # ─── build feature DataFrame ───
    feat_rows = []
    for _, row in trades_df.iterrows():
        sym = row['symbol']
        t0  = row['entry_time']
        bar = history[sym].loc[t0]
        feat_rows.append({
            'roc':       row['roc'],
            'atr20':     bar['atr20'],
            'rv20':      bar['rv20'],
            'ma10_50':   bar['ma10_50'],
            'rsi14':     bar['rsi14'],
            'vol_spike': bar['vol_spike'],
            'hold_hours':row['hold_hours'],
            'hour':      t0.hour,
            'weekend':   int(t0.weekday()>=5),
            'sl_pct':    BEST[sym]['sl_pct'],
            'tp_pct':    BEST[sym]['tp_pct'],
            'y':         int(row['pnl']>0),
        })

    feat_df = pd.DataFrame(feat_rows).dropna()
    X = feat_df.drop(columns=['y'])
    y = feat_df['y']

    # ─── train / evaluate ───
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    scaler = StandardScaler().fit(Xtr)
    model  = RandomForestClassifier(
        n_estimators=200,
        class_weight='balanced',
        random_state=42
    )
    model.fit(scaler.transform(Xtr), ytr)

    ypr = model.predict(scaler.transform(Xte))
    print("\n=== Classification Report ===")
    print(classification_report(yte, ypr))

    # ─── persist ───
    joblib.dump(scaler,    'scaler.pkl')
    joblib.dump(model,     'classifier.pkl')
    print("\n→ Saved scaler.pkl and classifier.pkl")

if __name__ == '__main__':
    main()
