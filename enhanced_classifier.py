#!/usr/bin/env python3
import json
import os
from datetime import timedelta
import numpy as np
import pandas as pd
import ccxt
import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble    import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# ─────────────── PROGRESS BAR FALLBACK ────────────────────────────────────
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, **kwargs): 
        return x

# ─────────────── CONFIG ──────────────────────────────────────────────────
NOTIONAL     = 10_000         # USD per trade
FEE_RATE     = 0.001          # 0.1% per side
SLIPPAGE_PCT = 0.0005         # 0.05% round-trip
TIMEFRAME    = '1h'
START_TS     = '2024-06-26T00:00:00Z'
END_TS       = '2025-06-26T00:00:00Z'

# ─────────────── LOAD BEST PARAMS ────────────────────────────────────────
with open('best_params.json') as f:
    BEST = json.load(f)
SYMBOLS = list(BEST.keys())

# ─────────────── EXCHANGE SETUP ─────────────────────────────────────────
exchange = ccxt.kraken({'enableRateLimit': True})
exchange.load_markets()

def load_history(symbol):
    """Fetch full history between START_TS and END_TS."""
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
    # true range for ATR
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

print(f"→ fetching 1h bars for {len(SYMBOLS)} symbols…")
history = {}
for sym in tqdm(SYMBOLS):
    try:
        history[sym] = load_history(sym)
    except Exception as e:
        print(f"⚠️  failed to fetch {sym}: {e}")

# ─────────────── BACKTEST LOOP ──────────────────────────────────────────
positions = {}   # symbol → {entry_price, qty, entry_dt, exit_dt}
trades    = []   # list of all ENTER/EXIT events

master_idx = history[next(iter(history))].index
for now in tqdm(master_idx):
    for sym in SYMBOLS:
        df = history.get(sym)
        if df is None or now not in df.index:
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
        roc  = close / past - 1
        pos  = positions.get(sym)

        # EXIT
        if pos and now >= pos['exit_dt']:
            entry = pos['entry_price']
            qty   = pos['qty']
            exit_p= close * (1 - FEE_RATE - SLIPPAGE_PCT/2)
            pnl   = (exit_p - entry) * qty
            trades.append({
                'symbol':    sym,
                'entry_dt':  pos['entry_dt'],
                'exit_dt':   now,
                'entry_p':   entry,
                'exit_p':    exit_p,
                'qty':       qty,
                'roc':       roc,
                'pnl':       pnl,
                'hold_hrs':  (now - pos['entry_dt']).total_seconds()/3600,
            })
            del positions[sym]

        # ENTER
        elif not pos and roc > thr:
            entry_p = close * (1 + FEE_RATE + SLIPPAGE_PCT/2)
            qty     = NOTIONAL / entry_p
            exit_dt = now + timedelta(hours=hold_h)
            positions[sym] = {
                'entry_price': entry_p,
                'qty':         qty,
                'entry_dt':    now,
                'exit_dt':     exit_dt,
            }

# ─────────────── BUILD FEATURE DATAFRAME ────────────────────────────────
trades_df = pd.DataFrame(trades)
# drop any weird incomplete rows
trades_df = trades_df.dropna(subset=['pnl'])

feat_rows = []
for _, row in trades_df.iterrows():
    sym    = row['symbol']
    t0     = row['entry_dt']
    bar    = history[sym].loc[t0]
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
        'pnl_pos':   int(row['pnl'] > 0),
    })

feat_df = pd.DataFrame(feat_rows).dropna()
X = feat_df.drop(columns=['pnl_pos'])
y = feat_df['pnl_pos']

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

# ─────────────── PERSIST ARTIFACTS ──────────────────────────────────────
joblib.dump(scaler,        'scaler.pkl')
joblib.dump(model,         'classifier.pkl')
print("\n→ Saved scaler.pkl and classifier.pkl")



