#!/usr/bin/env python3
import json
import numpy as np
import pandas as pd
import ccxt
from datetime import timedelta
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# ─────────────── CONFIG ──────────────────────────────────────────────────
NOTIONAL      = 10_000         # USD per trade
FEE_RATE      = 0.001          # 0.1% per side
SLIPPAGE_PCT  = 0.0005         # 0.05% round-trip
TIMEFRAME     = '1h'
START_TS      = '2024-06-26T00:00:00Z'
END_TS        = '2025-06-26T00:00:00Z'

# ─────────────── LOAD PARAMS ─────────────────────────────────────────────
with open('best_params.json') as f:
    BEST = json.load(f)
SYMBOLS = list(BEST.keys())

# ─────────────── INIT EXCHANGE ──────────────────────────────────────────
exchange = ccxt.kraken({'enableRateLimit': True})
exchange.load_markets()

# ─────────────── FETCH HISTORICAL DATA ───────────────────────────────────
def load_history(symbol):
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
    return df.set_index('dt')

print(f"→ fetching {TIMEFRAME} bars for {len(SYMBOLS)} symbols…")
history = {}
for sym in tqdm(SYMBOLS, desc='Price history'):
    try:
        history[sym] = load_history(sym)
    except Exception as e:
        print(f"⚠️  failed to fetch {sym}: {e}")
print("→ done fetching history.\n")

# ─────────────── LOAD AND FILTER TRADES ─────────────────────────────────
trades_df = pd.read_csv('hist_trade_roc_trades.csv', parse_dates=['time'])
print(f"→ loaded {len(trades_df)} total trade‐events")
# keep only completed (exit) trades
trades_df = trades_df.dropna(subset=['pnl']).reset_index(drop=True)
print(f"→ {len(trades_df)} completed trades with PnL\n")

# ─────────────── FEATURE ENGINEERING ────────────────────────────────────
def add_features(trades_df, history):
    feats = []
    for _, row in tqdm(trades_df.iterrows(),
                       total=len(trades_df),
                       desc='Engineering features'):
        sym        = row['symbol']
        enter_time = row['time']
        df         = history.get(sym)
        if df is None or enter_time not in df.index:
            continue

        idx = df.index.get_loc(enter_time)
        if idx < 50:
            continue  # need at least 50 bars for MA50

        # Basic ROI momentum
        roc = row['roc']

        # 20-bar ATR
        window20 = df.iloc[idx-20:idx]
        tr20     = (window20['high'] - window20['low'])
        atr20    = tr20.mean()

        # Realized vol: std of log returns
        rets20   = np.log(df['close'].iloc[idx-20:idx] /
                           df['close'].iloc[idx-21:idx-1])
        rv20     = rets20.std()

        # MA10 vs MA50
        ma10 = df['close'].iloc[idx-10:idx].mean()
        ma50 = df['close'].iloc[idx-50:idx].mean()

        # RSI14
        deltas = df['close'].diff().iloc[idx-14:idx]
        ups    = deltas[deltas>0].sum()/14
        downs  = -deltas[deltas<0].sum()/14
        rsi14  = 100 - 100 / (1 + (ups/downs if downs>0 else np.inf))

        # Volume spike vs 20-bar average
        vol_spike = df['vol'].iloc[idx] / df['vol'].iloc[idx-20:idx].mean()

        # Time features
        hour    = enter_time.hour
        weekend = int(enter_time.weekday() >= 5)

        # Actual hold duration (hours)
        # here we assume that this row is an EXIT event, so row.time is exit time.
        # If you logged entry_time separately, swap in that timestamp here.
        hold_hrs = row.get('hold_h', np.nan)

        feats.append({
            'roc':       roc,
            'atr20':     atr20,
            'rv20':      rv20,
            'ma10_50':   ma10 - ma50,
            'rsi14':     rsi14,
            'vol_spike': vol_spike,
            'hour':      hour,
            'weekend':   weekend,
            'hold_hrs':  hold_hrs,
            'pnl':       row['pnl'],
        })
    return pd.DataFrame(feats)

feat_df = add_features(trades_df, history)
feat_df = feat_df.dropna().reset_index(drop=True)
print(f"\n→ built feature set of {len(feat_df)} rows\n")

# ─────────────── TRAIN / EVALUATE ───────────────────────────────────────
# binary target: profitable (>0) or not
y = (feat_df['pnl'] > 0).astype(int)
X = feat_df.drop(columns=['pnl'])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

clf = RandomForestClassifier(
    n_estimators=100,
    class_weight='balanced',
    random_state=42,
)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print("=== Classification Report ===")
print(classification_report(y_test, y_pred))

importances = pd.Series(clf.feature_importances_, index=X.columns)
print("\n=== Feature Importances ===")
for feat, imp in importances.sort_values(ascending=False).items():
    print(f"  {feat:<10}: {imp:.4f}")


