#!/usr/bin/env python3
import json
from datetime import timedelta
import numpy as np
import pandas as pd
import ccxt
from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report, roc_curve
from sklearn.calibration import CalibratedClassifierCV
from imblearn.over_sampling import SMOTE
import xgboost as xgb

# ─────────────── CONFIG ────────────────────────────────────────────────
TIMEFRAME  = '1h'
START_TS   = '2024-06-26T00:00:00Z'
END_TS     = '2025-06-26T00:00:00Z'
ROC_CSV    = 'hist_trade_roc_trades.csv'
BEST_JSON  = 'best_params.json'

# ─────────────── LOAD BEST PARAMS & SYMBOLS ─────────────────────────────
with open(BEST_JSON) as f:
    BEST = json.load(f)
SYMBOLS = list(BEST.keys())

# ─────────────── INITIALIZE EXCHANGE ─────────────────────────────────────
exchange = ccxt.kraken({'enableRateLimit': True})
exchange.load_markets()

# ─────────────── FETCH PRICE HISTORY ────────────────────────────────────
def load_history(sym):
    since = exchange.parse8601(START_TS)
    end   = exchange.parse8601(END_TS)
    all_bars = []
    while since < end:
        chunk = exchange.fetch_ohlcv(sym, TIMEFRAME, since, limit=500)
        if not chunk:
            break
        all_bars += chunk
        since = chunk[-1][0] + 1
    df = pd.DataFrame(all_bars, columns=['ts','open','high','low','close','vol'])
    df['dt'] = pd.to_datetime(df['ts'], unit='ms')
    return df.set_index('dt')

print(f"→ fetching {TIMEFRAME} bars for {len(SYMBOLS)} symbols…")
history = {}
for sym in tqdm(SYMBOLS, desc="Price history"):
    try:
        history[sym] = load_history(sym)
    except Exception as e:
        tqdm.write(f"⚠️  failed to fetch {sym}: {e}")
print("→ done fetching history.\n")

# ─────────────── LOAD TRADE LOG ────────────────────────────────────────
tr = pd.read_csv(ROC_CSV, parse_dates=['time'])
tr['trade_id'] = (tr.action=='ENTER').groupby(tr.symbol).cumsum()
entries = tr[tr.action=='ENTER'].rename(columns={
    'time':'entry_time',
    'price':'entry_price',
})
exits   = tr[tr.action=='EXIT'].rename(columns={
    'time':'exit_time',
    'price':'exit_price',
    # 'pnl' stays as 'pnl'
})
df = pd.merge(
    entries,
    exits[['symbol','trade_id','exit_time','exit_price','pnl']],
    on=['symbol','trade_id'],
    how='inner',
)

# ─────────────── FEATURE ENGINEERING ───────────────────────────────────
def add_features(df):
    feats = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Engineering features"):
        try:
            sym = row['symbol']
            hist = history.get(sym)
            t0  = row['entry_time']
            win = hist.loc[:t0].iloc[-100:]
            if len(win) < 50:
                continue

            close = win['close']
            high  = win['high']
            low   = win['low']
            vol   = win['vol']

            # ATR-20
            tr1 = pd.concat([
                high - low,
                (high - close.shift(1)).abs(),
                (low  - close.shift(1)).abs(),
            ], axis=1).max(axis=1)
            atr20 = tr1.rolling(20).mean().iloc[-1]

            # Realized vol (20)
            rv20 = close.pct_change().rolling(20).std().iloc[-1]

            # MA10 vs MA50
            ma10 = close.rolling(10).mean().iloc[-1]
            ma50 = close.rolling(50).mean().iloc[-1]
            ma_diff = (ma10 - ma50) / ma50

            # RSI-14
            delta = close.diff()
            up, down = delta.clip(lower=0), -delta.clip(upper=0)
            ema_up = up.ewm(com=13, adjust=False).mean()
            ema_dn = down.ewm(com=13, adjust=False).mean()
            rsi14 = 100 - 100/(1 + ema_up/ema_dn)
            rsi14 = rsi14.iloc[-1]

            # MACD(12,26) & signal(9)
            ema12 = close.ewm(span=12, adjust=False).mean()
            ema26 = close.ewm(span=26, adjust=False).mean()
            macd  = ema12 - ema26
            sig   = macd.ewm(span=9, adjust=False).mean()
            macd_val, sig_val = macd.iloc[-1], sig.iloc[-1]

            # Volume spike
            vol_spike = vol.iloc[-1] / vol.rolling(20).mean().iloc[-1] - 1

            # Time features
            hour    = t0.hour
            dow     = t0.weekday()
            weekend = int(dow >= 5)

            # Hold hours
            hold_hrs = (row['exit_time'] - row['entry_time']).total_seconds() / 3600.

            feats.append({
                'symbol':     sym,
                'roc':        row['roc'],
                'atr20':      atr20,
                'rv20':       rv20,
                'ma_diff':    ma_diff,
                'rsi14':      rsi14,
                'macd':       macd_val,
                'macd_sig':   sig_val,
                'vol_spike':  vol_spike,
                'hour':       hour,
                'weekend':    weekend,
                'hold_hrs':   hold_hrs,
                'pnl':        row['pnl'],
            })
        except Exception as e:
            tqdm.write(f"⚠️  skipping trade due to error: {e}")
    return pd.DataFrame(feats)

feat_df = add_features(df).dropna()
feat_df['y'] = (feat_df['pnl'] > 0).astype(int)

# ─────────────── PREPARE DATA ───────────────────────────────────────────
X = feat_df.drop(['symbol','pnl','y'], axis=1)
y = feat_df['y']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)

# SMOTE
sm = SMOTE(random_state=42)
X_tr, y_tr = sm.fit_resample(X_train, y_train)

# ─────────────── TRAIN & CALIBRATE ─────────────────────────────────────
model = xgb.XGBClassifier(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=4,
    use_label_encoder=False,
    eval_metric='logloss',
    random_state=42,
)
clf = CalibratedClassifierCV(model, method='isotonic', cv=3)
print("\n→ training & calibrating model…")
clf.fit(X_tr, y_tr)

# ─────────────── EVALUATION ─────────────────────────────────────────────
y_pred_proba = clf.predict_proba(X_test)[:,1]
auc = roc_auc_score(y_test, y_pred_proba)
print(f"\nROC-AUC on test set: {auc:.4f}\n")
print(classification_report(y_test, clf.predict(X_test)))

fpr, tpr, thresh = roc_curve(y_test, y_pred_proba)
opt_idx    = np.argmax(tpr - fpr)
opt_thresh = thresh[opt_idx]
print(f"Optimal threshold ≈ {opt_thresh:.3f}\n")

# ─────────────── OVERLAY BACKTEST ───────────────────────────────────────
feat_df['pred_proba'] = clf.predict_proba(X)[:,1]
mask = feat_df['pred_proba'] >= opt_thresh
sim_pnl = feat_df.loc[mask,'pnl'].sum()

print(f"Overlay PnL (p>={opt_thresh:.2f}): ${sim_pnl:,.2f}")

