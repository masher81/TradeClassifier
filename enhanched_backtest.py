#!/usr/bin/env python3
import os, json
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import ccxt
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier

# ───────── CONFIG ──────────────────────────────────────────────────────
TIMEFRAME    = '1h'
NOTIONAL     = float(os.getenv('NOTIONAL', 10_000))
FEE_RATE     = float(os.getenv('FEE_RATE', 0.001))
SLIPPAGE_PCT = float(os.getenv('SLIPPAGE_PCT', 0.0005))

# load your optimized ROC params
with open('best_params.json') as f:
    BEST = json.load(f)
SYMBOLS = list(BEST.keys())

# initialize exchange (public only)
exchange = ccxt.kraken({'enableRateLimit': True})
exchange.load_markets()


# ──────────── 1) fetch OHLCV history ───────────────────────────────────
def load_history(symbol, start_ts, end_ts):
    since = exchange.parse8601(start_ts)
    end   = exchange.parse8601(end_ts)
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

START_TS = '2024-06-26T00:00:00Z'
END_TS   = '2025-06-26T00:00:00Z'

print(f"→ fetching {TIMEFRAME} bars for {len(SYMBOLS)} symbols…")
history = {}
for sym in tqdm(SYMBOLS):
    try:
        history[sym] = load_history(sym, START_TS, END_TS)
    except Exception as e:
        print(f"⚠️  failed to fetch {sym}: {e}")

# ──────────── 2) backtest ROC strategy → trades_df ───────────────────
trades = []
positions = {}

for sym, df in history.items():
    params = BEST[sym]
    roc_p  = int(params.get('roc_period', params.get('roc', 1)))
    thr    = float(params.get('threshold', params.get('thr', 0)))
    hold_h = int(params.get('hold_bars', params.get('hold', 1)))

    for now in df.index:
        price = df.at[now, 'close']
        idx   = df.index.get_loc(now)
        if idx < roc_p: 
            continue
        past  = df['close'].iat[idx - roc_p]
        roc   = price / past - 1
        pos   = positions.get(sym)

        # EXIT
        if pos and now >= pos['exit_dt']:
            entry_p = pos['entry_price']
            qty     = pos['qty']
            exit_p  = price * (1 - FEE_RATE - SLIPPAGE_PCT/2)
            pnl     = (exit_p - entry_p) * qty
            trades.append({
                'symbol': sym, 'entry_time': pos['entry_time'],
                'exit_time': now, 'entry_price': entry_p,
                'exit_price': exit_p, 'qty': qty,
                'roc':  pos['roc'], 'hold_hrs': pos['hold_hrs'],
                'pnl': pnl
            })
            del positions[sym]

        # ENTER
        elif (not pos) and (roc > thr):
            entry_p   = price * (1 + FEE_RATE + SLIPPAGE_PCT/2)
            qty       = NOTIONAL / entry_p
            exit_dt   = now + timedelta(hours=hold_h)
            positions[sym] = {
                'entry_time': now, 'entry_price': entry_p,
                'qty': qty, 'roc': roc, 'hold_hrs': hold_h,
                'exit_dt': exit_dt
            }

trades_df = pd.DataFrame(trades)
print(f"→ generated {len(trades_df)} completed trades")

# ──────────── 3) feature engineering ─────────────────────────────────
def compute_features(row):
    sym = row.symbol
    df  = history[sym]
    i   = df.index.get_loc(row.entry_time)
    window20 = df.iloc[max(0, i-20):i+1]
    window14 = df.iloc[max(0, i-14):i+1]

    # ATR20
    h, l, c = window20.high, window20.low, window20.close.shift()
    tr = pd.concat([
        window20.high - window20.low,
        (window20.high - c).abs(),
        (window20.low  - c).abs(),
    ], axis=1).max(axis=1)
    atr20 = tr.mean()

    # realized vol20
    logr = np.log(window20.close / window20.close.shift()).dropna()
    rv20 = logr.std() * np.sqrt(len(logr))

    # MA10-50
    ma10   = df.close.rolling(10).mean().iat[i]
    ma50   = df.close.rolling(50).mean().iat[i]
    ma10_50 = ma10 - ma50

    # RSI14
    delta = window14.close.diff().dropna()
    up, down = delta.clip(lower=0), (-delta).clip(lower=0)
    rs = up.mean() / down.mean() if down.mean()>0 else np.nan
    rsi14 = 100 - (100 / (1 + rs))

    # vol spike
    recent_vol = window20.vol.iloc[:-1].mean() if len(window20)>1 else np.nan
    vol_spike = df.vol.iat[i] / recent_vol if recent_vol>0 else np.nan

    # time features
    hour    = row.entry_time.hour
    weekend = int(row.entry_time.weekday()>=5)

    return {
      'pnl':       row.pnl,
      'roc':       row.roc,
      'hold_hrs':  row.hold_hrs,
      'atr20':     atr20,
      'rv20':      rv20,
      'ma10_50':   ma10_50,
      'rsi14':     rsi14,
      'vol_spike': vol_spike,
      'hour':      hour,
      'weekend':   weekend,
    }

feat_rows = []
for _, row in tqdm(trades_df.iterrows(), total=len(trades_df), desc="Engineering features"):
    try:
        feat_rows.append(compute_features(row))
    except Exception:
        pass

feat_df = pd.DataFrame(feat_rows).dropna()
print(f"→ built feature set of {len(feat_df)} rows")

# target & inputs
feat_df['y'] = (feat_df.pnl > 0).astype(int)
X = feat_df.drop(columns=['pnl','y'])
y = feat_df.y.values

# ──────────── 4) train classifier ──────────────────────────────────────
model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
model.fit(X, y)
feat_df['prob'] = model.predict_proba(X)[:,1]

# ──────────── 5) threshold scan ────────────────────────────────────────
ths = np.linspace(0.0,1.0,51)
results = []
for t in ths:
    mask = feat_df.prob >= t
    net = feat_df.loc[mask, 'pnl'].sum()
    cnt = mask.sum()
    results.append({'threshold': t, 'net_pnl': net, 'trades': cnt})

res_df = pd.DataFrame(results)
opt = res_df.loc[res_df.net_pnl.idxmax()]
print(f"\n→ optimal threshold: {opt.threshold:.2f}  net PnL: ${opt.net_pnl:,.2f} over {int(opt.trades)} trades")

# ──────────── 6) plotting ─────────────────────────────────────────────
plt.figure()
plt.plot(res_df.threshold, res_df.net_pnl)
plt.title("Threshold vs. Net PnL")
plt.xlabel("Probability Threshold")
plt.ylabel("Net PnL")
plt.grid(True)

# equity curves
def equity_curve(df, mask=None):
    if mask is not None:
        df = df.loc[mask]
    df = df.sort_index()
    return df.pnl.cumsum()

base_eq = equity_curve(feat_df)
filt_eq = equity_curve(feat_df, feat_df.prob >= opt.threshold)

plt.figure()
plt.plot(base_eq, label='baseline')
plt.plot(filt_eq, label=f'filtered ≥{opt.threshold:.2f}')
plt.title("Equity Curves")
plt.xlabel("Trade index")
plt.ylabel("Cumulative PnL")
plt.legend()
plt.grid(True)

plt.show()
