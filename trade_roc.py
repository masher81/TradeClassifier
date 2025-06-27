#!/usr/bin/env python3
import os
import time
import json
import numpy as np
import pandas as pd
import ccxt
import joblib

from datetime import datetime, timedelta
from dotenv import load_dotenv

# ─────────────── LOAD ENV ─────────────────────────────────────────
load_dotenv()
API_KEY      = os.getenv('KRAKEN_APIKEY')
API_SECRET   = os.getenv('KRAKEN_SECRET')
NOTIONAL     = float(os.getenv('NOTIONAL', 10_000))    # USD per trade
FEE_RATE     = float(os.getenv('FEE_RATE', 0.001))     # 0.1% per side
SLIPPAGE_PCT = float(os.getenv('SLIPPAGE_PCT', 0.0005)) # 0.05% round-trip
DRY_RUN      = os.getenv('DRY_RUN', 'false').lower() in ('1','true','yes')

# ─────────────── LOAD BEST PARAMS ─────────────────────────────────
with open('best_params.json') as f:
    BEST = json.load(f)

# ─────────────── LOAD CLASSIFIER ──────────────────────────────────
MODEL_PATH     = os.getenv('CLASSIFIER_PATH', 'classifier.pkl')
SCALER_PATH    = os.getenv('SCALER_PATH', None)  # optional
THRESHOLD      = float(os.getenv('CLASSIFIER_THRESHOLD', 0.40))

model  = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH) if SCALER_PATH else None

# ─────────────── EXCHANGE SETUP ───────────────────────────────────
TIMEFRAME = '1h'
exchange  = ccxt.kraken({
    'apiKey':         API_KEY,
    'secret':         API_SECRET,
    'enableRateLimit': True,
})
exchange.load_markets()

# track open positions: symbol → {'exit_time': datetime, 'qty': float}
positions = {}

def fetch_bars(symbol, hours):
    """Fetch at least `hours+2` bars so we can compute all indicators."""
    since = exchange.milliseconds() - int((hours + 2) * 3600 * 1000)
    bars  = exchange.fetch_ohlcv(symbol, TIMEFRAME, since)
    df    = pd.DataFrame(bars, columns=['ts','open','high','low','close','volume'])
    df['dt'] = pd.to_datetime(df['ts'], unit='ms')
    return df.set_index('dt')

def sleep_till_next():
    """Sleep until the top of the next minute (roughly)."""
    now = datetime.utcnow()
    to_sleep = 60 - now.second
    time.sleep(to_sleep if to_sleep > 0 else 1)

def compute_features(df, roc_p, hold_h):
    """
    Build feature vector for the classifier.
    Assumes df is indexed by datetime with columns: open, high, low, close, volume.
    """
    closes = df['close']
    highs  = df['high']
    lows   = df['low']
    vols   = df['volume']
    now    = df.index[-1]

    # 1) ROC (we recompute here for consistency)
    roc = closes.iloc[-1] / closes.iloc[-1-roc_p] - 1

    # 2) RSI14
    delta = closes.diff()
    gain  = delta.clip(lower=0)
    loss  = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean().iloc[-1]
    avg_loss = loss.rolling(14).mean().iloc[-1]
    rsi14 = 100 - 100 / (1 + (avg_gain / avg_loss)) if avg_loss != 0 else 100

    # 3) MA crossover
    ma10    = closes.rolling(10).mean().iloc[-1]
    ma50    = closes.rolling(50).mean().iloc[-1]
    ma10_50 = ma10 - ma50

    # 4) ATR20
    prev_close = closes.shift(1)
    tr1 = highs - lows
    tr2 = (highs - prev_close).abs()
    tr3 = (lows  - prev_close).abs()
    tr  = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr20 = tr.rolling(20).mean().iloc[-1]

    # 5) Realized vol 20
    rv20 = closes.pct_change().rolling(20).std().iloc[-1]

    # 6) Volume spike
    vol_spike = vols.iloc[-1] / vols.rolling(20).mean().iloc[-1]

    # 7) Time features
    hour    = now.hour
    weekend = int(now.weekday() >= 5)

    return {
        'roc':       roc,
        'rsi14':     rsi14,
        'ma10_50':   ma10_50,
        'atr20':     atr20,
        'rv20':      rv20,
        'vol_spike': vol_spike,
        'hour':      hour,
        'weekend':   weekend,
        'hold_hrs':  hold_h,
    }

def main():
    print(f"▶️  Starting ROC‐based trader… Dry run={DRY_RUN}, classifier threshold={THRESHOLD:.2f}")
    try:
        while True:
            now = datetime.utcnow()

            # 1) fetch up-to-date bars for each symbol
            dfs = {}
            for symbol, params in BEST.items():
                roc_p   = int(params.get('roc_period', params.get('roc', 1)))
                hold_h  = int(params.get('hold_bars', params.get('hold', 1)))
                try:
                    dfs[symbol] = fetch_bars(symbol, roc_p + hold_h)
                except Exception as e:
                    print(f"⚠️ {symbol} fetch failed: {e}")

            # 2) process each symbol
            for symbol, params in BEST.items():
                df = dfs.get(symbol)
                if df is None:
                    continue

                # read params
                roc_p  = int(params.get('roc_period', params.get('roc', 1)))
                thr    = float(params.get('threshold', params.get('thr', 0)))
                hold_h = int(params.get('hold_bars', params.get('hold', 1)))

                # need enough data
                if len(df) < max(50, roc_p, 20):
                    continue

                closes = df['close']
                roc     = closes.iloc[-1] / closes.iloc[-1-roc_p] - 1
                last_ts = df.index[-1]
                pos     = positions.get(symbol)

                # ────────── EXIT ──────────
                if pos and now >= pos['exit_time']:
                    qty       = pos['qty']
                    exit_price = closes.iloc[-1] * (1 - FEE_RATE - SLIPPAGE_PCT/2)
                    print(f"{now.isoformat()} ← EXIT  {symbol:8} @ {exit_price:.4f}  qty={qty:.6f}")
                    if DRY_RUN:
                        print(f"  [DRY_RUN] would SELL {symbol} qty={qty:.6f}")
                    else:
                        exchange.create_order(symbol, 'market', 'sell', qty)
                    del positions[symbol]

                # ────────── ENTER ──────────
                elif (symbol not in positions) and (roc > thr):
                    # compute classifier probability
                    try:
                        feats = compute_features(df, roc_p, hold_h)
                        X     = pd.DataFrame([feats])[[
                            'roc','rsi14','ma10_50','atr20','rv20',
                            'vol_spike','hour','weekend','hold_hrs'
                        ]]
                        if scaler is not None:
                            X = scaler.transform(X)
                        prob = model.predict_proba(X)[0,1]
                    except Exception as e:
                        print(f"⚠️ feature error for {symbol}: {e}")
                        continue

                    print(f"{now.isoformat()} → SIGNAL {symbol:8}  roc={roc:.4f}  clf_prob={prob:.2f}")
                    if prob < THRESHOLD:
                        print(f"  ↳ prob {prob:.2f} < {THRESHOLD:.2f}, skipping")
                        continue

                    # pass both ROC and classifier
                    entry_price = closes.iloc[-1] * (1 + FEE_RATE + SLIPPAGE_PCT/2)
                    qty         = NOTIONAL / entry_price
                    print(f"{now.isoformat()} → ENTER  {symbol:8} @ {entry_price:.4f}  qty={qty:.6f}")
                    if DRY_RUN:
                        print(f"  [DRY_RUN] would BUY  {symbol} qty={qty:.6f}")
                    else:
                        exchange.create_order(symbol, 'market', 'buy', qty)
                    exit_time = last_ts + timedelta(hours=hold_h)
                    positions[symbol] = {'exit_time': exit_time, 'qty': qty}

            sleep_till_next()

    except KeyboardInterrupt:
        print("📴  Stopping cleanly…")

if __name__ == '__main__':
    main()
