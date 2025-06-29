#!/usr/bin/env python3
"""
trade_roc_sltop.py — 2-resolution live trading:
  • 1 h ROC + classifier entries
  • 5 m SL/TP exits
"""

import os, sys, time, csv, json
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

# ─────────────── CONFIG ─────────────────────────────────────────────────

load_dotenv()
API_KEY               = os.getenv('KRAKEN_APIKEY')
API_SECRET            = os.getenv('KRAKEN_SECRET')
NOTIONAL              = float(os.getenv('NOTIONAL', 10_000))
FEE_RATE              = float(os.getenv('FEE_RATE', 0.001))
SLIPPAGE_PCT          = float(os.getenv('SLIPPAGE_PCT', 0.0005))
DRY_RUN               = os.getenv('DRY_RUN','').lower() in ('1','true')
CLASSIFIER_THRESHOLD  = float(os.getenv('CLASSIFIER_THRESHOLD', 0.40))
LOG_FILE              = os.getenv('TRADE_LOG_FILE', 'trade_log.csv')

# Two timeframes
ENTRY_TIMEFRAME       = '1h'    # ROC + features
EXIT_TIMEFRAME        = '5m'    # SL/TP checks

# Caching & filtering
HISTORY_HOURS         = 168     # how many hours of 1h data to cache
MIN_DATA_BARS         = 50
MIN_AVG_VOLUME        = 100
CACHE_FILE            = 'hourly_cache.pkl'
CACHE_EXPIRY          = timedelta(minutes=30)
MAX_WORKERS           = 5
VERBOSE               = True

# ─────────────── LOAD PARAMS & MODELS ───────────────────────────────────

with open('best_params.json') as f:
    BEST = json.load(f)
SYMBOLS = list(BEST.keys())

# load scaler + classifier
scaler, classifier = joblib.load('scaler.pkl'), joblib.load('classifier.pkl')

# ─────────────── EXCHANGE SETUP ─────────────────────────────────────────

exchange = ccxt.kraken({
    'enableRateLimit': True,
    'apiKey': API_KEY,
    'secret': API_SECRET,
})
exchange.load_markets()

# ─────────────── CACHING SYSTEM ─────────────────────────────────────────

_hourly_cache = {}  # symbol -> (timestamp, DataFrame)

def save_cache():
    """Persist hourly cache to disk."""
    joblib.dump(_hourly_cache, CACHE_FILE)
    if VERBOSE:
        print(f"♻️  Saved hourly cache ({len(_hourly_cache)} symbols)")

def load_cache():
    """Load hourly cache from disk."""
    global _hourly_cache
    try:
        _hourly_cache = joblib.load(CACHE_FILE)
        if VERBOSE:
            print(f"♻️  Loaded hourly cache ({len(_hourly_cache)} symbols)")
    except FileNotFoundError:
        if VERBOSE:
            print("♻️  No cache file, starting fresh")

def validate_cache():
    """Drop expired entries."""
    now = datetime.utcnow()
    expired = []
    for sym, (ts, df) in _hourly_cache.items():
        if now - ts > CACHE_EXPIRY:
            expired.append(sym)
    for sym in expired:
        del _hourly_cache[sym]
    if VERBOSE and expired:
        print(f"♻️  Cleared {len(expired)} expired cache entries")

# ─────────────── RATE LIMIT HELPERS ─────────────────────────────────────

_last_req = 0.0
def rate_limited_request():
    """Enforce ~1 req/sec for public data."""
    global _last_req
    now = time.time()
    delta = now - _last_req
    if delta < 1.0:
        time.sleep(1.0 - delta)
    _last_req = time.time()

# ─────────────── DATA FETCH / INDICATORS ────────────────────────────────

def fetch_hourly_history(symbol):
    """
    Return a DataFrame of 1 h bars with all indicators
    (cached for CACHE_EXPIRY).
    """
    # return cache if fresh
    now = datetime.utcnow()
    if symbol in _hourly_cache:
        ts, df = _hourly_cache[symbol]
        if now - ts < CACHE_EXPIRY and len(df) >= MIN_DATA_BARS:
            return df

    # fetch raw
    since = exchange.milliseconds() - int(HISTORY_HOURS * 3600 * 1000)
    all_bars = []
    while True:
        rate_limited_request()
        chunk = exchange.fetch_ohlcv(symbol, ENTRY_TIMEFRAME, since, limit=500)
        if not chunk:
            break
        all_bars += chunk
        since = chunk[-1][0] + 1
        if len(all_bars) >= HISTORY_HOURS + 2:
            break

    if not all_bars:
        return None

    df = pd.DataFrame(all_bars, columns=['ts','open','high','low','close','vol'])
    df['dt'] = pd.to_datetime(df['ts'], unit='ms')
    df.set_index('dt', inplace=True)
    df.sort_index(inplace=True)

    # compute indicators
    df['prev_close'] = df['close'].shift(1)
    df['tr'] = np.maximum.reduce([
        df['high'] - df['low'],
        (df['high'] - df['prev_close']).abs(),
        (df['low']  - df['prev_close']).abs(),
    ])
    df['atr20']    = df['tr'].rolling(20).mean()
    df['rv20']     = np.log(df['close']).diff().rolling(20).std()
    df['ma10']     = df['close'].rolling(10).mean()
    df['ma50']     = df['close'].rolling(50).mean()
    df['ma10_50']  = df['ma10'] - df['ma50']
    delta = df['close'].diff()
    up, down = delta.clip(lower=0), -delta.clip(upper=0)
    df['rsi14']    = 100 - 100/(1 + up.rolling(14).mean()/down.rolling(14).mean())
    df['vol_spike']= df['vol']/df['vol'].rolling(20).mean()

    df = df.dropna()
    _hourly_cache[symbol] = (now, df)
    return df

def fetch_latest_5m(symbol):
    """
    Fetch just the most recent 5 m bar.
    Returns (timestamp: datetime, close: float), or (None,None).
    """
    rate_limited_request()
    bars = exchange.fetch_ohlcv(symbol, EXIT_TIMEFRAME, limit=2)
    if not bars:
        return None, None
    df = pd.DataFrame(bars, columns=['ts','o','h','l','c','v'])
    df['dt'] = pd.to_datetime(df['ts'], unit='ms')
    df.set_index('dt', inplace=True)
    t0 = df.index[-1]
    return t0.to_pydatetime(), float(df['c'].iloc[-1])

# ─────────────── TRADE LOGGING ──────────────────────────────────────────

if not os.path.exists(LOG_FILE):
    with open(LOG_FILE,'w',newline='') as f:
        csv.writer(f).writerow([
            'symbol','action','timestamp',
            'price','qty','roc','prob',
            'entry_time','entry_price',
            'exit_time','exit_price','exit_type','pnl'
        ])

def log_trade(row):
    with open(LOG_FILE,'a',newline='') as f:
        csv.writer(f).writerow([
            row.get('symbol'),
            row.get('action'),
            row.get('timestamp'),
            row.get('price'),
            row.get('qty'),
            row.get('roc',''),
            row.get('prob',''),
            row.get('entry_time',''),
            row.get('entry_price',''),
            row.get('exit_time',''),
            row.get('exit_price',''),
            row.get('exit_type',''),
            row.get('pnl',''),
        ])

# ─────────────── POSITION PROCESSING ────────────────────────────────────

positions = {}  # symbol → {entry_time, entry_price, qty}

def process_symbol(symbol):
    """
    Check both exit (5m) and entry (1h + classifier) for `symbol`.
    Returns one of {'EXIT','ENTER',None}.
    """
    global positions

    params = BEST[symbol]
    # 1) EXIT check on 5 m
    if symbol in positions:
        exit_time, exit_price = fetch_latest_5m(symbol)
        if exit_time is None:
            return None

        entry = positions[symbol]
        tp_lvl = entry['entry_price']*(1+params['tp_pct'])
        sl_lvl = entry['entry_price']*(1-params['sl_pct'])
        if exit_price >= tp_lvl:
            etype='TP'
        elif exit_price <= sl_lvl:
            etype='SL'
        else:
            etype=None

        if etype:
            # adjust for fees & slippage
            adj_exit = exit_price*(1-FEE_RATE-SLIPPAGE_PCT/2)
            pnl = (adj_exit - entry['entry_price'])*entry['qty']
            ts = exit_time.isoformat()
            print(f"{ts} ← EXIT {symbol} @ {adj_exit:.4f} ({etype}) PnL={pnl:.2f}")
            if not DRY_RUN:
                exchange.create_order(symbol,'market','sell', entry['qty'])
            log_trade({
                'symbol':symbol,'action':'EXIT','timestamp':ts,
                'price':f"{adj_exit:.4f}",'qty':f"{entry['qty']:.6f}",
                'roc':'','prob':'',
                'entry_time':entry['entry_time'].isoformat(),
                'entry_price':f"{entry['entry_price']:.4f}",
                'exit_time':ts,'exit_price':f"{adj_exit:.4f}",
                'exit_type':etype,'pnl':f"{pnl:.2f}",
            })
            del positions[symbol]
            return 'EXIT'

    # 2) ENTRY check on 1 h + classifier
    df = fetch_hourly_history(symbol)
    if df is None or len(df) < max(params['roc_period'],50,20,14):
        return None

    # compute ROC on last two bars
    last = df['close'].iat[-1]
    prev = df['close'].iat[-1-params['roc_period']]
    roc  = last/prev - 1
    if roc <= params['threshold']:
        return None

    # build feature vector
    feat = pd.DataFrame([{
        'roc':       roc,
        'atr20':     df['atr20'].iat[-1],
        'rv20':      df['rv20'].iat[-1],
        'ma10_50':   df['ma10_50'].iat[-1],
        'rsi14':     df['rsi14'].iat[-1],
        'vol_spike': df['vol_spike'].iat[-1],
        'hold_hrs':  params['hold_bars'],
        'hour':      df.index[-1].hour,
        'weekend':   int(df.index[-1].weekday()>=5),
    }])
    prob = classifier.predict_proba(scaler.transform(feat))[0,1]
    if prob < CLASSIFIER_THRESHOLD:
        return None

    # execute entry
    entry_price = last*(1+FEE_RATE+SLIPPAGE_PCT/2)
    qty = NOTIONAL/entry_price
    ts = datetime.utcnow().isoformat()
    print(f"{ts} → ENTER {symbol} @ {entry_price:.4f} ROC={roc:.2%} P={prob:.2f}")
    if not DRY_RUN:
        exchange.create_order(symbol,'market','buy',qty)

    positions[symbol] = {
        'entry_time':   datetime.utcnow(),
        'entry_price':  entry_price,
        'qty':          qty
    }
    log_trade({
        'symbol':symbol,'action':'ENTER','timestamp':ts,
        'price':f"{entry_price:.4f}",'qty':f"{qty:.6f}",
        'roc':f"{roc:.4f}",'prob':f"{prob:.2f}",
        'entry_time':ts,'entry_price':f"{entry_price:.4f}",
        'exit_time':'','exit_price':'','exit_type':'','pnl':''
    })
    return 'ENTER'

# ─────────────── MAIN LOOP ─────────────────────────────────────────────

def main():
    load_cache()
    print(f"▶️  Starting 2-resolution live trader — Dry run={DRY_RUN}")
    try:
        while True:
            validate_cache()
            print(f"\n[{datetime.utcnow().isoformat()}] open positions: {len(positions)}")
            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as execr:
                futures = { execr.submit(process_symbol,s): s for s in SYMBOLS }
                for f in as_completed(futures):
                    f.result()
            # persist cache occasionally
            save_cache()
            # wait until next 5 m candle
            now = datetime.utcnow()
            secs = 300 - (now.minute%5*60 + now.second)
            time.sleep(max(secs,1))
    except KeyboardInterrupt:
        print("\n📴  Stopped cleanly, saving cache…")
        save_cache()

if __name__=="__main__":
    main()

