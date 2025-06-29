#!/usr/bin/env python3
"""
trade_roc_sltp_full.py

2-resolution live trader:
  • 1 h ROC + classifier filter for entries
  • 5 m bars for SL/TP exits
Preserves:
  – caching (raw+processed hourly)
  – verbose/logging/market snapshot
  – ThreadPoolExecutor
  – persistent open positions across restarts
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
from sklearn.ensemble    import RandomForestClassifier

# ─────────────── CONFIG ─────────────────────────────────────────────────

load_dotenv()

API_KEY              = os.getenv('KRAKEN_APIKEY')
API_SECRET           = os.getenv('KRAKEN_SECRET')
NOTIONAL             = float(os.getenv('NOTIONAL', 10_000))
FEE_RATE             = float(os.getenv('FEE_RATE', 0.001))
SLIPPAGE_PCT         = float(os.getenv('SLIPPAGE_PCT', 0.0005))
DRY_RUN              = os.getenv('DRY_RUN','false').lower() in ('1','true')
CLASSIFIER_THRESHOLD = float(os.getenv('CLASSIFIER_THRESHOLD', 0.40))
LOG_FILE             = os.getenv('TRADE_LOG_FILE','trade_log.csv')
CACHE_FILE           = 'data_cache.pkl'
POSITIONS_FILE       = 'positions.json'

ENTRY_TIMEFRAME      = '1h'
EXIT_TIMEFRAME       = '5m'
HISTORY_HOURS        = 168       # how many hours of 1h data to cache
MIN_DATA_BARS        = 50
MIN_AVG_VOLUME       = 100
CACHE_EXPIRY         = timedelta(minutes=30)
MAX_WORKERS          = 5
VERBOSE              = True

# ─────────────── LOAD PARAMS & MODELS ───────────────────────────────────

with open('best_params.json') as f:
    BEST = json.load(f)
SYMBOLS = list(BEST.keys())

# load scaler + classifier
scaler, classifier = joblib.load('scaler.pkl'), joblib.load('classifier.pkl')

# ─────────────── SETUP EXCHANGE ─────────────────────────────────────────

exchange = ccxt.kraken({
    'enableRateLimit': True,
    'apiKey':          API_KEY,
    'secret':          API_SECRET,
})
exchange.load_markets()

# ─────────────── CACHING SYSTEM ─────────────────────────────────────────

RAW_CACHE = {}         # symbol -> (timestamp:datetime, raw_bars:list)
PROC_CACHE = {}        # symbol -> (timestamp:datetime, df:DataFrame)

def load_persistent_cache():
    """Load RAW_CACHE+PROC_CACHE from disk."""
    global RAW_CACHE, PROC_CACHE
    try:
        data = joblib.load(CACHE_FILE)
        RAW_CACHE.update(data.get('raw', {}))
        PROC_CACHE.update(data.get('proc', {}))
        if VERBOSE:
            print(f"♻️ Loaded cache: {len(RAW_CACHE)} raw, {len(PROC_CACHE)} processed")
    except FileNotFoundError:
        if VERBOSE:
            print("♻️ No cache file, starting fresh")

def save_persistent_cache():
    """Save RAW_CACHE+PROC_CACHE to disk."""
    joblib.dump({'raw': RAW_CACHE, 'proc': PROC_CACHE}, CACHE_FILE)
    if VERBOSE:
        print(f"♻️ Saved cache: {len(RAW_CACHE)} raw, {len(PROC_CACHE)} processed")

def validate_cache():
    """Drop any cache entries older than CACHE_EXPIRY."""
    now = datetime.utcnow()
    def prune(d):
        removed = []
        for k,(ts,_) in list(d.items()):
            if now - ts > CACHE_EXPIRY:
                removed.append(k)
                del d[k]
        return removed

    r1 = prune(RAW_CACHE)
    r2 = prune(PROC_CACHE)
    if VERBOSE and (r1 or r2):
        print(f"♻️ Cleared expired cache: raw {len(r1)}, proc {len(r2)} symbols")

# ─────────────── RATE LIMIT HELPERS ─────────────────────────────────────

_last_req = 0.0
def rate_limited_request():
    """~1 req/sec for public data."""
    global _last_req
    now = time.time()
    dt  = now - _last_req
    if dt < 1.0:
        time.sleep(1.0 - dt)
    _last_req = time.time()

# ─────────────── DATA FETCH & INDICATORS ────────────────────────────────

def fetch_raw_data(symbol):
    """Fetch raw OHLCV for ENTRY_TIMEFRAME, cached in RAW_CACHE."""
    now = datetime.utcnow()
    if symbol in RAW_CACHE:
        ts, data = RAW_CACHE[symbol]
        if now - ts < CACHE_EXPIRY and len(data) >= MIN_DATA_BARS:
            return data

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

    if all_bars:
        RAW_CACHE[symbol] = (now, all_bars)
    return all_bars if len(all_bars) >= MIN_DATA_BARS else None

def compute_indicators(raw_bars):
    """Compute all your standard indicators on a list of OHLCV bars."""
    df = pd.DataFrame(raw_bars, columns=['ts','open','high','low','close','vol'])
    df['dt'] = pd.to_datetime(df['ts'], unit='ms')
    df.set_index('dt', inplace=True)
    df.sort_index(inplace=True)

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
    up,down      = delta.clip(lower=0), -delta.clip(upper=0)
    df['rsi14']   = 100 - 100/(1 + up.rolling(14).mean()/down.rolling(14).mean())
    df['vol_spike']= df['vol']/df['vol'].rolling(20).mean()

    return df.dropna()

def load_history(symbol):
    """
    Return a DataFrame of processed 1 h bars for `symbol`.
    Uses RAW_CACHE + PROC_CACHE.
    """
    now = datetime.utcnow()
    if symbol in PROC_CACHE:
        ts, df = PROC_CACHE[symbol]
        if now - ts < CACHE_EXPIRY and len(df) >= MIN_DATA_BARS:
            return df

    raw = fetch_raw_data(symbol)
    if not raw:
        return None

    df = compute_indicators(raw)
    if not df.empty:
        PROC_CACHE[symbol] = (now, df)
        return df
    return None

def fetch_latest_5m(symbol):
    """
    Fetch the latest 5 m bar for `symbol`.
    Returns (dt:datetime, close:float) or (None,None)
    """
    rate_limited_request()
    bars = exchange.fetch_ohlcv(symbol, EXIT_TIMEFRAME, limit=2)
    if not bars:
        return None,None
    df = pd.DataFrame(bars,columns=['ts','o','h','l','c','v'])
    df['dt'] = pd.to_datetime(df['ts'],unit='ms')
    df.set_index('dt', inplace=True)
    return df.index[-1].to_pydatetime(), float(df['c'].iloc[-1])

# ─────────────── TRADE LOGGING ──────────────────────────────────────────

if not os.path.exists(LOG_FILE):
    with open(LOG_FILE,'w',newline='') as f:
        csv.writer(f).writerow([
            'symbol','action','timestamp','price','qty','roc','prob',
            'entry_time','entry_price','exit_time','exit_price','exit_type','pnl'
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

# ─────────────── PERSISTENT POSITIONS ───────────────────────────────────

def load_positions():
    """Load open positions from disk."""
    try:
        with open(POSITIONS_FILE) as f:
            raw = json.load(f)
        # parse entry_time
        for sym,pos in raw.items():
            pos['entry_time'] = datetime.fromisoformat(pos['entry_time'])
        return raw
    except:
        return {}

def save_positions(pos):
    """Save open positions to disk."""
    serial = {}
    for sym,p in pos.items():
        serial[sym] = {
            'entry_time':   p['entry_time'].isoformat(),
            'entry_price': p['entry_price'],
            'qty':         p['qty']
        }
    with open(POSITIONS_FILE,'w') as f:
        json.dump(serial,f,indent=2)

# ─────────────── MARKET SUMMARY ─────────────────────────────────────────

def print_market_summary():
    """Show a quick glance at price + volume for first few symbols."""
    snapshot = {}
    for sym in SYMBOLS[:5]:
        df = load_history(sym)
        if df is not None and len(df) >= 2:
            last,prev = df['close'].iloc[-1], df['close'].iloc[-2]
            pct = (last/prev -1)*100
            vol = df['vol'].iloc[-1]
            snapshot[sym] = (last,pct,vol)
    print("\n📊 Market Summary (1 h bars):")
    for s,(p,pc,v) in snapshot.items():
        print(f"  {s:8} {p:.4f} ({pc:+.2f}%)  Vol={v:.0f}")

# ─────────────── PROCESS ONE SYMBOL ────────────────────────────────────

positions = {}  # symbol → {entry_time,entry_price,qty}

def process_symbol(symbol):
    """
    1) If in a position, check 5 m SL/TP exit
    2) Else, check 1 h ROC + classifier entry
    """
    global positions

    params = BEST[symbol]

    # ─ Exit on 5 m
    if symbol in positions:
        t5, close5 = fetch_latest_5m(symbol)
        if t5 and close5 is not None:
            entry = positions[symbol]
            tp_lvl = entry['entry_price']*(1+params['tp_pct'])
            sl_lvl = entry['entry_price']*(1-params['sl_pct'])
            etype = None
            if close5 >= tp_lvl:
                etype = 'TP'
            elif close5 <= sl_lvl:
                etype = 'SL'
            if etype:
                adj_exit = close5*(1-FEE_RATE-SLIPPAGE_PCT/2)
                pnl      = (adj_exit - entry['entry_price'])*entry['qty']
                ts       = t5.isoformat()
                print(f"{ts} ← EXIT {symbol} @ {adj_exit:.4f} ({etype})  pnl={pnl:.2f}")
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

    # ─ Entry on 1 h + classifier
    df = load_history(symbol)
    if df is None or len(df) < max(params['roc_period'],50,20,14):
        return None

    last = df['close'].iat[-1]
    prev = df['close'].iat[-1-params['roc_period']]
    roc  = last/prev - 1
    if roc <= params['threshold']:
        return None

    feat = pd.DataFrame([{
        'roc':       roc,
        'atr20':     df['atr20'].iat[-1],
        'rv20':      df['rv20'].iat[-1],
        'ma10_50':   df['ma10_50'].iat[-1],
        'rsi14':     df['rsi14'].iat[-1],
        'vol_spike': df['vol_spike'].iat[-1],
        'hold_hrs':  params.get('hold_bars',1),
        'hour':      df.index[-1].hour,
        'weekend':   int(df.index[-1].weekday()>=5),
    }])
    prob = classifier.predict_proba(scaler.transform(feat))[0,1]
    if prob < CLASSIFIER_THRESHOLD:
        return None

    # place entry
    entry_price = last*(1+FEE_RATE+SLIPPAGE_PCT/2)
    qty         = NOTIONAL/entry_price
    ts          = datetime.utcnow().isoformat()
    print(f"{ts} → ENTER {symbol} @ {entry_price:.4f}  roc={roc:.2%}  P={prob:.2f}")
    if not DRY_RUN:
        exchange.create_order(symbol,'market','buy',qty)

    positions[symbol] = {
        'entry_time':  datetime.fromisoformat(ts),
        'entry_price': entry_price,
        'qty':         qty
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
    global positions

    # load caches + positions
    load_persistent_cache()
    positions = load_positions()

    print(f"▶️  Starting 2-res live trader — Dry run={DRY_RUN}, TH={CLASSIFIER_THRESHOLD}")
    print_market_summary()

    try:
        while True:
            validate_cache()
            print(f"\n[{datetime.utcnow().isoformat()}] Open positions: {len(positions)}")

            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as exe:
                futures = {exe.submit(process_symbol, s): s for s in SYMBOLS}
                for f in as_completed(futures):
                    # exceptions inside process_symbol will be printed there
                    f.result()

            # persist cache + positions
            save_persistent_cache()
            save_positions(positions)

            # sleep til next 5 m boundary
            now = datetime.utcnow()
            secs = 300 - (now.minute%5*60 + now.second)
            time.sleep(max(secs,1))

    except KeyboardInterrupt:
        print("\n📴  Stopping cleanly…")
        save_persistent_cache()
        save_positions(positions)

if __name__=="__main__":
    main()
