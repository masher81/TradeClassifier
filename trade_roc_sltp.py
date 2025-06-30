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
Enhancements:
  – detailed open-positions table
  – non-blocking manual-exit prompt (3 s timeout)
"""

import os, sys, time, csv, json, select
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

def show_open_positions():
    """
    Print a detailed table of all current positions:
      • symbol
      • entry time
      • raw entry price
      • cost basis (incl. fees/slippage)
      • quantity
      • latest market price
      • current PnL
      • target TP and SL levels
    """
    if not positions:
        print("No open positions.")
        return

    print("\n� Detailed Open Positions:")
    header = (
        f"{'SYMBOL':<10} {'ENTRY':<19} {'RAW':>7} {'BASIS':>7} "
        f"{'QTY':>10} {'PRICE':>8} {'PNL':>10} {'TP LVL':>8} {'SL LVL':>8}"
    )
    print(header)
    print("-" * len(header))

    for sym, pos in positions.items():
        raw        = pos.get('raw_entry_price', pos.get('entry_price', float('nan')))
        cost_basis = pos.get('cost_basis', raw * (1 + FEE_RATE + SLIPPAGE_PCT/2))
        qty        = pos.get('qty', float('nan'))
        t5, price5 = fetch_latest_5m(sym)
        price5     = price5 if price5 is not None else float('nan')
        tp_lvl     = raw * (1 + BEST[sym]['tp_pct'])
        sl_lvl     = raw * (1 - BEST[sym]['sl_pct'])
        pnl        = (price5 - cost_basis) * qty

        print(
            f"{sym:<10} "
            f"{pos['entry_time'].isoformat():<19} "
            f"{raw:7.4f} "
            f"{cost_basis:7.4f} "
            f"{qty:10.4f} "
            f"{price5:8.4f} "
            f"{pnl:10.2f} "
            f"{tp_lvl:8.4f} "
            f"{sl_lvl:8.4f}"
        )
    print()

def manual_exit(symbol: str):
    """
    Do a market sell for the given symbol and clean up.
    """
    if symbol not in positions:
        print(f"❌ No open position for {symbol}")
        return

    entry      = positions[symbol]
    raw        = entry.get('raw_entry_price', entry.get('entry_price', None))
    cost_basis = entry.get('cost_basis', raw * (1 + FEE_RATE + SLIPPAGE_PCT/2))
    qty        = entry.get('qty', 0.0)

    _, price5 = fetch_latest_5m(symbol)
    if price5 is None:
        print(f"❌ Could not fetch latest price for {symbol}, aborting manual exit.")
        return

    exit_price = price5 * (1 - FEE_RATE - SLIPPAGE_PCT/2)
    pnl        = (exit_price - cost_basis) * qty
    ts         = datetime.utcnow().isoformat()

    print(f"{ts} ← MANUAL EXIT {symbol} @ {exit_price:.4f}  pnl={pnl:.2f}")
    if not DRY_RUN:
        exchange.create_order(symbol, 'market', 'sell', qty)

    log_trade({
        'symbol':      symbol,
        'action':      'EXIT',
        'timestamp':   ts,
        'price':       f"{exit_price:.4f}",
        'qty':         f"{qty:.6f}",
        'roc':         '',
        'prob':        '',
        'entry_time':  entry['entry_time'].isoformat(),
        'entry_price': f"{raw:.4f}",
        'exit_time':   ts,
        'exit_price':  f"{exit_price:.4f}",
        'exit_type':   'MANUAL',
        'pnl':         f"{pnl:.2f}",
    })
    del positions[symbol]

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
            print("♻️ No cache file found – starting fresh")

def save_persistent_cache():
    """Save RAW_CACHE+PROC_CACHE to disk."""
    joblib.dump({'raw': RAW_CACHE, 'proc': PROC_CACHE}, CACHE_FILE)
    if VERBOSE:
        print(f"♻️ Saved cache: {len(RAW_CACHE)} raw, {len(PROC_CACHE)} processed")

def validate_cache():
    """Drop any cache entries older than CACHE_EXPIRY."""
    now = datetime.utcnow()
    def prune(d):
        for k,(ts,_) in list(d.items()):
            if now - ts > CACHE_EXPIRY:
                del d[k]
    prune(RAW_CACHE)
    prune(PROC_CACHE)

# ─────────────── RATE LIMIT HELPERS ─────────────────────────────────────

_last_req = 0.0
def rate_limited_request():
    """~1 req/sec for public data."""
    global _last_req
    now = time.time()
    if now - _last_req < 1.0:
        time.sleep(1.0 - (now - _last_req))
    _last_req = time.time()

# ─────────────── DATA FETCH & INDICATORS ────────────────────────────────

def fetch_raw_data(symbol):
    """Fetch raw OHLCV for ENTRY_TIMEFRAME, cached in RAW_CACHE."""
    now = datetime.utcnow()
    if symbol in RAW_CACHE:
        ts, data = RAW_CACHE[symbol]
        if now - ts < CACHE_EXPIRY and len(data) >= MIN_DATA_BARS:
            return data

    since    = exchange.milliseconds() - int(HISTORY_HOURS * 3600 * 1000)
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
    """Compute your standard indicators on a list of OHLCV bars."""
    df = pd.DataFrame(raw_bars, columns=['ts','open','high','low','close','vol'])
    df['dt']       = pd.to_datetime(df['ts'], unit='ms')
    df.set_index('dt', inplace=True)
    df.sort_index(inplace=True)

    df['prev_close'] = df['close'].shift(1)
    df['tr']         = np.maximum.reduce([
        df['high'] - df['low'],
        (df['high'] - df['prev_close']).abs(),
        (df['low']  - df['prev_close']).abs(),
    ])
    df['atr20']     = df['tr'].rolling(20).mean()
    df['rv20']      = np.log(df['close']).diff().rolling(20).std()
    df['ma10']      = df['close'].rolling(10).mean()
    df['ma50']      = df['close'].rolling(50).mean()
    df['ma10_50']   = df['ma10'] - df['ma50']
    delta           = df['close'].diff()
    up,down         = delta.clip(lower=0), -delta.clip(upper=0)
    df['rsi14']     = 100 - 100/(1 + up.rolling(14).mean()/down.rolling(14).mean())
    df['vol_spike'] = df['vol']/df['vol'].rolling(20).mean()

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
    PROC_CACHE[symbol] = (now, df)
    return df

def fetch_latest_5m(symbol):
    """
    Fetch the latest 5m bar for `symbol`.
    Returns (dt:datetime, close:float) or (None,None)
    """
    rate_limited_request()
    bars = exchange.fetch_ohlcv(symbol, EXIT_TIMEFRAME, limit=2)
    if not bars:
        return None, None
    df = pd.DataFrame(bars, columns=['ts','o','h','l','c','v'])
    df['dt'] = pd.to_datetime(df['ts'], unit='ms')
    df.set_index('dt', inplace=True)
    return df.index[-1].to_pydatetime(), float(df['c'].iloc[-1])

# ─────────────── TRADE LOGGING ──────────────────────────────────────────

if not os.path.exists(LOG_FILE):
    with open(LOG_FILE,'w', newline='') as f:
        csv.writer(f).writerow([
            'symbol','action','timestamp','price','qty','roc','prob',
            'entry_time','entry_price','exit_time','exit_price','exit_type','pnl'
        ])

def log_trade(row):
    with open(LOG_FILE,'a', newline='') as f:
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
            'entry_price':  p.get('entry_price', p.get('raw_entry_price',0)),
            'qty':          p['qty']
        }
    with open(POSITIONS_FILE,'w') as f:
        json.dump(serial, f, indent=2)

# ─────────────── MARKET SUMMARY ─────────────────────────────────────────

def print_market_summary():
    snapshot = {}
    for sym in SYMBOLS[:5]:
        df = load_history(sym)
        if df is not None and len(df) >= 2:
            last, prev = df['close'].iloc[-1], df['close'].iloc[-2]
            pct = (last/prev - 1) * 100
            vol = df['vol'].iloc[-1]
            snapshot[sym] = (last, pct, vol)
    print("\n� Market Summary (1h bars):")
    for s,(p,pc,v) in snapshot.items():
        print(f"  {s:8} {p:.4f} ({pc:+.2f}%)  Vol={v:.0f}")

# ─────────────── TRADING LOGIC ─────────────────────────────────────────

positions = {}  # symbol → {entry_time,raw_entry_price,cost_basis,entry_price,qty}

def process_symbol(symbol):
    """
    1) If in a position, check 5m SL/TP exit (on raw entry price)
    2) Else, check 1h ROC + classifier entry, then require
       the last 20 x 5m bars to be strictly increasing
    """
    global positions
    params = BEST[symbol]

    # ─── EXIT on 5m ────────────────────────────────────────────────
    if symbol in positions:
        t5, close5 = fetch_latest_5m(symbol)
        if t5 is not None and close5 is not None:
            entry      = positions[symbol]
            raw_price  = entry.get('raw_entry_price', entry.get('entry_price'))
            cost_basis = entry.get('cost_basis', raw_price * (1 + FEE_RATE + SLIPPAGE_PCT/2))
            qty        = entry.get('qty', 0)

            tp_lvl = raw_price * (1 + params['tp_pct'])
            sl_lvl = raw_price * (1 - params['sl_pct'])
            etype  = None

            if close5 >= tp_lvl:
                etype = 'TP'
            elif close5 <= sl_lvl:
                etype = 'SL'

            if etype:
                exit_cost = close5 * (1 - FEE_RATE - SLIPPAGE_PCT/2)
                pnl       = (exit_cost - cost_basis) * qty
                ts        = t5.isoformat()
                print(f"{ts} ← EXIT {symbol} @ {exit_cost:.4f} ({etype})  pnl={pnl:.2f}")

                if not DRY_RUN:
                    exchange.create_order(symbol, 'market', 'sell', qty)

                log_trade({
                    'symbol':      symbol,
                    'action':      'EXIT',
                    'timestamp':   ts,
                    'price':       f"{exit_cost:.4f}",
                    'qty':         f"{qty:.6f}",
                    'roc':         '',
                    'prob':        '',
                    'entry_time':  entry['entry_time'].isoformat(),
                    'entry_price': f"{raw_price:.4f}",
                    'exit_time':   ts,
                    'exit_price':  f"{exit_cost:.4f}",
                    'exit_type':   etype,
                    'pnl':         f"{pnl:.2f}",
                })

                del positions[symbol]
                return 'EXIT'

        # still in a position, skip entry logic
        return None

    # ─── ENTRY on 1h + classifier ───────────────────────────────────
    df = load_history(symbol)
    if df is None or len(df) < max(params['roc_period'], 50, 20, 14):
        return None

    # 1h ROC filter
    last = df['close'].iat[-1]
    prev = df['close'].iat[-1 - params['roc_period']]
    roc  = last / prev - 1
    if roc <= params['threshold']:
        return None

    # classifier check
    now = datetime.utcnow()
    feat = pd.DataFrame([{
        'roc':        roc,
        'atr20':      df['atr20'].iat[-1],
        'rv20':       df['rv20'].iat[-1],
        'ma10_50':    df['ma10_50'].iat[-1],
        'rsi14':      df['rsi14'].iat[-1],
        'vol_spike':  df['vol_spike'].iat[-1],
        'hold_hours': params.get('hold_bars', 1),
        'hour':       now.hour,
        'weekend':    int(now.weekday() >= 5),
        'sl_pct':     params['sl_pct'],
        'tp_pct':     params['tp_pct'],
    }])
    prob = classifier.predict_proba(scaler.transform(feat))[0, 1]
    if prob < CLASSIFIER_THRESHOLD:
        return None

    # ─── 20×5m monotonicity check ────────────────────────────────────
    rate_limited_request()
    bars5 = exchange.fetch_ohlcv(symbol, EXIT_TIMEFRAME, limit=20)
    if not bars5 or len(bars5) < 20:
        return None
    closes5 = [b[4] for b in bars5]
    if any(closes5[i] < closes5[i - 1] for i in range(1, len(closes5))):
        if VERBOSE:
            print(f"⛔ {symbol}: skipped entry, 5m prices not strictly rising")
        return None

    # ─── PLACE ENTRY ────────────────────────────────────────────────
    raw_entry_price = last
    cost_basis      = last * (1 + FEE_RATE + SLIPPAGE_PCT/2)
    qty             = NOTIONAL / cost_basis
    ts              = now.isoformat()
    print(f"{ts} → ENTER {symbol} @ {cost_basis:.4f}  roc={roc:.2%}  P={prob:.2f}")

    if not DRY_RUN:
        exchange.create_order(symbol, 'market', 'buy', qty)

    positions[symbol] = {
        'entry_time':       now,
        'raw_entry_price':  raw_entry_price,
        'cost_basis':       cost_basis,
        'entry_price':      raw_entry_price,
        'qty':              qty
    }

    log_trade({
        'symbol':      symbol,
        'action':      'ENTER',
        'timestamp':   ts,
        'price':       f"{cost_basis:.4f}",
        'qty':         f"{qty:.6f}",
        'roc':         f"{roc:.4f}",
        'prob':        f"{prob:.2f}",
        'entry_time':  ts,
        'entry_price': f"{raw_entry_price:.4f}",
        'exit_time':   '',
        'exit_price':  '',
        'exit_type':   '',
        'pnl':         '',
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
            show_open_positions()

            # non-blocking manual-exit prompt (3s)
            sys.stdout.write("Type a symbol to EXIT manually (3s)… ")
            sys.stdout.flush()
            r,_,_ = select.select([sys.stdin], [], [], 3)
            if r:
                choice = sys.stdin.readline().strip().upper()
                if choice:
                    manual_exit(choice)
                    save_persistent_cache()
                    save_positions(positions)
                    continue
            else:
                print()  # newline if timed out

            # automated trade cycle
            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as exe:
                futures = {exe.submit(process_symbol, s): s for s in SYMBOLS}
                for f in as_completed(futures):
                    f.result()

            save_persistent_cache()
            save_positions(positions)

            # sleep til next 5m boundary
            now = datetime.utcnow()
            secs = 300 - (now.minute % 5 * 60 + now.second)
            time.sleep(max(secs, 1))

    except KeyboardInterrupt:
        print("\n📴  Stopping cleanly…")
        save_persistent_cache()
        save_positions(positions)

if __name__=="__main__":
    main()








