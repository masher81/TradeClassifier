#!/usr/bin/env python3
import os
import sys
import time
import csv
import json
from datetime import datetime, timedelta

import ccxt
import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm

# ─────────────── LOAD ENV ─────────────────────────────────────────
load_dotenv()
API_KEY      = os.getenv('KRAKEN_APIKEY')
API_SECRET   = os.getenv('KRAKEN_SECRET')
NOTIONAL     = float(os.getenv('NOTIONAL', 10_000))     # USD per trade
FEE_RATE     = float(os.getenv('FEE_RATE', 0.001))      # 0.1% per side
SLIPPAGE_PCT = float(os.getenv('SLIPPAGE_PCT', 0.0005)) # 0.05% round-trip
DRY_RUN      = os.getenv('DRY_RUN', 'false').lower() in ('1', 'true')
THRESHOLD    = float(os.getenv('CLASSIFIER_THRESHOLD', 0.40))
LOG_FILE     = os.getenv('TRADE_LOG_FILE', 'trade_log.csv')
POS_FILE     = os.getenv('POSITIONS_FILE', 'open_positions.json')

# ensure log file exists with header
if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'symbol','action','timestamp','price','qty','roc',
            'entry_time','entry_price','exit_time','exit_price','pnl'
        ])

# ─────────────── LOAD BEST PARAMS ─────────────────────────────────
with open('best_params.json') as f:
    BEST = json.load(f)

print("▶️  Loaded BEST_PARAMS keys example:")
for sym, p in list(BEST.items())[:3]:
    print(f"   {sym}: {p}")
print("────────────────────────────────────────────────────────────────")

TIMEFRAME = '1h'
exchange  = ccxt.kraken({
    'apiKey':         API_KEY,
    'secret':         API_SECRET,
    'enableRateLimit': True,
})
exchange.load_markets()

# ─────────────── PERSISTENT POSITIONS ──────────────────────────────

def load_positions():
    """Load open positions from disk (if any)."""
    if not os.path.exists(POS_FILE):
        return {}
    try:
        raw = json.load(open(POS_FILE))
        pos = {}
        for sym, p in raw.items():
            # parse exit_time back to datetime
            et = datetime.fromisoformat(p['exit_time'])
            pos[sym] = {
                'entry_time':  p['entry_time'],
                'entry_price': float(p['entry_price']),
                'qty':         float(p['qty']),
                'roc':         float(p['roc']),
                'exit_time':   et,
            }
        return pos
    except Exception as e:
        print(f"⚠️  Could not load positions file: {e}")
        return {}

def save_positions():
    """Write current open positions to disk."""
    out = {}
    for sym, p in positions.items():
        out[sym] = {
            'entry_time':  p['entry_time'],
            'entry_price': p['entry_price'],
            'qty':         p['qty'],
            'roc':         p['roc'],
            'exit_time':   p['exit_time'].isoformat(),
        }
    with open(POS_FILE, 'w') as f:
        json.dump(out, f, indent=2)

# load at startup
positions = load_positions()

# ─────────────── UTILITIES ────────────────────────────────────────

def fetch_bars(symbol, hours):
    """Fetch at least `hours+2` bars so we can compute ROC."""
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

def log_trade(row):
    """Append a row dict to the CSV."""
    with open(LOG_FILE, 'a', newline='') as f:
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

# ─────────────── MAIN LOOP ─────────────────────────────────────────

def main():
    print(f"▶️  Starting ROC‐based trader...  Dry run={DRY_RUN}, threshold={THRESHOLD}")
    try:
        while True:
            now = datetime.utcnow()

            # heartbeat + time-to-exit display
            sys.stdout.write(f"\n[{now.isoformat()}] scanning {len(BEST)} symbols; open positions: {len(positions)}\n")
            if positions:
                sys.stdout.write("Open positions time-to-exit:\n")
                for sym, pos in positions.items():
                    delta = pos['exit_time'] - now
                    if delta.total_seconds() > 0:
                        hrs, rem = divmod(int(delta.total_seconds()), 3600)
                        mins, secs = divmod(rem, 60)
                        sys.stdout.write(f"  {sym}: {hrs}h{mins}m{secs}s\n")
                    else:
                        sys.stdout.write(f"  {sym}: EXIT TIME PASSED\n")
                sys.stdout.flush()

            for symbol, params in tqdm(BEST.items(), desc="symbols", leave=False):
                # read params
                try:
                    roc_p  = int(params.get('roc', params.get('roc_period')))
                    thr    = float(params.get('threshold', params.get('thr')))
                    hold_h = int(params.get('hold', params.get('hold_bars')))
                except:
                    print(f"⚠️  Skipping {symbol!r}, malformed params: {params}")
                    continue

                # fetch data
                df = fetch_bars(symbol, roc_p + hold_h)
                if len(df) < roc_p + 1:
                    continue

                closes = df['close']
                roc     = closes.iat[-1] / closes.iat[-1-roc_p] - 1
                last_ts = df.index[-1]
                pos     = positions.get(symbol)

                # EXIT
                if pos and now >= pos['exit_time']:
                    qty        = pos['qty']
                    exit_price = closes.iat[-1] * (1 - FEE_RATE - SLIPPAGE_PCT/2)
                    pnl        = (exit_price - pos['entry_price']) * qty
                    timestamp  = now.isoformat()
                    print(f"{timestamp} ← EXIT {symbol:8} @ {exit_price:.4f}  qty={qty:.6f}  pnl={pnl:.2f}")
                    if not DRY_RUN:
                        exchange.create_order(symbol, 'market', 'sell', qty)

                    # log & persist exit
                    log_trade({
                        'symbol':      symbol,
                        'action':      'EXIT',
                        'timestamp':   timestamp,
                        'price':       f"{exit_price:.4f}",
                        'qty':         f"{qty:.6f}",
                        'roc':         '',
                        'entry_time':  pos['entry_time'],
                        'entry_price': f"{pos['entry_price']:.4f}",
                        'exit_time':   timestamp,
                        'exit_price':  f"{exit_price:.4f}",
                        'pnl':         f"{pnl:.2f}",
                    })
                    del positions[symbol]
                    save_positions()

                # ENTER
                elif (symbol not in positions) and (roc > thr):
                    entry_price = closes.iat[-1] * (1 + FEE_RATE + SLIPPAGE_PCT/2)
                    qty         = NOTIONAL / entry_price
                    exit_time   = last_ts + timedelta(hours=hold_h)
                    timestamp   = now.isoformat()

                    print(f"{timestamp} → ENTER{symbol:8} @ {entry_price:.4f}  qty={qty:.6f}  roc={roc:.4f}")
                    if not DRY_RUN:
                        exchange.create_order(symbol, 'market', 'buy', qty)

                    positions[symbol] = {
                        'entry_time':   timestamp,
                        'entry_price':  entry_price,
                        'qty':          qty,
                        'roc':          roc,
                        'exit_time':    exit_time,
                    }
                    save_positions()

                    # log entry
                    log_trade({
                        'symbol':      symbol,
                        'action':      'ENTER',
                        'timestamp':   timestamp,
                        'price':       f"{entry_price:.4f}",
                        'qty':         f"{qty:.6f}",
                        'roc':         f"{roc:.4f}",
                        'entry_time':  timestamp,
                        'entry_price': f"{entry_price:.4f}",
                        'exit_time':   '',
                        'exit_price':  '',
                        'pnl':         '',
                    })

            sys.stdout.write(f"[{datetime.utcnow().isoformat()}] scan complete.\n")
            sys.stdout.flush()
            sleep_till_next()

    except KeyboardInterrupt:
        print("\n📴  Stopping cleanly…")

if __name__ == '__main__':
    main()




