#!/usr/bin/env python3
import os
import sys
import time
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

# track open positions: symbol → {'exit_time': datetime, 'qty': float}
positions = {}

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

def main():
    print(f"▶️  Starting ROC‐based trader...  Dry run={DRY_RUN}, classifier threshold={THRESHOLD}")
    try:
        while True:
            now = datetime.utcnow()
            # heartbeat
            sys.stdout.write(f"\n[{now.isoformat()}] scanning {len(BEST)} symbols; open positions: {len(positions)}\n")
            sys.stdout.flush()

            # per-symbol progress bar
            for symbol, params in tqdm(BEST.items(), desc="symbols", leave=False):
                # read params
                try:
                    roc_p  = int(params.get('roc', params.get('roc_period')))
                    thr    = float(params.get('threshold', params.get('thr')))
                    hold_h = int(params.get('hold', params.get('hold_bars')))
                except Exception:
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
                    qty   = pos['qty']
                    price = closes.iat[-1] * (1 - FEE_RATE - SLIPPAGE_PCT/2)
                    print(f"{now.isoformat()} ← EXIT {symbol:8} @ {price:.4f}  qty={qty:.6f}")
                    if DRY_RUN:
                        print(f"  [DRY_RUN] would SELL {symbol} qty={qty:.6f}")
                    else:
                        exchange.create_order(symbol, 'market', 'sell', qty)
                    del positions[symbol]

                # ENTER
                elif (symbol not in positions) and (roc > thr):
                    price = closes.iat[-1] * (1 + FEE_RATE + SLIPPAGE_PCT/2)
                    qty   = NOTIONAL / price
                    print(f"{now.isoformat()} → ENTER {symbol:8} @ {price:.4f}  qty={qty:.6f}  roc={roc:.4f}")
                    if DRY_RUN:
                        print(f"  [DRY_RUN] would BUY  {symbol} qty={qty:.6f}")
                    else:
                        exchange.create_order(symbol, 'market', 'buy', qty)
                    exit_time = last_ts + timedelta(hours=hold_h)
                    positions[symbol] = {'exit_time': exit_time, 'qty': qty}

            # end of one full scan
            sys.stdout.write(f"[{datetime.utcnow().isoformat()}] scan complete.\n")
            sys.stdout.flush()

            sleep_till_next()

    except KeyboardInterrupt:
        print("📴  Stopping cleanly…")

if __name__ == '__main__':
    main()

