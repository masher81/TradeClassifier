#!/usr/bin/env python3
"""
trade_roc_sltp_full.py

2-resolution live trader:
  • 1 h ROC + classifier filter for entries
  • 5 m bars for TP exits (plus live TP via WS)
  • 1 h bars for SL exits
Preserves:
  – caching (raw+processed hourly)
  – verbose/logging/market snapshot
  – ThreadPoolExecutor
  – persistent open positions across restarts
Enhancements:
  – detailed open-positions table
  – non-blocking manual-exit prompt (3 s timeout)
  – live TP exits via Kraken WebSocket
"""
import os, sys, time, csv, json, select, threading, asyncio
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import ccxt
import joblib
import websockets
from dotenv import load_dotenv
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from ccxt.base.errors import ExchangeNotAvailable, NetworkError

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
CACHE_EXPIRY         = timedelta(minutes=30)
MAX_WORKERS          = 5
VERBOSE              = True

# ─────────────── LOAD PARAMS & MODELS ───────────────────────────────────

with open('best_params.json') as f:
    BEST = json.load(f)
SYMBOLS = list(BEST.keys())

scaler, classifier = joblib.load('scaler.pkl'), joblib.load('classifier.pkl')

# ─────────────── SETUP EXCHANGE ─────────────────────────────────────────

exchange = ccxt.kraken({
    'enableRateLimit': True,
    'apiKey':          API_KEY,
    'secret':          API_SECRET,
})
exchange.load_markets()

# ─────────────── CACHING SYSTEM ─────────────────────────────────────────

RAW_CACHE = {}   # symbol -> (timestamp, raw_bars)
PROC_CACHE = {}  # symbol -> (timestamp, df)

def load_persistent_cache():
    global RAW_CACHE, PROC_CACHE
    try:
        data = joblib.load(CACHE_FILE)
        RAW_CACHE.update(data.get('raw', {}))
        PROC_CACHE.update(data.get('proc', {}))
        if VERBOSE:
            print(f"♻️ Loaded cache: {len(RAW_CACHE)} raw, {len(PROC_CACHE)} processed")
    except FileNotFoundError:
        if VERBOSE:
            print("♻️ No cache file – starting fresh")

def save_persistent_cache():
    joblib.dump({'raw': RAW_CACHE, 'proc': PROC_CACHE}, CACHE_FILE)
    if VERBOSE:
        print(f"♻️ Saved cache: {len(RAW_CACHE)} raw, {len(PROC_CACHE)} processed")

def validate_cache():
    now = datetime.utcnow()
    for d in (RAW_CACHE, PROC_CACHE):
        for sym,(ts,_) in list(d.items()):
            if now - ts > CACHE_EXPIRY:
                del d[sym]
    if VERBOSE:
        print(f"♻️ Pruned cache older than {CACHE_EXPIRY}")

# ─────────────── RATE LIMIT HELPERS ─────────────────────────────────────

_last_req = 0.0
def rate_limited_request():
    global _last_req
    now = time.time()
    if now - _last_req < 1.0:
        time.sleep(1.0 - (now - _last_req))
    _last_req = time.time()

# ─────────────── DATA FETCH & INDICATORS ────────────────────────────────

def fetch_raw_data(symbol):
    now = datetime.utcnow()
    if symbol in RAW_CACHE:
        ts,data = RAW_CACHE[symbol]
        if now - ts < CACHE_EXPIRY and len(data)>=MIN_DATA_BARS:
            return data
    since = exchange.milliseconds() - int(HISTORY_HOURS*3600*1000)
    all_bars=[]
    while True:
        rate_limited_request()
        chunk = exchange.fetch_ohlcv(symbol, ENTRY_TIMEFRAME, since, limit=500)
        if not chunk: break
        all_bars+=chunk
        since = chunk[-1][0]+1
        if len(all_bars)>=HISTORY_HOURS+2: break
    if len(all_bars)>=MIN_DATA_BARS:
        RAW_CACHE[symbol]=(now,all_bars)
        return all_bars
    return None

def compute_indicators(raw_bars):
    df = pd.DataFrame(raw_bars,columns=['ts','open','high','low','close','vol'])
    df['dt']=pd.to_datetime(df['ts'],unit='ms')
    df.set_index('dt',inplace=True)
    df.sort_index(inplace=True)
    df['prev_close']=df['close'].shift(1)
    df['tr']=np.maximum.reduce([
      df['high']-df['low'],
      (df['high']-df['prev_close']).abs(),
      (df['low']-df['prev_close']).abs(),
    ])
    df['atr20']=df['tr'].rolling(20).mean()
    df['rv20']=np.log(df['close']).diff().rolling(20).std()
    df['ma10']=df['close'].rolling(10).mean()
    df['ma50']=df['close'].rolling(50).mean()
    df['ma10_50']=df['ma10']-df['ma50']
    delta=df['close'].diff()
    up,down=delta.clip(lower=0),-delta.clip(upper=0)
    df['rsi14']=100-100/(1+up.rolling(14).mean()/down.rolling(14).mean())
    df['vol_spike']=df['vol']/df['vol'].rolling(20).mean()
    return df.dropna()

def load_history(symbol):
    now = datetime.utcnow()
    if symbol in PROC_CACHE:
        ts,df = PROC_CACHE[symbol]
        if now - ts < CACHE_EXPIRY and len(df)>=MIN_DATA_BARS:
            return df
    raw = fetch_raw_data(symbol)
    if not raw: return None
    df = compute_indicators(raw)
    PROC_CACHE[symbol] = (now,df)
    return df

def fetch_latest_5m(symbol):
    rate_limited_request()
    bars = exchange.fetch_ohlcv(symbol, EXIT_TIMEFRAME, limit=2)
    if not bars: return None,None
    df = pd.DataFrame(bars,columns=['ts','o','h','l','c','v'])
    df['dt']=pd.to_datetime(df['ts'],unit='ms')
    df.set_index('dt',inplace=True)
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
    """Load open positions from disk, and migrate old entries."""
    try:
        with open(POSITIONS_FILE) as f:
            raw = json.load(f)
        for sym, pos in raw.items():
            # parse the timestamp
            pos['entry_time'] = datetime.fromisoformat(pos['entry_time'])
            # if this position was from the old format:
            if 'raw_entry_price' not in pos:
                pos['raw_entry_price'] = pos.get('entry_price')
            if 'cost_basis' not in pos:
                # cost_basis = entry_price including fees/slippage
                pos['cost_basis'] = pos['raw_entry_price'] * (1 + FEE_RATE + SLIPPAGE_PCT/2)
        return raw
    except:
        return {}


def save_positions(pos):
    serial={}
    for sym,p in pos.items():
        serial[sym]={
          'entry_time': p['entry_time'].isoformat(),
          'entry_price': p.get('raw_entry_price',p.get('entry_price',0)),
          'qty': p['qty']
        }
    with open(POSITIONS_FILE,'w') as f:
        json.dump(serial,f,indent=2)

# ─────────────── DETAILED POSITIONS / MANUAL EXIT ───────────────────────

positions={}  # symbol -> {entry_time,raw_entry_price,cost_basis,qty,...}

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
        # fall back safely if keys are missing
        raw       = pos.get('raw_entry_price', pos.get('entry_price', float('nan')))
        basis     = pos.get('cost_basis', raw * (1 + FEE_RATE + SLIPPAGE_PCT/2))
        qty       = pos.get('qty', float('nan'))

        # fetch last 5m close
        _, price5 = fetch_latest_5m(sym)
        price5    = price5 if price5 is not None else float('nan')

        tp_lvl = raw * (1 + BEST[sym]['tp_pct'])
        sl_lvl = raw * (1 - BEST[sym]['sl_pct'])
        pnl    = (price5 - basis) * qty

        print(
            f"{sym:<10} "
            f"{pos['entry_time'].isoformat():<19} "
            f"{raw:7.4f} "
            f"{basis:7.4f} "
            f"{qty:10.4f} "
            f"{price5:8.4f} "
            f"{pnl:10.2f} "
            f"{tp_lvl:8.4f} "
            f"{sl_lvl:8.4f}"
        )
    print()


def manual_exit(symbol):
    if symbol not in positions:
        print(f"❌ No open position for {symbol}")
        return
    p=positions[symbol]
    raw=p['raw_entry_price']; basis=p['cost_basis']; qty=p['qty']
    _,pr5=fetch_latest_5m(symbol)
    if pr5 is None:
        print(f"❌ No price for {symbol}, aborting manual exit.")
        return
    ex=pr5*(1-FEE_RATE-SLIPPAGE_PCT/2)
    pnl=(ex-basis)*qty; ts=datetime.utcnow().isoformat()
    print(f"{ts} ← MANUAL EXIT {symbol} @ {ex:.4f}  pnl={pnl:.2f}")
    if not DRY_RUN:
        exchange.create_order(symbol,'market','sell',qty)
    log_trade({
      'symbol':symbol,'action':'EXIT','timestamp':ts,'price':f"{ex:.4f}",'qty':f"{qty:.6f}",
      'roc':'','prob':'','entry_time':p['entry_time'].isoformat(),
      'entry_price':f"{raw:.4f}",'exit_time':ts,'exit_price':f"{ex:.4f}",
      'exit_type':'MANUAL','pnl':f"{pnl:.2f}"
    })
    del positions[symbol]

# ─────────────── LIVE TP MONITOR VIA WEBSOCKET ─────────────────────────

async def live_tp_monitor():
    uri = "wss://ws.kraken.com"
    async with websockets.connect(uri) as ws:
        sub = {"event":"subscribe","subscription":{"name":"ticker"},"pair":SYMBOLS}
        await ws.send(json.dumps(sub))
        if VERBOSE:
            print("� Subscribed to Kraken WS ticker feed for live TP")
        async for msg in ws:
            data = json.loads(msg)
            if not isinstance(data,list) or len(data)<4 or data[2]!="ticker":
                continue
            payload = data[1]
            pair    = data[3]
            last_px = float(payload["c"][0])
            if pair in positions:
                raw = positions[pair]['raw_entry_price']
                tp  = raw*(1+BEST[pair]['tp_pct'])
                if last_px >= tp:
                    p=positions[pair]; qty=p['qty']; basis=p['cost_basis']
                    ex=last_px*(1-FEE_RATE-SLIPPAGE_PCT/2)
                    pnl=(ex-basis)*qty; ts=datetime.utcnow().isoformat()
                    print(f"{ts} ← LIVE TP EXIT {pair} @ {ex:.4f}  pnl={pnl:.2f}")
                    if not DRY_RUN:
                        exchange.create_order(pair,'market','sell',qty)
                    log_trade({
                      'symbol':pair,'action':'EXIT','timestamp':ts,'price':f"{ex:.4f}",'qty':f"{qty:.6f}",
                      'roc':'','prob':'','entry_time':p['entry_time'].isoformat(),
                      'entry_price':f"{raw:.4f}",'exit_time':ts,'exit_price':f"{ex:.4f}",
                      'exit_type':'TP_LIVE','pnl':f"{pnl:.2f}"
                    })
                    del positions[pair]

# ─────────────── TRADING LOGIC ─────────────────────────────────────────

def process_symbol(symbol):
    """
    1) If in a position, check:
       • 5m TP fallback
       • 1h SL
    2) Else entry: 1h ROC + classifier + 20×5m monotonic
    """
    global positions
    try:
        params = BEST[symbol]

        # EXIT for existing positions
        if symbol in positions:
            ent = positions[symbol]
            raw=ent['raw_entry_price']; basis=ent['cost_basis']; qty=ent['qty']
            tp= raw*(1+params['tp_pct'])
            sl= raw*(1-params['sl_pct'])
            etype=None

            # 5m TP fallback
            _,c5 = fetch_latest_5m(symbol)
            if c5 and c5>=tp:
                etype='TP'

            # 1h SL
            df1h = load_history(symbol)
            if df1h is not None:
                c1h = df1h['close'].iat[-1]
                if c1h<=sl:
                    etype='SL'

            if etype:
                ex = (c5 or c1h)*(1-FEE_RATE-SLIPPAGE_PCT/2)
                pnl=(ex-basis)*qty; ts=datetime.utcnow().isoformat()
                print(f"{ts} ← EXIT {symbol} @ {ex:.4f} ({etype})  pnl={pnl:.2f}")
                if not DRY_RUN:
                    exchange.create_order(symbol,'market','sell',qty)
                log_trade({
                  'symbol':symbol,'action':'EXIT','timestamp':ts,'price':f"{ex:.4f}",
                  'qty':f"{qty:.6f}",'roc':'','prob':'','entry_time':ent['entry_time'].isoformat(),
                  'entry_price':f"{raw:.4f}",'exit_time':ts,'exit_price':f"{ex:.4f}",
                  'exit_type':etype,'pnl':f"{pnl:.2f}"
                })
                del positions[symbol]
                return 'EXIT'
            return None

        # ENTRY logic for new positions
        df = load_history(symbol)
        if df is None or len(df)<max(params['roc_period'],50,20,14):
            return None
        last,prev = df['close'].iat[-1], df['close'].iat[-1-params['roc_period']]
        roc = last/prev-1
        if roc<=params['threshold']:
            return None

        now=datetime.utcnow()
        feat=pd.DataFrame([{
          'roc':roc,'atr20':df['atr20'].iat[-1],'rv20':df['rv20'].iat[-1],
          'ma10_50':df['ma10_50'].iat[-1],'rsi14':df['rsi14'].iat[-1],
          'vol_spike':df['vol_spike'].iat[-1],'hold_hours':params.get('hold_bars',1),
          'hour':now.hour,'weekend':int(now.weekday()>=5),
          'sl_pct':params['sl_pct'],'tp_pct':params['tp_pct']
        }])
        prob=classifier.predict_proba(scaler.transform(feat))[0,1]
        if prob<CLASSIFIER_THRESHOLD:
            return None

        rate_limited_request()
        bars5=exchange.fetch_ohlcv(symbol, EXIT_TIMEFRAME, limit=20)
        if not bars5 or len(bars5)<20:
            return None
        closes5=[b[4] for b in bars5]
        if any(closes5[i]<closes5[i-1] for i in range(1,len(closes5))):
            if VERBOSE:
                print(f"⛔ {symbol}: skipped entry, 5m not strictly rising")
            return None

        raw=last
        basis=last*(1+FEE_RATE+SLIPPAGE_PCT/2)
        qty=NOTIONAL/basis
        ts=now.isoformat()
        print(f"{ts} → ENTER {symbol} @ {basis:.4f}  roc={roc:.2%}  P={prob:.2f}")
        if not DRY_RUN:
            exchange.create_order(symbol,'market','buy',qty)
        positions[symbol]={'entry_time':now,'raw_entry_price':raw,'cost_basis':basis,'qty':qty}

        log_trade({
          'symbol':symbol,'action':'ENTER','timestamp':ts,'price':f"{basis:.4f}",
          'qty':f"{qty:.6f}",'roc':f"{roc:.4f}",'prob':f"{prob:.2f}",
          'entry_time':ts,'entry_price':f"{raw:.4f}",'exit_time':'','exit_price':'',
          'exit_type':'','pnl':''
        })
        return 'ENTER'

    except (ExchangeNotAvailable,NetworkError) as e:
        if VERBOSE:
            print(f"⚠️ [{symbol}] exchange issue, skipping: {e}")
        return None
    except Exception as e:
        print(f"❌ [{symbol}] error: {type(e).__name__} {e}")
        return None

# ─────────────── MARKET SNAPSHOT ────────────────────────────────────────

def print_market_summary():
    print("\n� Market Summary (1h bars):")
    for sym in SYMBOLS[:5]:
        df=load_history(sym)
        if df is None or len(df)<2: continue
        last,prev=df['close'].iloc[-1],df['close'].iloc[-2]
        pct=(last/prev-1)*100; vol=df['vol'].iloc[-1]
        print(f"  {sym:8} {last:.4f} ({pct:+.2f}%)  Vol={vol:.0f}")

# ─────────────── MAIN LOOP ─────────────────────────────────────────────

def main():
    global positions
    load_persistent_cache()
    positions = load_positions()

    print(f"▶️ Starting 2-res live trader — Dry-run={DRY_RUN}, TH={CLASSIFIER_THRESHOLD}")
    print_market_summary()

    # launch live-TP WS monitor in background
    threading.Thread(target=lambda: asyncio.run(live_tp_monitor()), daemon=True).start()

    try:
        while True:
            validate_cache()
            print(f"\n[{datetime.utcnow().isoformat()}] Open positions: {len(positions)}")
            show_open_positions()

            # manual exit prompt (3s timeout)
            sys.stdout.write("Type a symbol to EXIT manually (3s)… ")
            sys.stdout.flush()
            r,_,_ = select.select([sys.stdin],[],[],3)
            if r:
                choice=sys.stdin.readline().strip().upper()
                if choice:
                    manual_exit(choice)
                    save_persistent_cache()
                    save_positions(positions)
                    continue
            else:
                print()

            # automated pass
            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as exe:
                for f in as_completed({exe.submit(process_symbol,s):s for s in SYMBOLS}):
                    f.result()

            save_persistent_cache()
            save_positions(positions)

            # sleep til next 5m boundary
            now=datetime.utcnow()
            secs=300-(now.minute%5*60+now.second)
            time.sleep(max(secs,1))

    except KeyboardInterrupt:
        print("\n� Stopping cleanly…")
        save_persistent_cache()
        save_positions(positions)

if __name__=="__main__":
    main()










