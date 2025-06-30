# 2-Resolution Live Trading Bot (`trade_roc_sltp_full.py`)

A Python-based live cryptocurrency trading bot that combines a **1-hour ROC + machine-learning classifier** filter for entries with **5-minute SL/TP** exits. It includes:

- **Raw + processed caching** of 1 h OHLCV data
- **5 m bar** checks for stop-loss / take-profit exits
- **Persistent open-positions** across restarts
- **Manual exit** prompt with a non-blocking timeout
- **Detailed position reporting** and CSV trade logging
- **ThreadPoolExecutor** for parallel symbol processing

---

## Table of Contents

- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [How It Works](#how-it-works)
- [Key Components](#key-components)
- [Customization](#customization)
- [Logging & Data](#logging--data)
- [Troubleshooting](#troubleshooting)
- [License](#license)

---

## Features

- **2-Resolution Logic**  
  - **Entry**: 1 h ROC filter + RandomForest classifier  
  - **Exit**: 5 m SL/TP based on raw entry price  
- **Caching**:  
  - Raw 1 h OHLCV and processed indicator DataFrames, refreshed every 30 m  
- **Parallel Processing**:  
  - Multi-threaded symbol evaluation (configurable `MAX_WORKERS`)  
- **Persistence**:  
  - Open positions saved to `positions.json`  
  - Cache saved to `data_cache.pkl`  
- **Manual Exit**:  
  - Type a symbol to exit manually, with a 3 s non-blocking prompt  
- **Detailed Reporting**:  
  - Market snapshot (first 5 symbols)  
  - Detailed open-positions table, current P/L, TP/SL levels  
- **Dry-Run Mode**:  
  - Set `DRY_RUN=true` to simulate without real orders  

---

## Requirements

- Python 3.8+
- [ccxt](https://github.com/ccxt/ccxt)
- pandas, numpy
- scikit-learn
- python-dotenv
- joblib
- tqdm

---

## Installation

1. Clone this repo:
   ```bash
   git clone https://github.com/yourusername/trade-roc-sltp-bot.git
   cd trade-roc-sltp-bot
Create & activate a virtual environment:

python3 -m venv venv
source venv/bin/activate

Install dependencies:
pip install -r requirements.txt

Configuration

Copy  arb.env and set your credentials & parameters:

KRAKEN_APIKEY=your_api_key
KRAKEN_SECRET=your_api_secret
NOTIONAL=10000
FEE_RATE=0.001
SLIPPAGE_PCT=0.0005
CLASSIFIER_THRESHOLD=0.40
DRY_RUN=true
TRADE_LOG_FILE=trade_log.csv

Place your trained scaler.pkl, classifier.pkl, and best_params.json in the project root.

Usage
Train your model (if not done already):


python3 trainer.py
Outputs scaler.pkl & classifier.pkl

Start live trading:


python3 trade_roc_sltp_full.py
On each 5 m cycle you will see:

Market summary (1 h bars)

Number of open positions

Detailed open-positions table

3 s prompt to enter a symbol for manual exit

Automated entry/exit processing

How It Works
Cache & Indicators

Fetches 1 h OHLCV for each symbol into RAW_CACHE.

Computes indicators (ATR, RV, MA, RSI, volume spike) into PROC_CACHE.

Entry Logic

Calculate 1 h ROC over a symbol-specific period.

If ROC > threshold, run RF classifier on latest features.

If proba ≥ CLASSIFIER_THRESHOLD, verify that the last 20×5 m closes are strictly increasing.

Place market buy at adjusted cost basis, record raw entry price & cost basis.

Exit Logic

For each open position, fetch latest 5 m close.

If price ≥ raw_entry_price×(1+tp_pct) → TP, or ≤ raw_entry_price×(1−sl_pct) → SL, place market sell.

P/L is computed vs. cost basis.

Manual Exit

Non-blocking 3 s prompt. If you type a symbol (e.g. BTC/USD), a manual market sell is executed.

Key Components
process_symbol(symbol)
Entry/exit logic per symbol

show_open_positions()
Prints table of open positions & their P/L, TP/SL levels

manual_exit(symbol)
Force-exit any open position via market sell

Cache Functions

fetch_raw_data / compute_indicators

fetch_latest_5m

Persistence

load_positions() / save_positions()

load_persistent_cache() / save_persistent_cache()

Customization
Timeframes

ENTRY_TIMEFRAME (default 1h)

EXIT_TIMEFRAME (default 5m)

Thresholds & Periods

Per-symbol in best_params.json (roc_period, threshold, sl_pct, tp_pct)

Classifier

Retrain frequency is up to you (e.g. weekly/monthly)

Concurrency

MAX_WORKERS controls parallelism

Logging & Data
Trade log: trade_log.csv
Records every ENTER & EXIT with timestamp, price, qty, P/L

Cache file: data_cache.pkl
Persists raw & processed OHLCV between runs

Positions file: positions.json
Persists open-position state across restarts

Troubleshooting
HTTP 502 / Exchange Not Available
Kraken public endpoints occasionally return 502. The bot backs off automatically, but you may need to restart on prolonged outages.

Missing Feature-Name Errors
If retraining changes feature names, re-run trainer.py and restart the bot with fresh scaler.pkl & classifier.pkl.

“No open positions” KeyErrors
Make sure your positions.json matches the expected schema (raw_entry_price, entry_price, qty, entry_time, cost_basis).

