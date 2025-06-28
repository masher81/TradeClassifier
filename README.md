# Kraken ROC-SL/TP Trading System

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

A machine learning-enhanced trading system for Kraken exchange that uses:
- **Rate of Change (ROC)** for entry signals
- **Stop-Loss/Take-Profit (SL/TP)** for exits
- **Random Forest Classifier** to filter trades

## System Components

### 1. `generate_best_params.py`
Brute-force optimizer that finds optimal parameters for all Kraken USD pairs.

### 2. `trade_roc_sltp.py`
Live trading bot using the optimized parameters with ML filtering.

## Features

- 🚀 **Automated parameter optimization** for each trading pair
- ⚖️ **Risk-managed exits** with dynamic SL/TP levels
- 🤖 **ML-powered trade filtering** (Random Forest)
- 📊 **Comprehensive logging** of all trades
- 🧪 **Dry-run mode** for testing
- 📈 **Technical indicators**:
  - ROC, ATR, RSI, Moving Averages
  - Realized volatility, volume spikes

## Installation

# 1. Create a virtual environment
python3 -m venv trading_env

# 2. Activate it
source trading_env/bin/activate  # Linux/macOS
# trading_env\Scripts\activate  # Windows (PowerShell)

# 3. Now install packages safely
pip install ccxt numpy pandas scikit-learn joblib tqdm python-dotenv
python generate_best_params.py
Outputs: best_params.json

Optimization Process:

Tests 150+ parameter combinations per pair

Uses 1 year of hourly data

Maximizes Sharpe ratio

python trade_roc_sltp.py
Live Trading Features:

Runs hourly checks

Logs trades to trade_log.csv

Classifier filters trades (adjust threshold in .env)

Manages open positions with SL/TP
File Structure
├── .env                     # Configuration
├── best_params.json         # Optimized parameters (generated)
├── trade_log.csv            # Trade history (generated)
├── scaler.pkl               # Feature scaler (generated)
├── classifier.pkl           # Trained model (generated)
├── generate_best_params.py  # Parameter optimizer
├── trade_roc_sltp.py        # Main trading bot
└── README.md                # This file

Technical Indicators
Indicator	Description	Usage
ROC	Rate of Change	Entry signal
ATR(20)	Average True Range	Volatility measure
RSI(14)	Relative Strength Index	Overbought/oversold
MA10-MA50	Moving Average Spread	Trend direction
RV20	20-period Realized Volatility	Risk assessment

Requirements

ccxt==4.2.85
numpy==1.26.0
pandas==2.1.0
scikit-learn==1.3.0
joblib==1.3.1
tqdm==4.66.1
python-dotenv==1.0.0
