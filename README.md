# Enhanced ROC-Based Trade Classifier

A Python pipeline that backtests a simple Rate-of-Change (ROC) trading strategy across dozens of crypto spot pairs, extracts rich market and trade features at each entry point, trains a calibrated XGBoost classifier to predict trade profitability, and overlays its predictions back onto the historical PnL stream.

├── best_params.json # Per-symbol ROC/threshold/hold settings
├── hist_trade_roc_trades.csv # Raw trade log from backtest (ENTER/EXIT with timestamps, prices, PnL)
├── enhanced_classifier.py # Main feature-engineering, model training & evaluation script
├── requirements.txt # pip-installable dependencies
└── README.md # This file


---

## ⚙️ Requirements

- Python ≥ 3.8  
- pip  
- (optionally) a virtualenv

ccxt
pandas
numpy
tqdm
scikit-learn
imbalanced-learn
xgboost
