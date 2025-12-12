# Big Data Final Project - Stock Prediction with Deep Learning

## Overview
Predicting stock prices for TAN, GOOG, and SOXL using 8 different ML models (DNN, CNN, LSTM, GRU, Transformer, LNN, Extra Trees, XGBoost).

**See [jaydenreadme.md](jaydenreadme.md) for the full breakdown!**

## Quick Start
```bash
python3 -m venv venv && source venv/bin/activate
pip install torch xgboost scikit-learn pandas numpy matplotlib scipy pynvml
python stock_prediction_final.py
```

## Best Results
| Stock | Best Model | R² Score |
|-------|-----------|----------|
| TAN | Tuned DNN | 0.7531 |
| GOOG | Tuned DNN | 0.7870 |
| SOXL | Extra Trees | 0.8718 |

## Files
- `stock_prediction_final.py` - Main script
- `results.txt` - All output and analysis
- `*_learning_curves.png` - Training visualizations
- `*_predictions.png` - Prediction plots
- `eei_comparison.png` - Energy efficiency comparison

