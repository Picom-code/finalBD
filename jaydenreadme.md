# Big Data Final Project - Stock Prediction with Deep Learning 📈

Hey Jayden! Here's everything you need to know about this project. I'll break it down so you can get it running quick.

## What This Project Does

We're predicting stock prices for **TAN**, **GOOG**, and **SOXL** using 8 different machine learning models. Then we compare which ones work best and which are most energy efficient.

---

## Quick Setup (For Jayden)

```bash
# 1. Clone the repo
git clone https://github.com/Picom-code/finalBD.git
cd finalBD

# 2. Create a virtual environment
python3 -m venv venv
source venv/bin/activate

# 3. Install dependencies
pip install torch xgboost scikit-learn pandas numpy matplotlib scipy pynvml

# 4. Run the script (takes about 1 hour on a good GPU)
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python stock_prediction_final.py
```

If you don't have a GPU, it'll still run but take longer. The script auto-detects CPU/GPU.

---

## Files in This Repo

| File | What It Does |
|------|--------------|
| `stock_prediction_final.py` | The main script - runs all 8 models on all 3 stocks |
| `TAN.csv` | Solar energy ETF stock data (93k+ minute-level records) |
| `GOOG.csv` | Google stock data (36k records) |
| `SOXL.csv` | Semiconductor 3x leveraged ETF (2.7k records) |
| `results.txt` | All the output - R², MSE, training times, analysis |
| `TAN_learning_curves.png` | Training loss graphs for TAN |
| `TAN_predictions.png` | Actual vs predicted plots for TAN |
| `GOOG_learning_curves.png` | Training loss graphs for GOOG |
| `GOOG_predictions.png` | Actual vs predicted plots for GOOG |
| `SOXL_learning_curves.png` | Training loss graphs for SOXL |
| `SOXL_predictions.png` | Actual vs predicted plots for SOXL |
| `eei_comparison.png` | Energy Efficiency Index comparison chart |

---

## The 8 Models We Used

1. **DNN** - Deep Neural Network (basic feedforward, 2048→1024→512→256→128→64→32→1)
2. **CNN** - Convolutional Neural Network (multi-scale convolutions for pattern detection)
3. **LSTM** - Long Short-Term Memory (great for sequences, captures long-range patterns)
4. **GRU** - Gated Recurrent Unit (like LSTM but faster)
5. **Transformer** - Self-attention mechanism (the GPT architecture basically)
6. **LNN** - Liquid Neural Network (multi-timescale dynamics, experimental)
7. **Extra Trees** - Ensemble of random decision trees
8. **XGBoost** - Gradient boosted trees (GPU accelerated)

Each model was trained twice - baseline and tuned (with better hyperparameters).

---

## Assignment Questions & Answers

### Q1: Use 80/20 train/test split with DNN, CNN, LNN, GRU, Transformer, LSTM, Extra Trees, XGBoost

✅ **Done.** All 8 models trained on all 3 stocks with 80/20 split.

### Q2: Give solid reason for your network architecture

✅ **Done.** See the `results.txt` file under "MODEL ARCHITECTURE DESIGN RATIONALE". Quick summary:

- **DNN**: Large layers for feature interactions, dropout to prevent overfitting
- **CNN**: Multi-scale kernels (3,5,7) to catch patterns at different time scales  
- **LSTM**: Bidirectional + stacked layers for temporal dependencies
- **Transformer**: 8-head attention, 4 transformer blocks, positional encoding
- **LNN**: Multiple GRU paths with different activations to simulate continuous-time dynamics

### Q3: Compare learning behaviors and R²/MSE

✅ **Done.** Here's the summary:

#### TAN Stock Results
| Model | R² | MSE | Training Time |
|-------|-----|-----|---------------|
| DNN | 0.1224 | 0.0091 | 27s |
| CNN | -2.15 | 0.0326 | 29s |
| LSTM | -1.24 | 0.0232 | 55s |
| GRU | -1.53 | 0.0261 | 51s |
| Transformer | -1.88 | 0.0298 | 111s |
| LNN | -0.02 | 0.0105 | 203s |
| Extra Trees | -1.03 | 0.0210 | 778s |
| XGBoost | -1.19 | 0.0227 | 8s |
| **Tuned DNN** | **0.7531** | **0.0026** | 36s |
| **Tuned LNN** | **0.4410** | **0.0058** | 146s |

#### GOOG Stock Results
| Model | R² | MSE |
|-------|-----|-----|
| Extra Trees | -0.62 | 0.0297 |
| XGBoost | -0.70 | 0.0312 |
| **Tuned DNN** | **0.7870** | **0.0039** |
| **Tuned Transformer** | **0.3719** | **0.0115** |

#### SOXL Stock Results (Best Results!)
| Model | R² | MSE |
|-------|-----|-----|
| LSTM | 0.7735 | 0.00025 |
| **Extra Trees** | **0.8718** | **0.00014** |
| **XGBoost** | **0.7849** | **0.00023** |

### Q4: Tune models for 3%+ improvement

✅ **Done.** Best improvements:

- TAN Tuned DNN: **515% improvement** (0.12 → 0.75 R²)
- TAN Tuned LNN: **2705% improvement** (basically went from garbage to decent)
- GOOG Tuned DNN: **126% improvement**
- GOOG Tuned Transformer: **108% improvement**

Key tuning changes:
- Used SGD optimizer with momentum instead of Adam
- Smaller batch sizes (256 → 128)
- Lower learning rates (0.001 → 0.0003-0.0005)

### Q5: Add meaningful features

✅ **Done.** Added 34 technical indicators:

- **Moving Averages**: SMA/EMA at 5, 10, 20, 50 periods
- **MACD**: 12/26 EMA with signal line
- **RSI**: 14-period Relative Strength Index
- **Bollinger Bands**: Width and position
- **Stochastic Oscillator**: %K and %D
- **ATR**: Average True Range
- **Momentum/ROC**: 5, 10, 20 period
- **Volume indicators**: Volume ratio, volume SMA
- **Lag features**: Previous 1, 2, 3, 5, 10 close prices and returns

### Q6: Build Energy Efficiency Index (EEI)

✅ **Done.** 

**Formula:** `EEI = (R² × 1000) / Energy_kJ`

Best EEI scores:
- **XGBoost on SOXL: 1255.69** (super efficient!)
- LSTM on SOXL: 149.70
- Extra Trees on SOXL: 77.70

This means XGBoost gives you the most prediction accuracy per watt of power used. Makes sense - it's fast and GPU-optimized.

**How to extend EEI to other models:**
```
Classification: EEI = (Accuracy × 1000) / Energy_kJ
Multi-metric:   EEI = (weighted_metrics × 1000) / Energy_kJ  
With inference: EEI = Performance × 1000 / (Train_Energy + Inference_Energy)
```

### Q7: Calculate Wasserstein, Euclidean, Cosine distances between train/val loss curves

✅ **Done.** Example from TAN:

| Model | Wasserstein | Euclidean | Cosine |
|-------|-------------|-----------|--------|
| DNN | 0.462 | 0.495 | 0.386 |
| Transformer | 0.084 | 0.098 | 0.016 |
| LNN | 0.403 | 0.461 | 0.626 |

**What we found:**
- Transformer has the lowest distances → most consistent learning (train and val loss track closely)
- High distances usually mean overfitting
- Cosine distance shows if the models learn similar *patterns* regardless of magnitude

---

## Conclusions (TL;DR for the Presentation)

1. **Best Models Overall**: XGBoost and Extra Trees for small datasets, Tuned DNN with SGD for larger ones

2. **SGD > Adam**: For these stock datasets, SGD with momentum actually worked way better than Adam

3. **Feature Engineering Matters**: Those 34 technical indicators made a huge difference

4. **Energy Efficiency**: Tree-based models (XGBoost, Extra Trees) are way more energy efficient than deep learning models

5. **LSTM Shines on Small Data**: On SOXL (smallest dataset), LSTM got 0.77 R² baseline - best deep learning performance

6. **Transformers Need More Data**: They struggled on smaller datasets but did okay on GOOG

---

## If Something Breaks

1. **CUDA out of memory**: The script handles this, but if it crashes, reduce batch_size in the code (search for `batch_size = 512` and make it smaller)

2. **No GPU**: It'll auto-fallback to CPU, just slower

3. **Missing packages**: 
   ```bash
   pip install torch xgboost scikit-learn pandas numpy matplotlib scipy pynvml
   ```

4. **Python version**: Tested on Python 3.12, should work on 3.9+

---

## Output Location

Everything gets saved to the same folder:
- `results.txt` - All the numbers and analysis
- `*_learning_curves.png` - Training graphs
- `*_predictions.png` - Actual vs predicted scatter plots
- `eei_comparison.png` - Energy efficiency bar chart

---

Good luck with the presentation! Hit me up if anything's broken.

