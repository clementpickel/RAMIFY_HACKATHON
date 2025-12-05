# LSTM Trading Bot - Setup and Usage Guide

## 📁 Project Structure

```
phase_1/
├── data/
│   ├── asset_a_train.csv  (or asset_train.csv)
│   └── asset_a_test.csv   (or asset_test.csv)
├── bot_trade.py           # Your trading bot (loads trained model)
├── train_model.py         # Training script for LSTM model
├── main.py                # Test runner provided by Ramify
└── model_lstm.h5          # Trained model (generated after training)
└── scaler.pkl             # Data scaler (generated after training)
```

## 🚀 Quick Start

### Step 1: Setup Environment

First, make sure you have the virtual environment set up:

```bash
chmod +x setup_env.sh
./setup_env.sh
```

### Step 2: Train the Model

Before running your bot, you need to train the LSTM model:

```bash
python3 train_model.py
```

**What this does:**
- Loads training data from `data/asset_a_train.csv` (or `data/asset_train.csv`)
- Trains an LSTM neural network using expanding window technique
- Evaluates the model on test data (if available)
- Saves the trained model to `model_lstm.h5`
- Saves the data scaler to `scaler.pkl`
- Creates visualization plots:
  - `training_losses.png` - Shows training progress
  - `predictions_comparison.png` - Shows predictions vs actual prices

**Training output:**
```
==================================================
LSTM Trading Bot - Training Script
==================================================
Loading training data from: data/asset_a_train.csv
Loaded 2000 data points
Price range: 95.23 to 105.67

Starting expanding window training...
[Window 1] Training on samples 0 to 500 (440 sequences)
...
Expanding window training completed!

Test MSE: 0.1234
Test RMSE: 0.3512
Directional Accuracy: 56.78%

✓ Model saved to model_lstm.h5
✓ Scaler saved to scaler.pkl
```

### Step 3: Test Your Bot

Once the model is trained, test your bot:

```bash
python3 main.py data/asset_a_test.csv
```

To see the performance graph:

```bash
python3 main.py data/asset_a_test.csv --show-graph
```

## 🧠 How It Works

### LSTM Model Architecture

The model uses a deep LSTM network:
- **Layer 1**: LSTM(50 units) with return sequences
- **Layer 2**: Dropout(0.2) - prevents overfitting
- **Layer 3**: LSTM(50 units) with return sequences
- **Layer 4**: Dropout(0.2)
- **Layer 5**: LSTM(25 units)
- **Layer 6**: Dropout(0.2)
- **Layer 7**: Dense(1) - output layer

### Training Strategy: Expanding Window

The model uses an **expanding window** approach:
1. Start with initial 500 data points
2. Train the model
3. Expand window by 200 points
4. Retrain on all data up to new window
5. Repeat until all data is used

This simulates realistic trading conditions where you continuously update your model with new data.

### Trading Decision Logic

The bot makes decisions based on:

1. **Price Prediction**: Uses last 60 prices to predict next price
2. **Confidence Calculation**: Based on recent volatility
3. **Allocation Strategy**:
   - **Predicted increase (>0.3%)**: Allocate 62-85% to Asset A
   - **Predicted decrease (<-0.3%)**: Allocate 15-38% to Asset A
   - **Neutral**: Keep 50/50 allocation

## 📊 Model Performance Metrics

After training, you'll see:

- **MSE (Mean Squared Error)**: Average squared difference between predictions and actual
- **RMSE (Root Mean Squared Error)**: Square root of MSE, in same units as price
- **MAE (Mean Absolute Error)**: Average absolute difference
- **Directional Accuracy**: % of times the model correctly predicted price direction

Good performance:
- Directional Accuracy > 52% (better than random)
- Lower RMSE/MAE values indicate more accurate predictions

## ⚙️ Configuration

You can adjust training parameters in `train_model.py`:

```python
# Training parameters
lookback = 60              # Number of historical prices to use
initial_window = 500       # Initial training window size
expand_size = 200          # How much to expand window each iteration
```

You can adjust trading strategy in `bot_trade.py`:

```python
threshold_high = 0.003     # 0.3% - minimum increase to go long
threshold_low = -0.003     # -0.3% - minimum decrease to go short
```

## 🔧 Troubleshooting

### Model file not found

**Error**: `✗ Model file not found at model_lstm.h5`

**Solution**: Run `python3 train_model.py` first to train and save the model.

### Training data not found

**Error**: `Error: Training data not found!`

**Solution**: Make sure your CSV file is in the `data/` folder. The script looks for:
- `data/asset_a_train.csv` (first choice)
- `data/asset_train.csv` (fallback)

### Out of memory during training

**Solution**: Reduce batch size or window sizes in `train_model.py`:
```python
model.fit(X_train, y_train, epochs=10, batch_size=16, ...)  # Reduce from 32 to 16
```

### Bot returns 50/50 allocation

This is normal during warm-up (first 60 epochs) while the bot accumulates price history.

## 📈 Improving Performance

### 1. Adjust Lookback Period
```python
lookback = 40  # Try shorter lookback for faster-moving markets
lookback = 90  # Try longer lookback for smoother predictions
```

### 2. Tune Model Architecture
Add more LSTM layers or units for more complex patterns:
```python
LSTM(100, return_sequences=True),  # Increase from 50 to 100
```

### 3. Adjust Training Parameters
```python
epochs=20,              # More epochs for better learning
learning_rate=0.0005,   # Lower learning rate for stability
```

### 4. Feature Engineering
Add technical indicators to the model (momentum, moving averages, etc.)

## 📦 Submission

When submitting, include:
- `bot_trade.py` (your trading bot)
- `model_lstm.h5` (trained model)
- `scaler.pkl` (data scaler)
- Any other Python files your bot depends on

Create submission ZIP:
```bash
zip submission.zip bot_trade.py model_lstm.h5 scaler.pkl
```

## 🎯 Tips for Better Results

1. **Train on more data**: More training data generally leads to better predictions
2. **Monitor directional accuracy**: Focus on predicting direction correctly, not exact prices
3. **Manage risk**: Don't go all-in on predictions - keep some balance
4. **Test thoroughly**: Run multiple tests on different data periods
5. **Watch for overfitting**: If training performance is great but test performance is poor, your model is overfitting

## 📞 Support

If you encounter issues:
1. Check that all dependencies are installed: `pip3 list`
2. Verify TensorFlow installation: `python3 -c "import tensorflow; print(tensorflow.__version__)"`
3. Contact the hackathon team via Discord

Good luck! 🚀