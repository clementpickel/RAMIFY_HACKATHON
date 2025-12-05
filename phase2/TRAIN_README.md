# Phase 2: Random Forest Trading Bot

This directory contains an improved trading bot using a Random Forest classifier.

## Files

- **train_model.py**: Trains a Random Forest model on historical price data and computes metrics
- **bot_trade.py**: Trading bot that uses the trained model to make allocation decisions
- **main.py**: Main entry point that runs the trading simulation
- **model_rf.pkl**: Saved trained model (created after running train_model.py)

## Usage

### Step 1: Train the Model

```bash
python train_model.py data/asset_b_train.csv model_rf.pkl --show-plot
```

This will:
- Load the training data
- Create technical features (momentum, moving averages, price changes, volatility)
- Train a Random Forest classifier
- Compute metrics (Accuracy, Precision, Recall, F1-Score, ROC-AUC)
- Display a confusion matrix and feature importance
- Save the model to `model_rf.pkl`
- Generate `model_performance.png` with performance visualizations (if `--show-plot` flag is used)

### Step 2: Run the Trading Bot

```bash
python main.py data/asset_b_test.csv
```

The bot will:
- Load the trained Random Forest model (`model_rf.pkl`)
- Generate trading decisions based on model predictions
- Fall back to a simple momentum strategy if the model cannot be used
- Display the portfolio value and performance metrics

## Model Details

### Features Computed

The model uses 18 features derived from price history:
1. **Price Momentum** (10 values): Current price minus the last 10 prices
2. **Moving Averages** (3 values): 5-period, 10-period, and 20-period MAs
3. **Price Changes** (3 values): 1-period, 5-period, and 10-period returns
4. **Volatility** (1 value): Standard deviation of recent returns
5. **Current Price** (1 value)

### Model Configuration

- **Type**: Random Forest Classifier
- **Estimators**: 100 trees
- **Max Depth**: 15
- **Min Samples Split**: 10
- **Min Samples Leaf**: 5
- **Class Weighting**: Balanced (handles imbalanced classes)

### Trading Strategy

The bot's allocation is determined by the model's prediction probability:
- If probability of "up" > 0.7 → 70% Asset B, 30% Cash
- If probability of "up" < 0.3 → 30% Asset B, 70% Cash
- Otherwise → intermediate allocation (30-70% range based on probability)

### Fallback Strategy

If the model cannot be loaded or there's insufficient history, the bot uses a simple momentum-based strategy:
- If price increased → 70% Asset B, 30% Cash
- If price decreased → 30% Asset B, 70% Cash

## Requirements

- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn

Install with: `pip install -r requirement.txt`

## Performance Metrics

After training, the model outputs:
- **Accuracy**: Percentage of correct predictions
- **Precision**: Of predicted "ups", how many were correct
- **Recall**: Of actual "ups", how many were predicted
- **F1-Score**: Harmonic mean of precision and recall
- **ROC-AUC**: Area under the ROC curve
- **Confusion Matrix**: True/False positives and negatives
