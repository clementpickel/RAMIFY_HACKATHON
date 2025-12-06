# Multi-Asset Trading Bot - Phase 3

## Overview
Advanced trading bot that manages a portfolio of two risky assets (Asset A, Asset B) plus cash allocation.

## Strategy Components

### 1. **Independent Asset Analysis**
Each asset is analyzed separately using:
- **Moving Averages** (SMA-20, SMA-10): Trend identification
- **Momentum**: Rate of price change
- **Volatility**: Risk measurement
- **RSI**: Overbought/oversold detection

### 2. **Individual Asset Signals**
For each asset, a signal is calculated ranging from -1.0 (strong sell) to 1.0 (strong buy) based on:
- **Price vs SMA**: ±0.3 points
- **Momentum**: ±0.2 points  
- **RSI Extremes**: ±0.15 points
- **Volatility Adjustment**: ±0.1 points

### 3. **Portfolio Allocation**

#### Base Allocation
```
allocation = 0.5 + (signal / 2) * 0.3
Range: [0.2, 0.8] per asset
```

#### Risk Management
- **Volatility-Based Cash Buffer**: 20-50%
  - Low volatility → 20% cash
  - High volatility → 50% cash
- **Diversification Constraints**:
  - No single asset more than 60% of risky portfolio
  - Minimum 5% per asset for diversification

#### Signal Consensus Adjustments
- **Both Bullish** (signals > 0.4): Reduce cash by 5%
- **Both Bearish** (signals < -0.4): Increase cash by 5%

### 4. **Performance Characteristics**

| Dataset | Sharpe Score | PnL | Max Drawdown |
|---------|-------------|-----|--------------|
| asset_a_b_train_1.csv | 0.1004 | +13.50% | -83.86% |
| asset_a_b_train_2.csv | 0.2632 | +37.37% | -84.05% |

## Key Features

✅ **Multi-Asset Support**: Analyzes 2+ assets independently  
✅ **Dynamic Risk Management**: Adjusts cash allocation based on volatility  
✅ **Diversification Enforcement**: Prevents concentration in single asset  
✅ **Consensus-Based Boosting**: Rewards agreement between assets  
✅ **Technical Analysis**: 6+ indicators per asset  
✅ **No Look-Ahead Bias**: Uses only past data for decisions  

## Technical Indicators

### Per Asset:
- Simple Moving Average (20-period)
- Simple Moving Average (10-period)
- Volatility (standard deviation)
- Momentum (cumulative return)
- RSI (relative strength index)
- Price vs Moving Average ratio

## Allocation Bounds

**Minimum Allocation**: 5% per asset, 10% cash  
**Maximum Allocation**: 60% per asset (of risky capital), 50% cash  
**Required Sum**: 100%

## Usage

### Run backtest
```bash
python main.py data/asset_a_b_train_1.csv
```

### View performance graph
```bash
python main.py data/asset_a_b_train_1.csv --show-graph
```

## Files
- `bot_trade.py`: Multi-asset trading logic
- `main.py`: Backtest runner
- `scoring/scoring.py`: Performance metrics
- `data/asset_a_b_train_*.csv`: Training datasets

## How It Works

1. **Warmup Phase** (10 candles): Equal allocation (33% each)
2. **Analysis Phase**:
   - Calculate 6 indicators for Asset A
   - Calculate 6 indicators for Asset B
   - Generate signals (-1 to 1)
3. **Allocation Phase**:
   - Base allocation from signals
   - Apply diversification constraints
   - Adjust cash based on volatility
   - Boost/reduce based on signal consensus
4. **Normalization**: Ensure allocations sum to 1.0

## Edge Cases Handled

- ✅ Insufficient data (warmup period)
- ✅ Division by zero (momentum calculation)
- ✅ Deque slicing limitations
- ✅ Allocation normalization edge cases
- ✅ Minimum diversification enforcement
- ✅ NaN/inf value handling

## Future Improvements

- Train ML model to predict asset correlations
- Add transaction cost modeling
- Optimize allocation weights via backtesting
- Consider regime-based switching (trending vs mean-reversion)
- Add mean-reversion detection for pair trading
