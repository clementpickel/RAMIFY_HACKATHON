#!/usr/bin/env python3
"""
Neural Network Training for Multi-Asset Trading Bot
Trains a neural network to predict optimal portfolio allocations
"""

import numpy as np
import pandas as pd
import joblib
import os
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

def create_features(data_a, data_b, lookback=30):
    """
    Create feature set from multi-asset price history.
    
    Features for each asset:
    - Recent returns (5 lags)
    - Moving averages ratios
    - Momentum indicators
    - Volatility measures
    - Price position in range
    
    Args:
        data_a: Array of Asset A prices
        data_b: Array of Asset B prices
        lookback: Number of past periods to use
        
    Returns:
        X: Feature matrix (n_samples, n_features)
        y: Target allocations (n_samples, 3) - [Asset A, Asset B, Cash]
    """
    X, y = [], []
    
    for i in range(lookback, len(data_a) - 1):
        window_a = data_a[i-lookback:i]
        window_b = data_b[i-lookback:i]
        
        # ========== Asset A Features ==========
        returns_a = np.diff(window_a) / window_a[:-1]
        
        # Lagged returns
        lag_1_a = returns_a[-1] if len(returns_a) > 0 else 0
        lag_5_a = np.mean(returns_a[-5:]) if len(returns_a) >= 5 else 0
        lag_10_a = np.mean(returns_a[-10:]) if len(returns_a) >= 10 else 0
        
        # Moving averages
        sma_10_a = np.mean(window_a[-10:])
        sma_20_a = np.mean(window_a[-20:]) if len(window_a) >= 20 else np.mean(window_a)
        price_vs_sma_a = (window_a[-1] - sma_20_a) / sma_20_a if sma_20_a != 0 else 0
        
        # Momentum
        momentum_a = (window_a[-1] - window_a[0]) / window_a[0] if window_a[0] != 0 else 0
        
        # Volatility
        volatility_a = np.std(returns_a) if len(returns_a) > 0 else 0
        
        # RSI-like indicator
        gains_a = np.sum(np.maximum(returns_a, 0))
        losses_a = np.sum(np.maximum(-returns_a, 0))
        rs_a = gains_a / losses_a if losses_a > 0 else 1
        rsi_a = 100 - (100 / (1 + rs_a))
        
        # Position in range
        rolling_min_a = np.min(window_a[-20:])
        rolling_max_a = np.max(window_a[-20:])
        range_a = rolling_max_a - rolling_min_a
        position_a = (window_a[-1] - rolling_min_a) / range_a if range_a > 0 else 0.5
        
        # ========== Asset B Features ==========
        returns_b = np.diff(window_b) / window_b[:-1]
        
        # Lagged returns
        lag_1_b = returns_b[-1] if len(returns_b) > 0 else 0
        lag_5_b = np.mean(returns_b[-5:]) if len(returns_b) >= 5 else 0
        lag_10_b = np.mean(returns_b[-10:]) if len(returns_b) >= 10 else 0
        
        # Moving averages
        sma_10_b = np.mean(window_b[-10:])
        sma_20_b = np.mean(window_b[-20:]) if len(window_b) >= 20 else np.mean(window_b)
        price_vs_sma_b = (window_b[-1] - sma_20_b) / sma_20_b if sma_20_b != 0 else 0
        
        # Momentum
        momentum_b = (window_b[-1] - window_b[0]) / window_b[0] if window_b[0] != 0 else 0
        
        # Volatility
        volatility_b = np.std(returns_b) if len(returns_b) > 0 else 0
        
        # RSI-like indicator
        gains_b = np.sum(np.maximum(returns_b, 0))
        losses_b = np.sum(np.maximum(-returns_b, 0))
        rs_b = gains_b / losses_b if losses_b > 0 else 1
        rsi_b = 100 - (100 / (1 + rs_b))
        
        # Position in range
        rolling_min_b = np.min(window_b[-20:])
        rolling_max_b = np.max(window_b[-20:])
        range_b = rolling_max_b - rolling_min_b
        position_b = (window_b[-1] - rolling_min_b) / range_b if range_b > 0 else 0.5
        
        # ========== Correlation Features ==========
        correlation = np.corrcoef(window_a, window_b)[0, 1] if len(window_a) >= 2 else 0
        correlation = correlation if not np.isnan(correlation) else 0
        
        # Relative momentum
        relative_momentum = momentum_a - momentum_b
        
        # Feature vector
        features = [
            lag_1_a, lag_5_a, lag_10_a,
            sma_10_a, sma_20_a, price_vs_sma_a,
            momentum_a, volatility_a, rsi_a, position_a,
            
            lag_1_b, lag_5_b, lag_10_b,
            sma_10_b, sma_20_b, price_vs_sma_b,
            momentum_b, volatility_b, rsi_b, position_b,
            
            correlation, relative_momentum
        ]
        
        X.append(features)
        
        # ========== Target: Optimal Allocations ==========
        # Use simple heuristic: allocate based on momentum
        alloc_a = 0.5 + np.clip(momentum_a / 0.1, -0.25, 0.25)
        alloc_b = 0.5 + np.clip(momentum_b / 0.1, -0.25, 0.25)
        
        # Normalize to sum with cash
        total = alloc_a + alloc_b
        cash = 0.3  # Base cash allocation
        
        if total > 0:
            alloc_a = (alloc_a / total) * (1 - cash)
            alloc_b = (alloc_b / total) * (1 - cash)
        else:
            alloc_a = (1 - cash) / 2
            alloc_b = (1 - cash) / 2
        
        # Ensure valid allocations
        alloc_a = np.clip(alloc_a, 0.05, 0.95)
        alloc_b = np.clip(alloc_b, 0.05, 0.95)
        alloc_c = 1.0 - alloc_a - alloc_b
        alloc_c = np.clip(alloc_c, 0.05, 0.95)
        
        # Normalize
        total = alloc_a + alloc_b + alloc_c
        y.append([alloc_a / total, alloc_b / total, alloc_c / total])
    
    return np.array(X), np.array(y)

def load_multi_asset_data(data_paths):
    """Load and combine multi-asset training data"""
    all_data_a = []
    all_data_b = []
    
    for path in data_paths:
        if os.path.exists(path):
            print(f"  Loading: {path}")
            df = pd.read_csv(path, index_col=0)
            
            if 'Asset A' in df.columns and 'Asset B' in df.columns:
                all_data_a.append(df['Asset A'].values)
                all_data_b.append(df['Asset B'].values)
                print(f"    Loaded {len(df)} data points")
            else:
                print(f"    ⚠ Missing Asset A or Asset B in {path}")
    
    if len(all_data_a) == 0:
        return None, None
    
    data_a = np.concatenate(all_data_a)
    data_b = np.concatenate(all_data_b)
    
    print(f"\n✓ Total combined data points: {len(data_a)}")
    print(f"  Asset A range: {data_a.min():.4f} - {data_a.max():.4f}")
    print(f"  Asset B range: {data_b.min():.4f} - {data_b.max():.4f}")
    
    return data_a, data_b

def train_neural_network(data_paths, lookback=30, hidden_layers=(64, 32)):
    """Train neural network for portfolio allocation"""
    
    print("="*70)
    print("🧠 Neural Network Training for Multi-Asset Portfolio Allocation")
    print("="*70)
    
    # Load data
    print("\n📊 Loading training data...")
    data_a, data_b = load_multi_asset_data(data_paths)
    
    if data_a is None:
        print("Error: No training data could be loaded!")
        return None, None, None, None
    
    # Create features
    print(f"\n🔧 Creating features (lookback={lookback})...")
    X, y = create_features(data_a, data_b, lookback=lookback)
    print(f"   Feature matrix shape: {X.shape}")
    print(f"   Target matrix shape: {y.shape}")
    
    # Normalize features
    print(f"\n⚖️  Normalizing features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, shuffle=True
    )
    
    print(f"   Training samples: {len(X_train)}")
    print(f"   Test samples: {len(X_test)}")
    
    # Train neural network
    print(f"\n🚀 Training Neural Network...")
    print(f"   Architecture: {X_scaled.shape[1]} -> {' -> '.join(map(str, hidden_layers))} -> 3")
    
    model = MLPRegressor(
        hidden_layer_sizes=hidden_layers,
        activation='relu',
        solver='adam',
        learning_rate='adaptive',
        learning_rate_init=0.001,
        max_iter=500,
        batch_size=32,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=20,
        verbose=False
    )
    
    model.fit(X_train, y_train)
    
    # Evaluate
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    train_mse = mean_squared_error(y_train, y_pred_train)
    test_mse = mean_squared_error(y_test, y_pred_test)
    test_mae = mean_absolute_error(y_test, y_pred_test)
    
    print(f"\n📈 Model Performance:")
    print(f"   Training R² Score: {train_score:.4f}")
    print(f"   Test R² Score: {test_score:.4f}")
    print(f"   Training MSE: {train_mse:.6f}")
    print(f"   Test MSE: {test_mse:.6f}")
    print(f"   Test MAE: {test_mae:.6f}")
    
    # Analyze prediction quality
    print(f"\n🎯 Allocation Predictions Analysis:")
    for i, asset in enumerate(['Asset A', 'Asset B', 'Cash']):
        pred_range = (y_pred_test[:, i].min(), y_pred_test[:, i].max())
        pred_mean = y_pred_test[:, i].mean()
        print(f"   {asset}: range [{pred_range[0]:.3f}, {pred_range[1]:.3f}], mean {pred_mean:.3f}")
    
    return model, scaler, X.shape[1], test_score

def save_model(model, scaler, n_features, lookback=30):
    """Save trained model and scaler"""
    model_data = {
        'model': model,
        'scaler': scaler,
        'n_features': n_features,
        'lookback': lookback,
        'model_type': 'MLPRegressor',
        'hidden_layers': model.hidden_layer_sizes,
        'activation': model.activation
    }
    
    model_path = 'nn_allocation_model.pkl'
    joblib.dump(model_data, model_path)
    
    print(f"\n💾 Model saved to {model_path}")
    print(f"   Features: {n_features}")
    print(f"   Lookback: {lookback}")
    print(f"   Hidden layers: {model.hidden_layer_sizes}")
    
    return model_path

def main():
    """Main training script"""
    
    # Training parameters
    lookback = 30
    hidden_layers = (64, 32, 16)  # 3-layer network
    
    # Training data paths
    train_data_paths = [
        'data/asset_a_b_train_1.csv',
        'data/asset_a_b_train_2.csv'
    ]
    
    # Train model
    model, scaler, n_features, test_score = train_neural_network(
        train_data_paths,
        lookback=lookback,
        hidden_layers=hidden_layers
    )
    
    if model is None:
        print("\n❌ Training failed!")
        return
    
    # Save model
    model_path = save_model(model, scaler, n_features, lookback)
    
    print("\n" + "="*70)
    print("✅ Training completed successfully!")
    print("="*70)
    print(f"\nModel is ready for bot_trade.py")
    print(f"Run: python main.py data/asset_a_b_train_1.csv")

if __name__ == "__main__":
    main()
