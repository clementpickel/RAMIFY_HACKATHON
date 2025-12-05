import numpy as np
import pandas as pd
import pickle
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error
import lightgbm as lgb
import matplotlib.pyplot as plt

def create_features(data, lookback=60):
    """
    Create comprehensive features for LightGBM from price data
    """
    df = pd.DataFrame(data, columns=['price'])
    
    # Lagged features (previous prices)
    for i in [1, 2, 3, 5, 7, 10, 15, 20, 30]:
        df[f'lag_{i}'] = df['price'].shift(i)
    
    # Moving averages
    for window in [3, 5, 7, 10, 15, 20, 30, 40, 50]:
        df[f'ma_{window}'] = df['price'].rolling(window=window).mean()
        df[f'std_{window}'] = df['price'].rolling(window=window).std()
    
    # Exponential moving averages
    for span in [5, 10, 20, 30]:
        df[f'ema_{span}'] = df['price'].ewm(span=span, adjust=False).mean()
    
    # Price relative to moving averages
    for window in [5, 10, 20, 30]:
        df[f'price_vs_ma{window}'] = (df['price'] - df[f'ma_{window}']) / df[f'ma_{window}']
    
    # Momentum features
    for period in [3, 5, 10, 20, 30]:
        df[f'momentum_{period}'] = df['price'] - df['price'].shift(period)
        df[f'roc_{period}'] = (df['price'] - df['price'].shift(period)) / df['price'].shift(period)
    
    # Volatility features
    for window in [5, 10, 20, 30]:
        df[f'volatility_{window}'] = df['price'].rolling(window=window).std()
        df[f'volatility_ratio_{window}'] = df[f'volatility_{window}'] / df['price']
    
    # High-low range
    for window in [5, 10, 20, 30]:
        df[f'range_{window}'] = df['price'].rolling(window=window).max() - df['price'].rolling(window=window).min()
        df[f'range_ratio_{window}'] = df[f'range_{window}'] / df['price']
    
    # Position in range
    for window in [10, 20, 30]:
        rolling_min = df['price'].rolling(window).min()
        rolling_max = df['price'].rolling(window).max()
        range_diff = rolling_max - rolling_min
        range_diff = range_diff.replace(0, 1)
        df[f'position_in_range_{window}'] = (df['price'] - rolling_min) / range_diff
    
    # Rate of change of moving averages
    for window in [5, 10, 20]:
        df[f'ma_roc_{window}'] = (df[f'ma_{window}'] - df[f'ma_{window}'].shift(5)) / df[f'ma_{window}'].shift(5)
    
    # Acceleration (second derivative)
    df['acceleration_5'] = df['momentum_5'] - df['momentum_5'].shift(5)
    df['acceleration_10'] = df['momentum_10'] - df['momentum_10'].shift(10)
    
    # Cross-overs (bullish/bearish signals)
    df['ma5_vs_ma20'] = df['ma_5'] - df['ma_20']
    df['ma10_vs_ma30'] = df['ma_10'] - df['ma_30']
    df['ema5_vs_ema20'] = df['ema_5'] - df['ema_20']
    
    # Bollinger Bands
    for window in [20, 30]:
        ma = df[f'ma_{window}']
        std = df[f'std_{window}']
        df[f'bb_upper_{window}'] = ma + 2 * std
        df[f'bb_lower_{window}'] = ma - 2 * std
        df[f'bb_position_{window}'] = (df['price'] - df[f'bb_lower_{window}']) / (df[f'bb_upper_{window}'] - df[f'bb_lower_{window}'])
    
    # Target: next price
    df['target'] = df['price'].shift(-1)
    
    # Drop rows with NaN values
    df = df.dropna()
    
    return df

def prepare_data(data, lookback=60):
    """Prepare features and target for training"""
    df = create_features(data, lookback)
    
    # Separate features and target
    X = df.drop(['price', 'target'], axis=1)
    y = df['target']
    
    feature_names = X.columns.tolist()
    
    return X.values, y.values, feature_names

def train_lightgbm_model(train_data_path, lookback=60):
    """Train LightGBM model with hyperparameter tuning"""
    
    # Load training data
    print(f"Loading training data from: {train_data_path}")
    df = pd.read_csv(train_data_path, index_col=0)
    data = df['Asset A'].values
    
    print(f"Loaded {len(data)} data points")
    print(f"Price range: {data.min():.2f} to {data.max():.2f}")
    
    # Create features
    print("\nCreating features...")
    X, y, feature_names = prepare_data(data, lookback)
    print(f"Created {X.shape[1]} features from {X.shape[0]} samples")
    
    # Standardize features (optional for LightGBM but can help)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Time series split for validation
    tscv = TimeSeriesSplit(n_splits=5)
    
    print("\n" + "="*70)
    print("Training LightGBM Model with Cross-Validation...")
    print("="*70)
    
    # LightGBM parameters
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1,
        'min_child_samples': 20,
        'reg_alpha': 0.1,
        'reg_lambda': 0.1
    }
    
    fold_scores = []
    models = []
    
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X_scaled)):
        print(f"\nFold {fold + 1}/5:")
        
        X_train_fold = X_scaled[train_idx]
        y_train_fold = y[train_idx]
        X_val_fold = X_scaled[val_idx]
        y_val_fold = y[val_idx]
        
        # Create LightGBM datasets
        train_data = lgb.Dataset(X_train_fold, label=y_train_fold)
        val_data = lgb.Dataset(X_val_fold, label=y_val_fold, reference=train_data)
        
        # Train model
        model = lgb.train(
            params,
            train_data,
            num_boost_round=500,
            valid_sets=[train_data, val_data],
            valid_names=['train', 'valid'],
            callbacks=[
                lgb.early_stopping(stopping_rounds=50),
                lgb.log_evaluation(period=0)  # Silent training
            ]
        )
        
        # Evaluate
        y_pred = model.predict(X_val_fold, num_iteration=model.best_iteration)
        rmse = np.sqrt(mean_squared_error(y_val_fold, y_pred))
        mae = mean_absolute_error(y_val_fold, y_pred)
        
        fold_scores.append(rmse)
        models.append(model)
        
        print(f"  Train samples: {len(X_train_fold)}")
        print(f"  Val samples: {len(X_val_fold)}")
        print(f"  Best iteration: {model.best_iteration}")
        print(f"  Validation RMSE: {rmse:.4f}")
        print(f"  Validation MAE: {mae:.4f}")
    
    avg_rmse = np.mean(fold_scores)
    print("\n" + "="*70)
    print(f"✓ Cross-Validation Complete")
    print(f"  Average RMSE: {avg_rmse:.4f}")
    print(f"  Std RMSE: {np.std(fold_scores):.4f}")
    print("="*70)
    
    # Train final model on all data
    print("\nTraining final model on all data...")
    train_data = lgb.Dataset(X_scaled, label=y)
    
    final_model = lgb.train(
        params,
        train_data,
        num_boost_round=500,
        valid_sets=[train_data],
        valid_names=['train'],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50),
            lgb.log_evaluation(period=100)
        ]
    )
    
    print(f"✓ Final model trained with {final_model.best_iteration} iterations")
    
    return final_model, scaler, feature_names

def evaluate_model(model, scaler, feature_names, test_data_path, lookback=60):
    """Evaluate model on test data"""
    print(f"\n{'='*70}")
    print(f"Evaluating model on test data: {test_data_path}")
    print('='*70)
    
    df = pd.read_csv(test_data_path, index_col=0)
    data = df['Asset A'].values
    
    X, y, _ = prepare_data(data, lookback)
    X_scaled = scaler.transform(X)
    
    # Make predictions
    y_pred = model.predict(X_scaled, num_iteration=model.best_iteration)
    
    # Calculate metrics
    mse = mean_squared_error(y, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y, y_pred)
    mape = np.mean(np.abs((y - y_pred) / y)) * 100
    
    print(f"\nTest Performance:")
    print(f"  MSE:  {mse:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  MAE:  {mae:.4f}")
    print(f"  MAPE: {mape:.2f}%")
    
    # Calculate directional accuracy
    actual_direction = np.sign(y[1:] - y[:-1])
    pred_direction = np.sign(y_pred[1:] - y[:-1])
    directional_accuracy = np.mean(actual_direction == pred_direction)
    print(f"  Directional Accuracy: {directional_accuracy*100:.2f}%")
    
    # R-squared
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    print(f"  R² Score: {r2:.4f}")
    
    return y_pred, y

def plot_feature_importance(model, feature_names, top_n=25):
    """Plot feature importance"""
    importance = model.feature_importance(importance_type='gain')
    feature_importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    }).sort_values('importance', ascending=False)
    
    # Plot top N features
    top_features = feature_importance_df.head(top_n)
    
    plt.figure(figsize=(10, 8))
    plt.barh(range(len(top_features)), top_features['importance'])
    plt.yticks(range(len(top_features)), top_features['feature'])
    plt.xlabel('Importance (Gain)')
    plt.title(f'Top {top_n} Feature Importances (LightGBM)')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig('lightgbm_feature_importance.png', dpi=300)
    print(f"\nFeature importance plot saved as 'lightgbm_feature_importance.png'")
    
    # Print top 10
    print(f"\nTop 10 Most Important Features:")
    for idx, row in feature_importance_df.head(10).iterrows():
        print(f"  {row['feature']:30s}: {row['importance']:>10.1f}")

def plot_predictions(predictions, actual, num_points=300):
    """Plot predictions vs actual values"""
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # Plot 1: Full comparison (last num_points)
    start_idx = max(0, len(predictions) - num_points)
    time_steps = range(len(predictions[start_idx:]))
    
    axes[0].plot(time_steps, actual[start_idx:], label='Actual Price', linewidth=2, alpha=0.8)
    axes[0].plot(time_steps, predictions[start_idx:], label='Predicted Price', 
                 linewidth=2, alpha=0.7, linestyle='--')
    axes[0].set_xlabel('Time Step')
    axes[0].set_ylabel('Price')
    axes[0].set_title(f'LightGBM Predictions vs Actual (Last {num_points} points)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Prediction errors
    errors = predictions - actual
    axes[1].plot(time_steps, errors[start_idx:], label='Prediction Error', 
                 linewidth=1, alpha=0.7, color='red')
    axes[1].axhline(y=0, color='black', linestyle='--', linewidth=1)
    axes[1].fill_between(time_steps, 0, errors[start_idx:], alpha=0.3, color='red')
    axes[1].set_xlabel('Time Step')
    axes[1].set_ylabel('Error (Predicted - Actual)')
    axes[1].set_title('Prediction Errors Over Time')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('lightgbm_predictions.png', dpi=300)
    print("Predictions plot saved as 'lightgbm_predictions.png'")

def save_model(model, scaler, feature_names, lookback=60):
    """Save model and scaler"""
    model_data = {
        'model': model,
        'scaler': scaler,
        'feature_names': feature_names,
        'model_type': 'LightGBM',
        'lookback': lookback,
        'num_features': len(feature_names),
        'best_iteration': model.best_iteration
    }
    
    model_path = 'lightgbm_model.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump(model_data, f)
    
    print(f"\n✓ Model saved to {model_path}")
    print(f"  Model type: LightGBM")
    print(f"  Features: {len(feature_names)}")
    print(f"  Best iteration: {model.best_iteration}")
    print(f"  Lookback: {lookback}")

def main():
    """Main training script"""
    # Set paths
    train_data_path = 'data/asset_a_train.csv'
    test_data_path = 'data/asset_a_test.csv'
    
    # Try alternative paths if primary doesn't exist
    if not os.path.exists(train_data_path):
        train_data_path = 'data/asset_train.csv'
    
    if not os.path.exists(test_data_path):
        test_data_path = 'data/asset_test.csv'
    
    if not os.path.exists(train_data_path):
        print(f"Error: Training data not found!")
        print(f"Tried: data/asset_a_train.csv and data/asset_train.csv")
        print(f"Please ensure your training data is in the 'data' folder")
        return
    
    print("="*70)
    print("LightGBM Trading Bot - Training Script")
    print("="*70)
    
    # Training parameters
    lookback = 60
    
    # Train model
    model, scaler, feature_names = train_lightgbm_model(train_data_path, lookback)
    
    # Plot feature importance
    plot_feature_importance(model, feature_names)
    
    # Evaluate on test data if available
    if os.path.exists(test_data_path):
        predictions, actual = evaluate_model(model, scaler, feature_names, test_data_path, lookback)
        plot_predictions(predictions, actual)
    else:
        print(f"\nTest data not found at {test_data_path}, skipping evaluation")
    
    # Save model
    save_model(model, scaler, feature_names, lookback)
    
    print("\n" + "="*70)
    print("Training completed successfully!")
    print("="*70)
    print("\nYou can now use bot_trade.py with the trained LightGBM model.")
    print("Run: python3 main.py data/asset_a_test.csv --show-graph")

if __name__ == "__main__":
    main()