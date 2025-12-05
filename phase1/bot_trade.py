import numpy as np
import pandas as pd
import pickle
import os

# Global variables
history = []
model = None
scaler = None
feature_names = None
lookback = 60
model_loaded = False

def load_trained_model():
    """Load pre-trained LightGBM model"""
    global model, scaler, feature_names, lookback, model_loaded
    
    current_dir = os.path.dirname(__file__) if os.path.dirname(__file__) else '.'
    model_path = os.path.join(current_dir, 'lightgbm_model.pkl')
    
    try:
        if os.path.exists(model_path):
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
            
            model = model_data['model']
            scaler = model_data['scaler']
            feature_names = model_data['feature_names']
            lookback = model_data['lookback']
            
            print("✓ LightGBM model loaded successfully")
            print(f"  Features: {model_data['num_features']}")
            print(f"  Best iteration: {model_data['best_iteration']}")
            print(f"  Lookback: {lookback}")
            model_loaded = True
            return True
        else:
            print(f"✗ Model file not found at {model_path}")
            print("  Please run: python3 train_lightgbm_model.py")
            return False
    
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        return False

def create_features_from_history(price_history, lookback=60):
    """
    Create features from price history
    Must match the features used during training
    """
    if len(price_history) < lookback + 50:  # Need extra for moving averages
        return None
    
    df = pd.DataFrame(price_history, columns=['price'])
    
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
    
    # Get last row (most recent features)
    df = df.dropna()
    
    if len(df) == 0:
        return None
    
    # Get the last row and drop 'price' column
    features = df.iloc[-1:].drop('price', axis=1)
    
    return features.values

def predict_next_price(price_history):
    """Predict next price using trained LightGBM model"""
    global model, scaler, lookback
    
    if model is None or scaler is None:
        return None
    
    if len(price_history) < lookback + 50:
        return None
    
    try:
        # Create features
        features = create_features_from_history(price_history, lookback)
        
        if features is None:
            return None
        
        # Scale features
        features_scaled = scaler.transform(features)
        
        # Make prediction
        predicted_price = model.predict(features_scaled, num_iteration=model.best_iteration)[0]
        
        return float(predicted_price)
    
    except Exception as e:
        print(f"Error in prediction: {e}")
        return None

def calculate_confidence(price_history, predicted_price, current_price):
    """Calculate confidence based on prediction strength and recent patterns"""
    if len(price_history) < 30:
        return 0.5
    
    recent_prices = np.array(price_history[-30:])
    
    # 1. Volatility-based confidence (lower volatility = higher confidence)
    volatility = np.std(recent_prices)
    mean_price = np.mean(recent_prices)
    relative_volatility = volatility / mean_price if mean_price != 0 else 1.0
    volatility_confidence = max(0, 1 - relative_volatility * 5)
    
    # 2. Trend strength (consistent direction = higher confidence)
    changes = np.diff(recent_prices)
    positive_changes = np.sum(changes > 0)
    negative_changes = np.sum(changes < 0)
    trend_consistency = abs(positive_changes - negative_changes) / len(changes)
    
    # 3. Prediction magnitude (stronger signal = higher confidence, but cap it)
    price_change_pct = abs(predicted_price - current_price) / current_price if current_price != 0 else 0
    magnitude_confidence = min(price_change_pct * 50, 1.0)  # Scale and cap at 1.0
    
    # Combine factors
    confidence = (
        0.4 * volatility_confidence +
        0.3 * trend_consistency +
        0.3 * magnitude_confidence
    )
    
    # Ensure confidence is between 0.3 and 0.85
    return np.clip(confidence, 0.3, 0.85)

def make_decision(epoch: int, price: float):
    """
    Make trading decision based on LightGBM predictions
    
    Parameters
    ----------
    epoch : int
        Current time step
    price : float
        Current price of Asset A
    
    Returns
    -------
    dict
        Portfolio allocation: {'Asset A': float, 'Cash': float}
        Values must sum to 1.0
    """
    global model, model_loaded, history
    
    # Store price history
    history.append(price)
    
    # Load model on first call if not already loaded
    if not model_loaded:
        print("\nLoading trained LightGBM model...")
        success = load_trained_model()
        if not success:
            print("⚠ Model not loaded, using neutral strategy (50/50)")
    
    # If model couldn't be loaded, use neutral strategy
    if model is None:
        return {'Asset A': 0.5, 'Cash': 0.5}
    
    # Need minimum history for prediction
    min_history = lookback + 50
    if len(history) < min_history:
        # Gradually build up position as we accumulate history
        progress = len(history) / min_history
        return {'Asset A': 0.35 + 0.15 * progress, 'Cash': 0.65 - 0.15 * progress}
    
    try:
        # Get prediction
        predicted_price = predict_next_price(history)
        
        if predicted_price is None:
            return {'Asset A': 0.5, 'Cash': 0.5}
        
        current_price = history[-1]
        
        # Calculate expected price change
        price_change = predicted_price - current_price
        price_change_pct = price_change / current_price if current_price != 0 else 0
        
        # Calculate confidence
        confidence = calculate_confidence(history, predicted_price, current_price)
        
        # Trading strategy with adaptive thresholds
        # Use smaller threshold to be more reactive
        threshold = 0.0015  # 0.15% threshold
        
        if price_change_pct > threshold:
            # Predicted increase - allocate more to Asset A
            # Scale allocation based on prediction strength and confidence
            strength = min(abs(price_change_pct) / 0.01, 1.0)  # Normalize to 0-1
            
            # Base allocation + confidence boost + strength boost
            base = 0.50
            confidence_boost = confidence * 0.25
            strength_boost = strength * 0.15
            
            asset_allocation = base + confidence_boost + strength_boost
            asset_allocation = np.clip(asset_allocation, 0.55, 0.90)
            
            return {'Asset A': asset_allocation, 'Cash': 1.0 - asset_allocation}
        
        elif price_change_pct < -threshold:
            # Predicted decrease - allocate more to Cash
            strength = min(abs(price_change_pct) / 0.01, 1.0)
            
            base = 0.50
            confidence_reduction = confidence * 0.25
            strength_reduction = strength * 0.15
            
            asset_allocation = base - confidence_reduction - strength_reduction
            asset_allocation = np.clip(asset_allocation, 0.10, 0.45)
            
            return {'Asset A': asset_allocation, 'Cash': 1.0 - asset_allocation}
        
        else:
            # Neutral prediction - balanced allocation with slight momentum bias
            if len(history) >= 10:
                # Add slight bias based on recent momentum
                recent_momentum = (history[-1] - history[-10]) / history[-10]
                momentum_bias = np.clip(recent_momentum * 5, -0.05, 0.05)
                asset_allocation = 0.50 + momentum_bias
            else:
                asset_allocation = 0.50
            
            return {'Asset A': asset_allocation, 'Cash': 1.0 - asset_allocation}
    
    except Exception as e:
        print(f"Error in decision making at epoch {epoch}: {e}")
        return {'Asset A': 0.5, 'Cash': 0.5}