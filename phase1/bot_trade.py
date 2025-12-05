import numpy as np
import pandas as pd
import joblib
import os

# Global variables
history = []
model = None
scaler = None
feature_names = None
lookback = 30
model_loaded = False

def load_trained_model():
    """Load pre-trained logistic regression model"""
    global model, scaler, feature_names, lookback, model_loaded
    
    current_dir = os.path.dirname(__file__) if os.path.dirname(__file__) else '.'
    model_path = os.path.join(current_dir, 'logistic_model.pkl')
    
    try:
        if os.path.exists(model_path):
            model_data = joblib.load(model_path)
            
            model = model_data['model']
            scaler = model_data['scaler']
            feature_names = model_data['feature_names']
            lookback = model_data['lookback']
            
            # print("✓ Logistic Regression model loaded successfully")
            # print(f"  Features: {len(feature_names)}")
            # print(f"  Lookback: {lookback}")
            model_loaded = True
            return True
        else:
            # print(f"✗ Model file not found at {model_path}")
            # print("  Please run: python3 train_logistic_model.py")
            return False
    
    except Exception as e:
        # print(f"✗ Error loading model: {e}")
        return False

def create_features_from_history(price_history, lookback=30):
    """
    Create features from price history
    Must match the features used during training
    """
    if len(price_history) < lookback + 20:  # Need extra for moving averages
        return None
    
    df = pd.DataFrame(price_history, columns=['price'])
    
    # Lagged features (previous prices)
    for i in range(1, lookback + 1):
        df[f'lag_{i}'] = df['price'].shift(i)
    
    # Moving averages
    for window in [3, 5, 10, 20]:
        df[f'ma_{window}'] = df['price'].rolling(window=window).mean()
    
    # Price relative to moving averages (signals)
    df['price_vs_ma3'] = (df['price'] - df['ma_3']) / df['ma_3']
    df['price_vs_ma5'] = (df['price'] - df['ma_5']) / df['ma_5']
    df['price_vs_ma10'] = (df['price'] - df['ma_10']) / df['ma_10']
    df['price_vs_ma20'] = (df['price'] - df['ma_20']) / df['ma_20']
    
    # Momentum features
    df['momentum_3'] = df['price'] - df['price'].shift(3)
    df['momentum_5'] = df['price'] - df['price'].shift(5)
    df['momentum_10'] = df['price'] - df['price'].shift(10)
    
    # Rate of change (percentage)
    df['roc_3'] = (df['price'] - df['price'].shift(3)) / df['price'].shift(3)
    df['roc_5'] = (df['price'] - df['price'].shift(5)) / df['price'].shift(5)
    df['roc_10'] = (df['price'] - df['price'].shift(10)) / df['price'].shift(10)
    
    # Volatility (rolling standard deviation)
    df['volatility_5'] = df['price'].rolling(window=5).std()
    df['volatility_10'] = df['price'].rolling(window=10).std()
    
    # High-low range
    df['range_5'] = df['price'].rolling(window=5).max() - df['price'].rolling(window=5).min()
    df['range_10'] = df['price'].rolling(window=10).max() - df['price'].rolling(window=10).min()
    
    # Position in range (0 = at min, 1 = at max)
    rolling_min = df['price'].rolling(20).min()
    rolling_max = df['price'].rolling(20).max()
    range_diff = rolling_max - rolling_min
    range_diff = range_diff.replace(0, 1)  # Avoid division by zero
    df['position_in_range'] = (df['price'] - rolling_min) / range_diff
    
    # Get last row (most recent features)
    df = df.dropna()
    
    if len(df) == 0:
        return None
    
    # Get the last row and drop 'price' column
    features = df.iloc[-1:].drop('price', axis=1)
    
    return features.values

def predict_direction(price_history):
    """
    Predict price direction using trained logistic regression
    Returns: (direction, probability)
        direction: 1 = UP, 0 = DOWN
        probability: confidence in the prediction (0-1)
    """
    global model, scaler, lookback
    
    if model is None or scaler is None:
        return None, None
    
    if len(price_history) < lookback + 20:
        return None, None
    
    try:
        # Create features
        features = create_features_from_history(price_history, lookback)
        
        if features is None:
            return None, None
        
        # Scale features
        features_scaled = scaler.transform(features)
        
        # Make prediction
        direction = model.predict(features_scaled)[0]
        probability = model.predict_proba(features_scaled)[0]
        
        # probability[0] = prob of DOWN, probability[1] = prob of UP
        confidence = probability[1] if direction == 1 else probability[0]
        
        return int(direction), float(confidence)
    
    except Exception as e:
        # print(f"Error in prediction: {e}")
        return None, None

def calculate_allocation(direction, confidence, current_price, price_history):
    """
    Calculate portfolio allocation based on prediction
    
    Parameters:
    - direction: 1 (UP) or 0 (DOWN)
    - confidence: probability (0.5 to 1.0)
    - current_price: current asset price
    - price_history: historical prices
    
    Returns:
    - asset_allocation: percentage to allocate to Asset A
    """
    
    # Base allocation
    if direction == 1:  # Predict UP
        # Start at 50%, increase based on confidence
        # Confidence ranges from 0.5 (neutral) to 1.0 (very confident)
        confidence_boost = (confidence - 0.5) / 0.5  # Normalize to 0-1
        
        # Allocate 50% + up to 35% more based on confidence
        base_allocation = 0.50
        max_additional = 0.35
        asset_allocation = base_allocation + (confidence_boost * max_additional)
        
        # Cap at 85%
        asset_allocation = min(asset_allocation, 0.85)
        
    else:  # Predict DOWN
        # Start at 50%, decrease based on confidence
        confidence_boost = (confidence - 0.5) / 0.5  # Normalize to 0-1
        
        # Allocate 50% - up to 35% less based on confidence
        base_allocation = 0.50
        max_reduction = 0.35
        asset_allocation = base_allocation - (confidence_boost * max_reduction)
        
        # Cap at 15%
        asset_allocation = max(asset_allocation, 0.15)
    
    # Add slight adjustment based on recent momentum
    if len(price_history) >= 5:
        recent_momentum = (price_history[-1] - price_history[-5]) / price_history[-5]
        momentum_adjustment = np.clip(recent_momentum * 5, -0.05, 0.05)
        asset_allocation += momentum_adjustment
    
    # Final bounds
    asset_allocation = np.clip(asset_allocation, 0.15, 0.85)
    
    return asset_allocation

def make_decision(epoch: int, price: float):
    """
    Make trading decision based on logistic regression predictions
    
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
        # print("\nLoading trained logistic regression model...")
        success = load_trained_model()
        if not success:
            pass
            # print("⚠ Model not loaded, using neutral strategy (50/50)")
    
    # If model couldn't be loaded, use neutral strategy
    if model is None:
        return {'Asset A': 0.5, 'Cash': 0.5}
    
    # Need minimum history for prediction
    min_history = lookback + 20
    if len(history) < min_history:
        # Gradually build up position as we accumulate history
        progress = len(history) / min_history
        return {'Asset A': 0.4 + 0.1 * progress, 'Cash': 0.6 - 0.1 * progress}
    
    try:
        # Get prediction
        direction, confidence = predict_direction(history)
        
        if direction is None or confidence is None:
            return {'Asset A': 0.5, 'Cash': 0.5}
        
        # Calculate allocation
        asset_allocation = calculate_allocation(direction, confidence, price, history)
        
        cash_allocation = 1.0 - asset_allocation
        
        return {
            'Asset A': float(asset_allocation),
            'Cash': float(cash_allocation)
        }
    
    except Exception as e:
        # print(f"Error in decision making at epoch {epoch}: {e}")
        return {'Asset A': 0.5, 'Cash': 0.5}