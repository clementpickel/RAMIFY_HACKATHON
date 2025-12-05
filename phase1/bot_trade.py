import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import pickle
import os

# Global variables
history = []
model = None
scaler = None
lookback = 60
predictions_history = []
model_loaded = False

def load_trained_model():
    """Load pre-trained model and scaler from pickle files"""
    global model, scaler, model_loaded
    
    current_dir = os.path.dirname(__file__) if os.path.dirname(__file__) else '.'
    model_h5_path = os.path.join(current_dir, 'model_lstm.h5')
    scaler_path = os.path.join(current_dir, 'scaler.pkl')
    
    try:
        # Load scaler
        if os.path.exists(scaler_path):
            with open(scaler_path, 'rb') as f:
                scaler = pickle.load(f)
            print("✓ Scaler loaded successfully")
        else:
            print(f"✗ Scaler file not found at {scaler_path}")
            print("  Please run: python3 train_model.py")
            return False
        
        # Load model with compile=False to avoid metric issues, then recompile
        if os.path.exists(model_h5_path):
            from tensorflow.keras.optimizers import Adam
            model = load_model(model_h5_path, compile=False)
            # Recompile with proper loss function
            model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
            print("✓ Model loaded successfully")
            print(f"  Model architecture: {len(model.layers)} layers")
            model_loaded = True
            return True
        else:
            print(f"✗ Model file not found at {model_h5_path}")
            print("  Please run: python3 train_model.py")
            return False
    
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        return False

def predict_next_price(price_history, lookback=60):
    """Predict next price using trained LSTM"""
    global scaler, model
    
    if model is None or scaler is None:
        return None
    
    if len(price_history) < lookback:
        return None  # Insufficient history
    
    try:
        # Prepare recent data
        recent_data = np.array(price_history[-lookback:]).reshape(-1, 1)
        normalized = scaler.transform(recent_data)
        
        # Reshape for LSTM [samples, timesteps, features]
        X = normalized.reshape((1, lookback, 1))
        
        # Make prediction
        predicted_normalized = model.predict(X, verbose=0)[0][0]
        
        # Denormalize
        predicted_price = scaler.inverse_transform([[predicted_normalized]])[0][0]
        
        return predicted_price
    except Exception as e:
        print(f"Error in prediction: {e}")
        return None

def calculate_confidence(price_history, predicted_price, current_price):
    """Calculate confidence in prediction based on recent volatility"""
    if len(price_history) < 20:
        return 0.5
    
    recent_prices = price_history[-20:]
    volatility = np.std(recent_prices)
    mean_price = np.mean(recent_prices)
    
    # Normalize volatility
    relative_volatility = volatility / mean_price if mean_price != 0 else 1.0
    
    # Price change magnitude
    price_change_pct = abs(predicted_price - current_price) / current_price if current_price != 0 else 0
    
    # Higher volatility -> lower confidence
    # Larger predicted change -> adjust confidence
    confidence = 0.5 + (0.3 * (1 - min(relative_volatility * 10, 1.0)))
    
    # Boost confidence slightly if prediction is strong
    if price_change_pct > 0.01:
        confidence += 0.1
    
    return min(max(confidence, 0.3), 0.8)

def make_decision(epoch: int, price: float):
    """
    Make trading decision based on LSTM predictions
    
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
    global model, predictions_history, model_loaded, history
    
    # Store price history
    history.append({"epoch": epoch, "price": price})
    price_values = [h["price"] for h in history]
    
    # Load model on first call if not already loaded
    if not model_loaded:
        print("\nLoading trained LSTM model...")
        success = load_trained_model()
        if not success:
            print("⚠ Model not loaded, using neutral strategy (50/50)")
    
    # If model couldn't be loaded, use neutral strategy
    if model is None or scaler is None:
        return {'Asset A': 0.5, 'Cash': 0.5}
    
    # Need minimum history for prediction
    if len(price_values) < lookback:
        # Gradually build up position as we accumulate history
        progress = len(price_values) / lookback
        return {'Asset A': 0.3 + 0.2 * progress, 'Cash': 0.7 - 0.2 * progress}
    
    try:
        # Get LSTM prediction
        predicted_price = predict_next_price(price_values, lookback)
        
        if predicted_price is None:
            return {'Asset A': 0.5, 'Cash': 0.5}
        
        current_price = price_values[-1]
        
        # Store prediction for analysis
        predictions_history.append({
            'epoch': epoch,
            'current': current_price,
            'predicted': predicted_price
        })
        
        # Calculate expected return
        price_change = predicted_price - current_price
        price_change_pct = price_change / current_price if current_price != 0 else 0
        
        # Calculate confidence
        confidence = calculate_confidence(price_values, predicted_price, current_price)
        
        # Trading strategy based on prediction and confidence
        threshold_high = 0.003  # 0.3% expected increase
        threshold_low = -0.003   # 0.3% expected decrease
        
        if price_change_pct > threshold_high:
            # Predicted increase - allocate more to Asset A
            allocation = 0.5 + confidence * 0.4  # Range: 0.62 to 0.82
            asset_allocation = min(allocation, 0.85)
            return {'Asset A': asset_allocation, 'Cash': 1.0 - asset_allocation}
        
        elif price_change_pct < threshold_low:
            # Predicted decrease - allocate more to Cash
            allocation = 0.5 - confidence * 0.4  # Range: 0.18 to 0.38
            asset_allocation = max(allocation, 0.15)
            return {'Asset A': asset_allocation, 'Cash': 1.0 - asset_allocation}
        
        else:
            # Neutral prediction - balanced allocation
            return {'Asset A': 0.5, 'Cash': 0.5}
    
    except Exception as e:
        print(f"Error in decision making at epoch {epoch}: {e}")
        return {'Asset A': 0.5, 'Cash': 0.5}

def get_prediction_stats():
    """Get statistics about predictions (useful for debugging)"""
    if len(predictions_history) == 0:
        return None
    
    df = pd.DataFrame(predictions_history)
    df['error'] = df['predicted'] - df['current']
    df['error_pct'] = (df['error'] / df['current']) * 100
    
    stats = {
        'total_predictions': len(df),
        'mean_error': df['error'].mean(),
        'mean_error_pct': df['error_pct'].mean(),
        'rmse': np.sqrt((df['error'] ** 2).mean()),
        'mae': df['error'].abs().mean()
    }
    
    return stats