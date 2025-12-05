

import pickle
import numpy as np
import os

history = []
model = None
scaler = None

def load_model(model_path: str = 'model_rf.pkl'):
    """Load the trained Random Forest model and scaler."""
    global model, scaler
    
    if not os.path.exists(model_path):
        print(f"Warning: Model file {model_path} not found. Using fallback strategy.")
        return False
    
    try:
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
            model = model_data['model']
            scaler = model_data['scaler']
        print(f"Model loaded successfully from {model_path}")
        return True
    except Exception as e:
        print(f"Error loading model: {e}")
        return False


def create_features(prices: list) -> np.ndarray:
    """Create features from price history for model prediction."""
    if len(prices) < 10:
        return None
    
    prices_arr = np.array(prices)
    window = 10
    i = len(prices_arr) - 1
    
    # Price momentum features
    price_momentum = prices_arr[i] - prices_arr[i-window:i]
    
    # Moving averages
    ma5 = np.mean(prices_arr[i-5:i]) if len(prices_arr) >= 5 else np.mean(prices_arr)
    ma10 = np.mean(prices_arr[i-10:i]) if len(prices_arr) >= 10 else np.mean(prices_arr)
    ma20 = np.mean(prices_arr[i-20:i]) if len(prices_arr) >= 20 else np.mean(prices_arr)
    
    # Price changes
    change1 = (prices_arr[i] - prices_arr[i-1]) / prices_arr[i-1] if len(prices_arr) >= 2 else 0
    change5 = (prices_arr[i] - prices_arr[i-5]) / prices_arr[i-5] if len(prices_arr) >= 5 else 0
    change10 = (prices_arr[i] - prices_arr[i-10]) / prices_arr[i-10] if len(prices_arr) >= 10 else 0
    
    # Volatility
    volatility = np.std(np.diff(prices_arr[max(0, i-10):i])) if len(prices_arr) >= 2 else 0
    
    # Current price
    current_price = prices_arr[i]
    
    # Combine features
    features = np.concatenate([
        price_momentum,
        [ma5, ma10, ma20, change1, change5, change10, volatility, current_price]
    ])
    
    return features.reshape(1, -1)


def make_decision(epoch: int, price: float):
    """Make trading decision based on Random Forest model or fallback strategy."""
    history.append({"epoch": epoch, "price": price})
    
    # Need at least 2 prices for any decision
    if len(history) < 2:
        return {'Asset B': 0.5, 'Cash': 0.5}
    
    # Try to use model-based decision
    if model is not None and scaler is not None and len(history) >= 11:
        try:
            prices = [h['price'] for h in history]
            features = create_features(prices)
            
            if features is not None:
                # Scale features
                features_scaled = scaler.transform(features)
                
                # Get prediction and probability
                prediction = model.predict(features_scaled)[0]
                probability = model.predict_proba(features_scaled)[0]
                
                # Convert probability to allocation
                # If model predicts up (1), allocate more to Asset B
                # If model predicts down (0), allocate more to Cash
                prob_up = probability[1]
                
                # Map probability to allocation (30%-70% range)
                asset_b_allocation = 0.3 + (prob_up * 0.4)  # 30% to 70%
                cash_allocation = 1.0 - asset_b_allocation
                
                return {'Asset B': asset_b_allocation, 'Cash': cash_allocation}
        except Exception as e:
            print(f"Error in model prediction: {e}")
    
    # Fallback strategy: simple momentum-based
    delta = history[-1]["price"] - history[-2]["price"]
    if delta > 0:
        return {'Asset B': 0.7, 'Cash': 0.3}
    else:
        return {'Asset B': 0.3, 'Cash': 0.7}


# Load model on import
load_model()