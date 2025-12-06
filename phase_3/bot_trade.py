import numpy as np
from collections import deque
import joblib
import os

# Global variables
history = deque(maxlen=100)
price_a_window = deque(maxlen=60)
price_b_window = deque(maxlen=60)

# Neural network model
nn_model = None
nn_scaler = None
nn_loaded = False

def load_nn_model():
    """Load pre-trained neural network model"""
    global nn_model, nn_scaler, nn_loaded
    
    if nn_loaded:
        return nn_model, nn_scaler
    
    current_dir = os.path.dirname(__file__) if os.path.dirname(__file__) else '.'
    model_path = os.path.join(current_dir, 'nn_allocation_model.pkl')
    
    try:
        if os.path.exists(model_path):
            model_data = joblib.load(model_path)
            nn_model = model_data['model']
            nn_scaler = model_data['scaler']
            nn_loaded = True
            # print("✓ Neural Network model loaded successfully")
            return nn_model, nn_scaler
    except Exception as e:
        # print(f"⚠ Could not load NN model: {e}")
        pass
    
    nn_loaded = True
    return None, None

def create_nn_features(price_a_window, price_b_window):
    """
    Create feature vector for neural network prediction.
    Must match the features used during training.
    """
    if len(price_a_window) < 30 or len(price_b_window) < 30:
        return None
    
    window_a = np.array(list(price_a_window))
    window_b = np.array(list(price_b_window))
    
    # ========== Asset A Features ==========
    returns_a = np.diff(window_a) / window_a[:-1]
    
    lag_1_a = returns_a[-1] if len(returns_a) > 0 else 0
    lag_5_a = np.mean(returns_a[-5:]) if len(returns_a) >= 5 else 0
    lag_10_a = np.mean(returns_a[-10:]) if len(returns_a) >= 10 else 0
    
    sma_10_a = np.mean(window_a[-10:])
    sma_20_a = np.mean(window_a[-20:]) if len(window_a) >= 20 else np.mean(window_a)
    price_vs_sma_a = (window_a[-1] - sma_20_a) / sma_20_a if sma_20_a != 0 else 0
    
    momentum_a = (window_a[-1] - window_a[0]) / window_a[0] if window_a[0] != 0 else 0
    volatility_a = np.std(returns_a) if len(returns_a) > 0 else 0
    
    gains_a = np.sum(np.maximum(returns_a, 0))
    losses_a = np.sum(np.maximum(-returns_a, 0))
    rs_a = gains_a / losses_a if losses_a > 0 else 1
    rsi_a = 100 - (100 / (1 + rs_a))
    
    rolling_min_a = np.min(window_a[-20:])
    rolling_max_a = np.max(window_a[-20:])
    range_a = rolling_max_a - rolling_min_a
    position_a = (window_a[-1] - rolling_min_a) / range_a if range_a > 0 else 0.5
    
    # ========== Asset B Features ==========
    returns_b = np.diff(window_b) / window_b[:-1]
    
    lag_1_b = returns_b[-1] if len(returns_b) > 0 else 0
    lag_5_b = np.mean(returns_b[-5:]) if len(returns_b) >= 5 else 0
    lag_10_b = np.mean(returns_b[-10:]) if len(returns_b) >= 10 else 0
    
    sma_10_b = np.mean(window_b[-10:])
    sma_20_b = np.mean(window_b[-20:]) if len(window_b) >= 20 else np.mean(window_b)
    price_vs_sma_b = (window_b[-1] - sma_20_b) / sma_20_b if sma_20_b != 0 else 0
    
    momentum_b = (window_b[-1] - window_b[0]) / window_b[0] if window_b[0] != 0 else 0
    volatility_b = np.std(returns_b) if len(returns_b) > 0 else 0
    
    gains_b = np.sum(np.maximum(returns_b, 0))
    losses_b = np.sum(np.maximum(-returns_b, 0))
    rs_b = gains_b / losses_b if losses_b > 0 else 1
    rsi_b = 100 - (100 / (1 + rs_b))
    
    rolling_min_b = np.min(window_b[-20:])
    rolling_max_b = np.max(window_b[-20:])
    range_b = rolling_max_b - rolling_min_b
    position_b = (window_b[-1] - rolling_min_b) / range_b if range_b > 0 else 0.5
    
    # ========== Correlation Features ==========
    correlation = np.corrcoef(window_a, window_b)[0, 1] if len(window_a) >= 2 else 0
    correlation = correlation if not np.isnan(correlation) else 0
    
    relative_momentum = momentum_a - momentum_b
    
    # Feature vector (22 features)
    features = np.array([
        lag_1_a, lag_5_a, lag_10_a,
        sma_10_a, sma_20_a, price_vs_sma_a,
        momentum_a, volatility_a, rsi_a, position_a,
        
        lag_1_b, lag_5_b, lag_10_b,
        sma_10_b, sma_20_b, price_vs_sma_b,
        momentum_b, volatility_b, rsi_b, position_b,
        
        correlation, relative_momentum
    ])
    
    return features.reshape(1, -1)

def get_nn_allocation(price_a_window, price_b_window):
    """
    Get allocation from neural network model.
    
    Returns:
        dict: Allocations or None if model not available
    """
    model, scaler = load_nn_model()
    
    if model is None or scaler is None:
        return None
    
    try:
        features = create_nn_features(price_a_window, price_b_window)
        if features is None:
            return None
        
        # Scale features
        features_scaled = scaler.transform(features)
        
        # Get prediction
        allocations_pred = model.predict(features_scaled)[0]
        
        # Ensure valid allocations
        allocations = {
            'Asset A': float(max(0.05, min(0.95, allocations_pred[0]))),
            'Asset B': float(max(0.05, min(0.95, allocations_pred[1]))),
            'Cash': float(max(0.05, min(0.95, allocations_pred[2])))
        }
        
        # Normalize
        total = sum(allocations.values())
        if total > 0:
            allocations = {k: v / total for k, v in allocations.items()}
        
        return allocations
    except Exception as e:
        # print(f"⚠ NN prediction error: {e}")
        return None

def calculate_indicators(price_history):
    """
    Calculate technical indicators from price history.
    
    Args:
        price_history: Deque of prices
        
    Returns:
        dict: Technical indicators
    """
    if len(price_history) < 2:
        return {}
    
    prices = np.array(list(price_history))
    
    # Moving averages
    if len(prices) >= 20:
        sma_20 = np.mean(prices[-20:])
    else:
        sma_20 = np.mean(prices)
    
    if len(prices) >= 10:
        sma_10 = np.mean(prices[-10:])
    else:
        sma_10 = np.mean(prices)
    
    # Volatility
    volatility = np.std(prices[-min(20, len(prices)):])
    
    # Momentum
    momentum = (prices[-1] - prices[0]) / prices[0] if prices[0] != 0 else 0
    
    # RSI-like indicator
    if len(prices) >= 2:
        returns = np.diff(prices) / prices[:-1]
        gains = np.sum(np.maximum(returns, 0))
        losses = np.sum(np.maximum(-returns, 0))
        rs = gains / losses if losses > 0 else 1
        rsi = 100 - (100 / (1 + rs))
    else:
        rsi = 50
    
    # Current position relative to moving average
    price_above_sma = 1 if prices[-1] > sma_20 else 0
    
    return {
        'sma_20': sma_20,
        'sma_10': sma_10,
        'volatility': volatility,
        'momentum': momentum,
        'rsi': rsi,
        'price_above_sma': price_above_sma,
        'current_price': prices[-1]
    }

def calculate_asset_signal(price_history, asset_name="Asset"):
    """
    Calculate trading signal for a single asset.
    
    Returns signal between -1.0 (strong sell) and 1.0 (strong buy)
    """
    if len(price_history) < 10:
        return 0.0
    
    indicators = calculate_indicators(price_history)
    
    signals = []
    
    # Price relative to SMA signal (increased weight for more aggressive trading)
    if indicators['price_above_sma']:
        signals.append(0.4)  # Stronger bullish signal
    else:
        signals.append(-0.4)  # Stronger bearish signal
    
    # Momentum signal (increased weight)
    if indicators['momentum'] > 0.01:
        signals.append(0.3)  # Stronger momentum boost
    elif indicators['momentum'] < -0.01:
        signals.append(-0.3)  # Stronger momentum penalty
    else:
        signals.append(0)
    
    # RSI signal (more aggressive extremes)
    if indicators['rsi'] > 70:
        signals.append(-0.2)  # Stronger overbought signal
    elif indicators['rsi'] < 30:
        signals.append(0.2)  # Stronger oversold signal
    else:
        signals.append(0)
    
    # Volatility adjustment (reduced penalty for more risk appetite)
    vol_signal = -0.05 if indicators['volatility'] > 0.03 else 0.08
    signals.append(vol_signal)
    
    return sum(signals)

def normalize_allocation(allocations):
    """
    Normalize allocations to sum to 1.0
    
    Args:
        allocations: Dict with Asset A, Asset B, Cash keys
        
    Returns:
        Normalized allocations
    """
    total = sum(allocations.values())
    if total <= 0:
        return {'Asset A': 1/3, 'Asset B': 1/3, 'Cash': 1/3}
    
    return {k: v / total for k, v in allocations.items()}

def make_decision(epoch: int, priceA: float, priceB: float):
    """
    Make portfolio allocation decision using neural network + technical analysis.
    
    Strategy:
    - Load trained neural network model for optimal allocations
    - Use technical analysis as fallback if NN not available
    - Blend both signals for robustness
    
    Args:
        epoch: Current time step
        priceA: Price of Asset A
        priceB: Price of Asset B
        
    Returns:
        dict: Portfolio allocation {'Asset A': float, 'Asset B': float, 'Cash': float}
    """
    
    # Store price history
    history.append({"epoch": epoch, "priceA": priceA, "priceB": priceB})
    price_a_window.append(priceA)
    price_b_window.append(priceB)
    
    # Warmup phase - equal allocation
    if len(price_a_window) < 30 or len(price_b_window) < 30:
        return {'Asset A': 1/3, 'Asset B': 1/3, 'Cash': 1/3}
    
    # Try to get neural network prediction
    nn_allocation = get_nn_allocation(price_a_window, price_b_window)
    
    if nn_allocation is not None:
        # Use NN allocation with slight smoothing
        # Weight NN: 70%, Technical: 30%
        tech_allocation = calculate_technical_allocation(price_a_window, price_b_window)
        
        # Blend allocations (NN 80% + Technical 20% for more aggressive positioning)
        blended = {
            'Asset A': nn_allocation['Asset A'] * 0.8 + tech_allocation['Asset A'] * 0.2,
            'Asset B': nn_allocation['Asset B'] * 0.8 + tech_allocation['Asset B'] * 0.2,
            'Cash': nn_allocation['Cash'] * 0.8 + tech_allocation['Cash'] * 0.2
        }
        
        # Normalize
        total = sum(blended.values())
        if total > 0:
            blended = {k: v / total for k, v in blended.items()}
        
        return blended
    else:
        # Fallback to technical analysis
        return calculate_technical_allocation(price_a_window, price_b_window)

def calculate_technical_allocation(price_a_window, price_b_window):
    """Calculate allocation using technical analysis (fallback)"""
    
    # Calculate signals for both assets
    signal_a = calculate_asset_signal(price_a_window, "Asset A")
    signal_b = calculate_asset_signal(price_b_window, "Asset B")
    
    # Get indicators for relative strength comparison
    indicators_a = calculate_indicators(price_a_window)
    indicators_b = calculate_indicators(price_b_window)
    
    # Calculate correlation/relative performance
    recent_a = np.array(list(price_a_window))[-10:]
    recent_b = np.array(list(price_b_window))[-10:]
    
    momentum_a = (recent_a[-1] - recent_a[0]) / recent_a[0] if recent_a[0] != 0 else 0
    momentum_b = (recent_b[-1] - recent_b[0]) / recent_b[0] if recent_b[0] != 0 else 0
    
    # Base allocations from signals (more aggressive range [0.1, 0.9])
    # Convert signals [-1, 1] to allocations with wider range for risk-on positioning
    base_a = 0.5 + (signal_a / 2) * 0.4  # Range [0.1, 0.9]
    base_b = 0.5 + (signal_b / 2) * 0.4  # Range [0.1, 0.9]
    
    # Normalize to available allocation (1 - cash buffer)
    total_risky = base_a + base_b
    
    # Aggressive risk management: maintain lower cash buffer (5-25%)
    vol_combined = (indicators_a['volatility'] + indicators_b['volatility']) / 2
    cash_ratio = 0.05 + (min(vol_combined / 0.06, 1.0) * 0.2)  # 5-25% cash (more aggressive)
    
    # Allocate remaining to assets based on relative signals
    risky_allocation = 1.0 - cash_ratio
    
    if total_risky > 0:
        # Proportional allocation between assets
        a_ratio = base_a / total_risky
        b_ratio = base_b / total_risky
    else:
        a_ratio = 0.5
        b_ratio = 0.5
    
    # Apply looser diversification constraint: allow up to 80% per asset for more aggressive positioning
    a_ratio = np.clip(a_ratio, 0.2, 0.8)
    b_ratio = np.clip(b_ratio, 0.2, 0.8)
    
    # Normalize ratios
    total_ratio = a_ratio + b_ratio
    a_ratio /= total_ratio
    b_ratio /= total_ratio
    
    # Final allocations
    allocation_a = a_ratio * risky_allocation
    allocation_b = b_ratio * risky_allocation
    allocation_cash = cash_ratio
    
    # Ensure minimum allocation to each asset (reduced from 5% to 2% for more risk appetite)
    min_asset = 0.02
    allocation_a = max(allocation_a, min_asset)
    allocation_b = max(allocation_b, min_asset)
    
    # Normalize to sum to 1.0
    allocations = {
        'Asset A': float(allocation_a),
        'Asset B': float(allocation_b),
        'Cash': float(allocation_cash)
    }
    
    allocations = normalize_allocation(allocations)
    
    # Aggressive momentum boost when signals strongly agree
    if signal_a > 0.5 and signal_b > 0.5:
        # Both strongly bullish - aggressively reduce cash
        allocations['Cash'] = max(allocations['Cash'] - 0.1, 0.05)
        boost = 0.1 / 2
        allocations['Asset A'] += boost
        allocations['Asset B'] += boost
    elif signal_a < -0.5 and signal_b < -0.5:
        # Both strongly bearish - increase cash more conservatively
        allocations['Cash'] = min(allocations['Cash'] + 0.08, 0.35)
        reduction = 0.08 / 2
        allocations['Asset A'] = max(allocations['Asset A'] - reduction, 0.02)
        allocations['Asset B'] = max(allocations['Asset B'] - reduction, 0.02)
    
    # Final normalization
    allocations = normalize_allocation(allocations)
    
    return allocations