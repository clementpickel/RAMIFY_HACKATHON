import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import pickle
import os

history = []
model = None
scaler = None
lookback = 60
predictions_history = []
model_loaded = False

def load_trained_model():
    """Load pre-trained model and scaler from pickle files"""
    global model, scaler, model_loaded
    
    current_dir = os.path.dirname(__file__)
    model_h5_path = os.path.join(current_dir, 'model_lstm.h5')
    scaler_path = os.path.join(current_dir, 'scaler.pkl')
    
    try:
        # Load scaler
        if os.path.exists(scaler_path):
            with open(scaler_path, 'rb') as f:
                scaler = pickle.load(f)
            print("Scaler loaded successfully")
        else:
            print(f"Scaler file not found at {scaler_path}")
            return False
        
        # Load model
        if os.path.exists(model_h5_path):
            model = load_model(model_h5_path)
            print("Model loaded successfully")
            model_loaded = True
            return True
        else:
            print(f"Model file not found at {model_h5_path}")
            print("Please run train_model.py to train and save the model first")
            return False
    
    except Exception as e:
        print(f"Error loading model: {e}")
        return False

def predict_next_direction(price_history, lookback=60):
    """Predict price direction using trained LSTM"""
    global scaler, model
    
    if model is None or scaler is None:
        return 0.5
    
    if len(price_history) < lookback:
        return 0.5  # Neutral if insufficient history
    
    try:
        # Prepare recent data
        recent_data = np.array(price_history[-lookback:]).reshape(-1, 1)
        normalized = scaler.transform(recent_data)
        
        # Reshape for LSTM
        X = normalized.reshape((1, lookback, 1))
        
        # Make prediction
        predicted_price = model.predict(X, verbose=0)[0][0]
        
        # Denormalize
        predicted_price = scaler.inverse_transform([[predicted_price]])[0][0]
        
        return predicted_price
    except Exception as e:
        print(f"Error in prediction: {e}")
        return 0.5

def make_decision(epoch: int, price: float):
    """Make trading decision based on LSTM predictions"""
    global model, predictions_history, model_loaded
    
    history.append({"epoch": epoch, "price": price})
    price_values = [h["price"] for h in history]
    
    # Load model on first call if not already loaded
    if not model_loaded:
        load_trained_model()
    
    # If model couldn't be loaded, use neutral strategy
    if model is None or scaler is None:
        return {'Asset A': 0.5, 'Cash': 0.5}
    
    # Need minimum history for prediction
    if len(price_values) < lookback + 1:
        return {'Asset A': 0.5, 'Cash': 0.5}
    
    try:
        # Get LSTM prediction
        predicted_price = predict_next_direction(price_values, lookback)
        current_price = price_values[-1]
        
        # Store prediction
        predictions_history.append({
            'epoch': epoch,
            'current': current_price,
            'predicted': predicted_price
        })
        
        # Trading decision based on predicted price direction
        price_change = predicted_price - current_price
        
        if price_change > 0.001:  # Predicted increase
            return {'Asset A': 0.75, 'Cash': 0.25}
        elif price_change < -0.001:  # Predicted decrease
            return {'Asset A': 0.25, 'Cash': 0.75}
        else:  # Neutral prediction
            return {'Asset A': 0.5, 'Cash': 0.5}
    
    except Exception as e:
        print(f"Error in decision making: {e}")
        return {'Asset A': 0.5, 'Cash': 0.5}

# import numpy as np
# import pandas as pd
# from sklearn.preprocessing import MinMaxScaler
# from sklearn.metrics import mean_squared_error
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import LSTM, Dense, Dropout
# from tensorflow.keras.optimizers import Adam
# import os

# history = []
# model = None
# scaler = MinMaxScaler(feature_range=(0, 1))
# lookback = 60
# predictions_history = []

# def prepare_lstm_data(data, lookback=60):
#     """Prepare data for LSTM with lookback window"""
#     X, y = [], []
#     for i in range(lookback, len(data)):
#         X.append(data[i-lookback:i, 0])
#         y.append(data[i, 0])
#     return np.array(X), np.array(y)

# def build_lstm_model(lookback=60):
#     """Build LSTM model architecture"""
#     model = Sequential([
#         LSTM(50, return_sequences=True, input_shape=(lookback, 1)),
#         Dropout(0.2),
#         LSTM(50, return_sequences=True),
#         Dropout(0.2),
#         LSTM(25),
#         Dropout(0.2),
#         Dense(1)
#     ])
#     model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
#     return model

# def train_expanding_window(train_data_path, initial_window=500, expand_size=200):
#     """Train LSTM using expanding window partition"""
#     global model, scaler
    
#     # Load training data
#     df = pd.read_csv(train_data_path, index_col=0)
#     data = df['Asset A'].values.reshape(-1, 1)
    
#     # Normalize data
#     normalized_data = scaler.fit_transform(data)
    
#     total_samples = len(normalized_data)
#     current_window_end = initial_window
    
#     print(f"Starting expanding window training on {total_samples} samples...")
    
#     while current_window_end < total_samples:
#         # Use data from start to current_window_end (expanding window)
#         train_window_data = normalized_data[:current_window_end]
        
#         # Prepare training data
#         X_train, y_train = prepare_lstm_data(train_window_data, lookback)
        
#         if len(X_train) < 10:  # Skip if too few samples
#             current_window_end += expand_size
#             continue
        
#         # Reshape for LSTM [samples, timesteps, features]
#         X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
        
#         # Build or use existing model
#         if model is None:
#             model = build_lstm_model(lookback)
        
#         # Train on expanding window
#         print(f"Training on window: 0 to {current_window_end} samples ({len(X_train)} batches)")
#         model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)
        
#         current_window_end += expand_size
    
#     print("Expanding window training completed!")
#     return model

# def initialize_model(train_data_path):
#     """Initialize and train the model"""
#     global model
#     model = train_expanding_window(train_data_path)

# def predict_next_direction(price_history, lookback=60):
#     """Predict price direction using trained LSTM"""
#     if len(price_history) < lookback:
#         return 0.5  # Neutral if insufficient history
    
#     # Prepare recent data
#     recent_data = np.array(price_history[-lookback:]).reshape(-1, 1)
#     normalized = scaler.transform(recent_data)
    
#     # Reshape for LSTM
#     X = normalized.reshape((1, lookback, 1))
    
#     # Make prediction
#     predicted_price = model.predict(X, verbose=0)[0][0]
    
#     # Denormalize
#     predicted_price = scaler.inverse_transform([[predicted_price]])[0][0]
    
#     return predicted_price

# def make_decision(epoch: int, price: float):
#     """Make trading decision based on LSTM predictions"""
#     global model, predictions_history
    
#     history.append({"epoch": epoch, "price": price})
#     price_values = [h["price"] for h in history]
    
#     # Initialize model on first call if not already done
#     if model is None:
#         try:
#             # Try to train on available data
#             train_data_path = os.path.join(os.path.dirname(__file__), 'data', 'asset_a_train.csv')
#             if os.path.exists(train_data_path):
#                 initialize_model(train_data_path)
#             else:
#                 # If no training data, use random strategy
#                 return {'Asset A': 0.5, 'Cash': 0.5}
#         except Exception as e:
#             print(f"Error initializing model: {e}")
#             return {'Asset A': 0.5, 'Cash': 0.5}
    
#     # Need minimum history for prediction
#     if len(price_values) < lookback + 1:
#         return {'Asset A': 0.5, 'Cash': 0.5}
    
#     try:
#         # Get LSTM prediction
#         predicted_price = predict_next_direction(price_values, lookback)
#         current_price = price_values[-1]
        
#         # Store prediction
#         predictions_history.append({
#             'epoch': epoch,
#             'current': current_price,
#             'predicted': predicted_price
#         })
        
#         # Trading decision based on predicted price direction
#         price_change = predicted_price - current_price
        
#         if price_change > 0.001:  # Predicted increase
#             return {'Asset A': 0.75, 'Cash': 0.25}
#         elif price_change < -0.001:  # Predicted decrease
#             return {'Asset A': 0.25, 'Cash': 0.75}
#         else:  # Neutral prediction
#             return {'Asset A': 0.5, 'Cash': 0.5}
    
#     except Exception as e:
#         print(f"Error in prediction: {e}")
#         return {'Asset A': 0.5, 'Cash': 0.5}