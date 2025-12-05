import numpy as np
import pandas as pd
import pickle
import os
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam

def prepare_lstm_data(data, lookback=60):
    """Prepare data for LSTM with lookback window"""
    X, y = [], []
    for i in range(lookback, len(data)):
        X.append(data[i-lookback:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

def build_lstm_model(lookback=60):
    """Build LSTM model architecture"""
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(lookback, 1)),
        Dropout(0.2),
        LSTM(50, return_sequences=True),
        Dropout(0.2),
        LSTM(25),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    return model

def train_expanding_window(train_data_path, initial_window=500, expand_size=200, lookback=60):
    """Train LSTM using expanding window partition"""
    
    # Load training data
    df = pd.read_csv(train_data_path, index_col=0)
    data = df['Asset A'].values.reshape(-1, 1)
    
    # Normalize data
    scaler = MinMaxScaler(feature_range=(0, 1))
    normalized_data = scaler.fit_transform(data)
    
    total_samples = len(normalized_data)
    current_window_end = initial_window
    
    print(f"Starting expanding window training on {total_samples} samples...")
    print(f"Initial window: {initial_window}, Expand size: {expand_size}")
    
    model = build_lstm_model(lookback)
    
    while current_window_end < total_samples:
        # Use data from start to current_window_end (expanding window)
        train_window_data = normalized_data[:current_window_end]
        
        # Prepare training data
        X_train, y_train = prepare_lstm_data(train_window_data, lookback)
        
        if len(X_train) < 10:  # Skip if too few samples
            current_window_end += expand_size
            continue
        
        # Reshape for LSTM [samples, timesteps, features]
        X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
        
        # Train on expanding window
        print(f"Training on window: 0 to {current_window_end} samples ({len(X_train)} batches)")
        model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)
        
        current_window_end += expand_size
    
    print("Expanding window training completed!")
    
    return model, scaler

def save_model(model, scaler, model_path='model_lstm.pkl', scaler_path='scaler.pkl'):
    """Save model and scaler to pickle files"""
    model_dir = os.path.dirname(model_path)
    if model_dir and not os.path.exists(model_dir):
        os.makedirs(model_dir)
    
    # Save Keras model as H5 format first, then pickle the reference
    # (Keras models are better saved as H5)
    model_h5_path = model_path.replace('.pkl', '.h5')
    model.save(model_h5_path)
    print(f"Model saved to {model_h5_path}")
    
    # Save scaler as pickle
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"Scaler saved to {scaler_path}")
    
    # Save metadata
    metadata = {
        'model_path': model_h5_path,
        'scaler_path': scaler_path,
        'lookback': 60
    }
    metadata_path = 'model_metadata.pkl'
    with open(metadata_path, 'wb') as f:
        pickle.dump(metadata, f)
    print(f"Metadata saved to {metadata_path}")

def main():
    """Main training script"""
    train_data_path = os.path.join(os.path.dirname(__file__), 'data', 'asset_a_train.csv')
    
    if not os.path.exists(train_data_path):
        print(f"Error: Training data not found at {train_data_path}")
        return
    
    # Train model
    model, scaler = train_expanding_window(train_data_path)
    
    # Save model and scaler
    save_model(model, scaler)
    
    print("Training completed and model saved successfully!")

if __name__ == "__main__":
    main()
