import numpy as np
import pandas as pd
import pickle
import os
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

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
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
    return model

def train_expanding_window(train_data_path, initial_window=500, expand_size=200, lookback=60):
    """Train LSTM using expanding window partition"""
    
    # Load training data
    print(f"Loading training data from: {train_data_path}")
    df = pd.read_csv(train_data_path, index_col=0)
    data = df['Asset A'].values.reshape(-1, 1)
    
    print(f"Loaded {len(data)} data points")
    print(f"Price range: {data.min():.2f} to {data.max():.2f}")
    
    # Normalize data
    scaler = MinMaxScaler(feature_range=(0, 1))
    normalized_data = scaler.fit_transform(data)
    
    total_samples = len(normalized_data)
    current_window_end = initial_window
    
    print(f"\nStarting expanding window training on {total_samples} samples...")
    print(f"Initial window: {initial_window}, Expand size: {expand_size}, Lookback: {lookback}")
    
    model = build_lstm_model(lookback)
    
    training_losses = []
    window_count = 0
    
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
        window_count += 1
        print(f"\n[Window {window_count}] Training on samples 0 to {current_window_end} ({len(X_train)} sequences)")
        history = model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=1, validation_split=0.1)
        
        final_loss = history.history['loss'][-1]
        training_losses.append(final_loss)
        print(f"Final loss for this window: {final_loss:.6f}")
        
        current_window_end += expand_size
    
    print("\n" + "="*50)
    print("Expanding window training completed!")
    print(f"Total windows trained: {window_count}")
    print(f"Average loss: {np.mean(training_losses):.6f}")
    print("="*50)
    
    return model, scaler, training_losses

def evaluate_model(model, scaler, test_data_path, lookback=60):
    """Evaluate model on test data"""
    print(f"\nEvaluating model on test data: {test_data_path}")
    
    df = pd.read_csv(test_data_path, index_col=0)
    data = df['Asset A'].values.reshape(-1, 1)
    
    normalized_data = scaler.transform(data)
    X_test, y_test = prepare_lstm_data(normalized_data, lookback)
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
    
    # Make predictions
    predictions = model.predict(X_test, verbose=0)
    
    # Denormalize
    predictions = scaler.inverse_transform(predictions)
    y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))
    
    # Calculate metrics
    mse = np.mean((predictions - y_test_actual) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(predictions - y_test_actual))
    
    print(f"Test MSE: {mse:.4f}")
    print(f"Test RMSE: {rmse:.4f}")
    print(f"Test MAE: {mae:.4f}")
    
    # Calculate directional accuracy
    actual_direction = np.sign(y_test_actual[1:] - y_test_actual[:-1])
    pred_direction = np.sign(predictions[1:] - y_test_actual[:-1])
    directional_accuracy = np.mean(actual_direction == pred_direction)
    print(f"Directional Accuracy: {directional_accuracy*100:.2f}%")
    
    return predictions, y_test_actual

def plot_training_results(training_losses):
    """Plot training losses"""
    plt.figure(figsize=(10, 6))
    plt.plot(training_losses, marker='o')
    plt.title('Training Loss per Expanding Window')
    plt.xlabel('Window Number')
    plt.ylabel('Loss')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('training_losses.png', dpi=300)
    print("\nTraining loss plot saved as 'training_losses.png'")

def plot_predictions(predictions, actual, num_points=200):
    """Plot predictions vs actual values"""
    plt.figure(figsize=(14, 6))
    
    # Plot only last num_points for clarity
    start_idx = max(0, len(predictions) - num_points)
    
    plt.plot(actual[start_idx:], label='Actual Price', linewidth=2)
    plt.plot(predictions[start_idx:], label='Predicted Price', linewidth=2, alpha=0.7)
    plt.title(f'LSTM Predictions vs Actual Prices (Last {num_points} points)')
    plt.xlabel('Time Step')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('predictions_comparison.png', dpi=300)
    print("Predictions plot saved as 'predictions_comparison.png'")

def save_model(model, scaler, lookback=60):
    """Save model and scaler"""
    model_h5_path = 'model_lstm.h5'
    scaler_path = 'scaler.pkl'
    
    # Save Keras model as H5
    model.save(model_h5_path)
    print(f"\nModel saved to {model_h5_path}")
    
    # Save scaler as pickle
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"Scaler saved to {scaler_path}")
    
    # Save metadata
    metadata = {
        'model_path': model_h5_path,
        'scaler_path': scaler_path,
        'lookback': lookback
    }
    metadata_path = 'model_metadata.pkl'
    with open(metadata_path, 'wb') as f:
        pickle.dump(metadata, f)
    print(f"Metadata saved to {metadata_path}")

def main():
    """Main training script"""
    # Set paths - update these to match your actual file locations
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
    
    print("="*50)
    print("LSTM Trading Bot - Training Script")
    print("="*50)
    
    # Training parameters
    lookback = 60
    initial_window = 500
    expand_size = 200
    
    # Train model
    model, scaler, training_losses = train_expanding_window(
        train_data_path, 
        initial_window=initial_window, 
        expand_size=expand_size,
        lookback=lookback
    )
    
    # Plot training progress
    plot_training_results(training_losses)
    
    # Evaluate on test data if available
    if os.path.exists(test_data_path):
        predictions, actual = evaluate_model(model, scaler, test_data_path, lookback)
        plot_predictions(predictions, actual)
    else:
        print(f"\nTest data not found at {test_data_path}, skipping evaluation")
    
    # Save model and scaler
    save_model(model, scaler, lookback)
    
    print("\n" + "="*50)
    print("Training completed successfully!")
    print("="*50)
    print("\nYou can now use bot_trade.py with the trained model.")
    print("Run: python3 main.py data/asset_a_test.csv")

if __name__ == "__main__":
    main()