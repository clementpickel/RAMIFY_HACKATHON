import numpy as np
import pandas as pd
import joblib
import os
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt

def create_features(data, lookback=30):
    """
    Create features for logistic regression from price data
    """
    df = pd.DataFrame(data, columns=['price'])
    
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
    
    # Target: Price direction (1 = up, 0 = down)
    # We predict if next price will be higher than current
    df['next_price'] = df['price'].shift(-1)
    df['target'] = (df['next_price'] > df['price']).astype(int)
    
    # Drop rows with NaN values
    df = df.dropna()
    
    return df

def prepare_data(data, lookback=30):
    """Prepare features and target for training"""
    df = create_features(data, lookback)
    
    # Separate features and target
    X = df.drop(['price', 'next_price', 'target'], axis=1)
    y = df['target']
    
    feature_names = X.columns.tolist()
    
    return X.values, y.values, feature_names

def train_logistic_model(train_data_path, lookback=30):
    """Train logistic regression model"""
    
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
    
    # Check class distribution
    unique, counts = np.unique(y, return_counts=True)
    print(f"\nClass distribution:")
    print(f"  Down (0): {counts[0]} samples ({counts[0]/len(y)*100:.1f}%)")
    print(f"  Up (1): {counts[1]} samples ({counts[1]/len(y)*100:.1f}%)")
    
    # Standardize features (important for logistic regression)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Time series split for validation
    tscv = TimeSeriesSplit(n_splits=5)
    
    print("\n" + "="*60)
    print("Training Logistic Regression Model...")
    print("="*60)
    
    # Train multiple models with different regularization strengths
    C_values = [0.01, 0.1, 1.0, 10.0]
    best_score = 0
    best_model = None
    best_C = None
    
    for C in C_values:
        print(f"\nTesting C={C}...")
        model = LogisticRegression(
            C=C,
            solver='lbfgs',
            max_iter=1000,
            random_state=42,
            class_weight='balanced'  # Handle class imbalance
        )
        
        fold_scores = []
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X_scaled)):
            X_train_fold = X_scaled[train_idx]
            y_train_fold = y[train_idx]
            X_val_fold = X_scaled[val_idx]
            y_val_fold = y[val_idx]
            
            model.fit(X_train_fold, y_train_fold)
            y_pred = model.predict(X_val_fold)
            score = accuracy_score(y_val_fold, y_pred)
            fold_scores.append(score)
            print(f"  Fold {fold+1}: Accuracy = {score*100:.2f}%")
        
        avg_score = np.mean(fold_scores)
        print(f"  Average Accuracy: {avg_score*100:.2f}%")
        
        if avg_score > best_score:
            best_score = avg_score
            best_C = C
            # Retrain on all data with best C
            best_model = LogisticRegression(
                C=C,
                solver='lbfgs',
                max_iter=1000,
                random_state=42,
                class_weight='balanced'
            )
            best_model.fit(X_scaled, y)
    
    print("\n" + "="*60)
    print(f"✓ Best model selected with C={best_C}")
    print(f"  Cross-validation accuracy: {best_score*100:.2f}%")
    print("="*60)
    
    return best_model, scaler, feature_names

def evaluate_model(model, scaler, feature_names, test_data_path, lookback=30):
    """Evaluate model on test data"""
    print(f"\nEvaluating model on test data: {test_data_path}")
    
    df = pd.read_csv(test_data_path, index_col=0)
    data = df['Asset A'].values
    
    X, y, _ = prepare_data(data, lookback)
    X_scaled = scaler.transform(X)
    
    # Make predictions
    y_pred = model.predict(X_scaled)
    y_pred_proba = model.predict_proba(X_scaled)[:, 1]  # Probability of UP
    
    # Calculate metrics
    accuracy = accuracy_score(y, y_pred)
    
    print(f"\nTest Accuracy: {accuracy*100:.2f}%")
    print(f"Baseline (always predict majority): {max(np.mean(y), 1-np.mean(y))*100:.2f}%")
    
    # Confusion matrix
    cm = confusion_matrix(y, y_pred)
    print(f"\nConfusion Matrix:")
    print(f"                Predicted Down  Predicted Up")
    print(f"Actual Down:         {cm[0,0]:6d}        {cm[0,1]:6d}")
    print(f"Actual Up:           {cm[1,0]:6d}        {cm[1,1]:6d}")
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(y, y_pred, target_names=['Down', 'Up']))
    
    # Calculate profitability metric
    # If we bet on UP when predicted UP, what's our success rate?
    up_predictions = y_pred == 1
    if np.sum(up_predictions) > 0:
        up_accuracy = np.mean(y[up_predictions] == 1)
        print(f"Accuracy when predicting UP: {up_accuracy*100:.2f}%")
    
    down_predictions = y_pred == 0
    if np.sum(down_predictions) > 0:
        down_accuracy = np.mean(y[down_predictions] == 0)
        print(f"Accuracy when predicting DOWN: {down_accuracy*100:.2f}%")
    
    return y_pred, y_pred_proba, y

def plot_feature_importance(model, feature_names):
    """Plot feature coefficients (importance) for logistic regression"""
    coefficients = model.coef_[0]
    
    # Get top 20 features by absolute coefficient value
    abs_coef = np.abs(coefficients)
    indices = np.argsort(abs_coef)[-20:]
    
    plt.figure(figsize=(10, 8))
    colors = ['red' if coefficients[i] < 0 else 'green' for i in indices]
    plt.barh(range(len(indices)), coefficients[indices], color=colors)
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
    plt.xlabel('Coefficient Value')
    plt.title('Top 20 Feature Coefficients (Red=Down, Green=Up)')
    plt.axvline(x=0, color='black', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.savefig('logistic_feature_importance.png', dpi=300)
    print("\nFeature importance plot saved as 'logistic_feature_importance.png'")

def plot_predictions(predictions, probabilities, actual):
    """Plot prediction probabilities over time"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    # Plot 1: Prediction probabilities
    time_steps = range(len(probabilities))
    ax1.plot(time_steps, probabilities, label='P(Price Up)', linewidth=1.5, alpha=0.7)
    ax1.axhline(y=0.5, color='black', linestyle='--', linewidth=1, label='Decision Threshold')
    ax1.fill_between(time_steps, 0.5, probabilities, where=(probabilities >= 0.5), 
                      alpha=0.3, color='green', label='Predict UP')
    ax1.fill_between(time_steps, probabilities, 0.5, where=(probabilities < 0.5), 
                      alpha=0.3, color='red', label='Predict DOWN')
    ax1.set_xlabel('Time Step')
    ax1.set_ylabel('Probability')
    ax1.set_title('Model Prediction Probabilities')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Actual outcomes vs predictions
    correct = predictions == actual
    incorrect = ~correct
    
    ax2.scatter(np.where(correct)[0], probabilities[correct], 
                c='green', alpha=0.5, s=10, label='Correct Predictions')
    ax2.scatter(np.where(incorrect)[0], probabilities[incorrect], 
                c='red', alpha=0.5, s=10, label='Incorrect Predictions')
    ax2.axhline(y=0.5, color='black', linestyle='--', linewidth=1)
    ax2.set_xlabel('Time Step')
    ax2.set_ylabel('Prediction Probability')
    ax2.set_title('Prediction Accuracy Over Time')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('logistic_predictions.png', dpi=300)
    print("Predictions plot saved as 'logistic_predictions.png'")

def save_model(model, scaler, feature_names, lookback=30):
    """Save model and scalers"""
    model_data = {
        'model': model,
        'scaler': scaler,
        'feature_names': feature_names,
        'model_type': 'LogisticRegression',
        'lookback': lookback
    }
    
    model_path = 'logistic_model.pkl'
    joblib.dump(model_data, model_path)
    
    print(f"\n✓ Model saved to {model_path}")
    print(f"  Model type: Logistic Regression")
    print(f"  Features: {len(feature_names)}")
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
    
    print("="*60)
    print("Logistic Regression Trading Bot - Training Script")
    print("="*60)
    
    # Training parameters
    lookback = 30  # Shorter lookback for logistic regression
    
    # Train model
    model, scaler, feature_names = train_logistic_model(train_data_path, lookback)
    
    # Plot feature importance
    plot_feature_importance(model, feature_names)
    
    # Evaluate on test data if available
    if os.path.exists(test_data_path):
        predictions, probabilities, actual = evaluate_model(
            model, scaler, feature_names, test_data_path, lookback
        )
        plot_predictions(predictions, probabilities, actual)
    else:
        print(f"\nTest data not found at {test_data_path}, skipping evaluation")
    
    # Save model
    save_model(model, scaler, feature_names, lookback)
    
    print("\n" + "="*60)
    print("Training completed successfully!")
    print("="*60)
    print("\nYou can now use bot_trade.py with the trained logistic model.")
    print("Run: python3 main.py data/asset_a_test.csv")

if __name__ == "__main__":
    main()