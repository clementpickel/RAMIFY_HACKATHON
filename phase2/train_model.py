#!/usr/bin/env python3

import pandas as pd
import numpy as np
import pickle
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score, 
    roc_auc_score,
    confusion_matrix,
    classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns


def load_data(csv_path: str) -> pd.DataFrame:
    """Load price data from CSV file."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Data file not found: {csv_path}")
    df = pd.read_csv(csv_path, index_col=0)
    return df


def create_features(prices: pd.Series, window: int = 10) -> tuple[np.ndarray, np.ndarray]:
    """
    Create features and labels for machine learning model.
    
    Features:
    - Price momentum (current - past prices)
    - Moving averages (5, 10, 20 periods)
    - Price changes (1, 5, 10 periods)
    - Volatility (std of returns)
    
    Labels:
    - 1 if price goes up in next period
    - 0 if price goes down
    """
    prices = prices.values
    X = []
    y = []
    
    for i in range(window, len(prices) - 1):
        # Price momentum features
        price_momentum = prices[i] - prices[i-window:i]
        
        # Moving averages
        ma5 = np.mean(prices[i-5:i]) if i >= 5 else np.mean(prices[:i])
        ma10 = np.mean(prices[i-10:i]) if i >= 10 else np.mean(prices[:i])
        ma20 = np.mean(prices[i-20:i]) if i >= 20 else np.mean(prices[:i])
        
        # Price changes
        change1 = (prices[i] - prices[i-1]) / prices[i-1]
        change5 = (prices[i] - prices[i-5]) / prices[i-5] if i >= 5 else 0
        change10 = (prices[i] - prices[i-10]) / prices[i-10] if i >= 10 else 0
        
        # Volatility
        volatility = np.std(np.diff(prices[max(0, i-10):i]))
        
        # Current price
        current_price = prices[i]
        
        # Combine features
        features = np.concatenate([
            price_momentum,
            [ma5, ma10, ma20, change1, change5, change10, volatility, current_price]
        ])
        
        # Label: 1 if next price is higher, 0 otherwise
        label = 1 if prices[i+1] > prices[i] else 0
        
        X.append(features)
        y.append(label)
    
    return np.array(X), np.array(y)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray = None) -> dict:
    """Compute classification metrics."""
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
    }
    
    if y_proba is not None:
        metrics['roc_auc'] = roc_auc_score(y_true, y_proba[:, 1])
    
    metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred)
    
    return metrics


def train_model(csv_path: str, output_path: str = 'model_rf.pkl', 
                test_split: float = 0.2, n_estimators: int = 100,
                show_plot: bool = False):
    """
    Train Random Forest model and save it to pickle file.
    
    Args:
        csv_path: Path to training data CSV
        output_path: Path where to save the model pickle file
        test_split: Fraction of data to use for testing
        n_estimators: Number of trees in the forest
        show_plot: Whether to display performance plots
    """
    print("=" * 60)
    print("Training Random Forest Model")
    print("=" * 60)
    
    # Load data
    print(f"\n1. Loading data from {csv_path}...")
    df = load_data(csv_path)
    prices = df.iloc[:, 0]  # Get first column (Asset B prices)
    print(f"   Loaded {len(prices)} price points")
    
    # Create features
    print("\n2. Creating features...")
    X, y = create_features(prices, window=10)
    print(f"   Created {X.shape[0]} samples with {X.shape[1]} features")
    print(f"   Class distribution: {np.sum(y==0)} downs, {np.sum(y==1)} ups")
    
    # Split data
    split_idx = int(len(X) * (1 - test_split))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    print(f"\n3. Splitting data:")
    print(f"   Train set: {len(X_train)} samples")
    print(f"   Test set: {len(X_test)} samples")
    
    # Scale features
    print("\n4. Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train Random Forest
    print(f"\n5. Training Random Forest ({n_estimators} estimators)...")
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=15,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1,
        class_weight='balanced'
    )
    model.fit(X_train_scaled, y_train)
    print("   Training complete!")
    
    # Predictions
    print("\n6. Making predictions...")
    y_train_pred = model.predict(X_train_scaled)
    y_test_pred = model.predict(X_test_scaled)
    y_train_proba = model.predict_proba(X_train_scaled)
    y_test_proba = model.predict_proba(X_test_scaled)
    
    # Compute metrics
    print("\n7. Computing metrics...")
    train_metrics = compute_metrics(y_train, y_train_pred, y_train_proba)
    test_metrics = compute_metrics(y_test, y_test_pred, y_test_proba)
    
    # Display results
    print("\n" + "=" * 60)
    print("TRAINING SET METRICS")
    print("=" * 60)
    print(f"Accuracy:  {train_metrics['accuracy']:.4f}")
    print(f"Precision: {train_metrics['precision']:.4f}")
    print(f"Recall:    {train_metrics['recall']:.4f}")
    print(f"F1-Score:  {train_metrics['f1']:.4f}")
    print(f"ROC-AUC:   {train_metrics['roc_auc']:.4f}")
    print("\nConfusion Matrix:")
    print(train_metrics['confusion_matrix'])
    
    print("\n" + "=" * 60)
    print("TEST SET METRICS")
    print("=" * 60)
    print(f"Accuracy:  {test_metrics['accuracy']:.4f}")
    print(f"Precision: {test_metrics['precision']:.4f}")
    print(f"Recall:    {test_metrics['recall']:.4f}")
    print(f"F1-Score:  {test_metrics['f1']:.4f}")
    print(f"ROC-AUC:   {test_metrics['roc_auc']:.4f}")
    print("\nConfusion Matrix:")
    print(test_metrics['confusion_matrix'])
    
    # Feature importance
    print("\n" + "=" * 60)
    print("FEATURE IMPORTANCE")
    print("=" * 60)
    feature_importance = model.feature_importances_
    top_indices = np.argsort(feature_importance)[-10:][::-1]
    for idx, importance in enumerate(feature_importance[top_indices]):
        print(f"Feature {top_indices[idx]:2d}: {importance:.4f}")
    
    # Save model and scaler
    print(f"\n8. Saving model to {output_path}...")
    model_data = {
        'model': model,
        'scaler': scaler,
        'feature_importance': feature_importance
    }
    with open(output_path, 'wb') as f:
        pickle.dump(model_data, f)
    print("   Model saved!")
    
    # Plot results if requested
    if show_plot:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Confusion matrix for test set
        sns.heatmap(test_metrics['confusion_matrix'], annot=True, fmt='d', 
                    cmap='Blues', ax=axes[0, 0])
        axes[0, 0].set_title('Test Set Confusion Matrix')
        axes[0, 0].set_ylabel('True Label')
        axes[0, 0].set_xlabel('Predicted Label')
        
        # Metrics comparison
        metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        train_vals = [train_metrics['accuracy'], train_metrics['precision'], 
                      train_metrics['recall'], train_metrics['f1']]
        test_vals = [test_metrics['accuracy'], test_metrics['precision'], 
                     test_metrics['recall'], test_metrics['f1']]
        
        x = np.arange(len(metrics_names))
        width = 0.35
        axes[0, 1].bar(x - width/2, train_vals, width, label='Train')
        axes[0, 1].bar(x + width/2, test_vals, width, label='Test')
        axes[0, 1].set_ylabel('Score')
        axes[0, 1].set_title('Metrics Comparison')
        axes[0, 1].set_xticks(x)
        axes[0, 1].set_xticklabels(metrics_names, rotation=45)
        axes[0, 1].legend()
        
        # Feature importance
        top_n = 15
        top_idx = np.argsort(feature_importance)[-top_n:][::-1]
        axes[1, 0].barh(range(top_n), feature_importance[top_idx])
        axes[1, 0].set_yticks(range(top_n))
        axes[1, 0].set_yticklabels([f'Feat {i}' for i in top_idx])
        axes[1, 0].set_xlabel('Importance')
        axes[1, 0].set_title('Top 15 Feature Importance')
        
        # Prediction distribution
        axes[1, 1].hist(y_test_proba[y_test == 0, 1], bins=30, alpha=0.6, label='Down (Actual)')
        axes[1, 1].hist(y_test_proba[y_test == 1, 1], bins=30, alpha=0.6, label='Up (Actual)')
        axes[1, 1].set_xlabel('Predicted Probability (Up)')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title('Prediction Probability Distribution')
        axes[1, 1].legend()
        
        plt.tight_layout()
        plt.savefig('model_performance.png', dpi=150, bbox_inches='tight')
        print("\n9. Performance plot saved to model_performance.png")
        plt.show()
    
    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)
    
    return model, scaler


if __name__ == '__main__':
    import sys
    
    csv_file = sys.argv[1] if len(sys.argv) > 1 else 'data/asset_b_train.csv'
    output_file = sys.argv[2] if len(sys.argv) > 2 else 'model_rf.pkl'
    show_plot = '--show-graph' in sys.argv or '--show-plot' in sys.argv
    
    train_model(csv_file, output_file, show_plot=show_plot)
