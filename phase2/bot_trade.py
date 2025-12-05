import numpy as np
from collections import deque
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class AdvancedTradingBot:
    def __init__(self):
        self.history = []
        self.price_window = deque(maxlen=100)
        
        self.gb_model = None
        self.rf_model = None
        self.scaler = StandardScaler()
        self.trained = False
        
        self.min_train_size = 120
        self.retrain_interval = 200
        self.confidence_threshold = 0.0012
        
        self.feature_cache = deque(maxlen=100)
        
    def calculate_technical_indicators(self, prices):
        if len(prices) < 2:
            return np.zeros(25)
        
        prices_array = np.array(prices)
        features = []
        
        for period in [1, 3, 5, 10, 20]:
            if len(prices_array) > period:
                ret = (prices_array[-1] - prices_array[-period-1]) / prices_array[-period-1]
                features.append(ret)
            else:
                features.append(0)
        
        for period in [5, 10, 20, 50]:
            if len(prices_array) >= period:
                sma = np.mean(prices_array[-period:])
                features.append((prices_array[-1] - sma) / sma)
            else:
                features.append(0)
        
        rsi = self.calculate_rsi(prices_array)
        features.append(rsi / 100.0)
        
        macd, signal = self.calculate_macd(prices_array)
        features.extend([macd, signal, macd - signal])
        
        for period in [5, 10, 20]:
            if len(prices_array) > period:
                returns = np.diff(prices_array[-period-1:]) / prices_array[-period-1:-1]
                vol = np.std(returns)
                features.append(vol)
            else:
                features.append(0)
        
        if len(prices_array) >= 20:
            bb_upper, bb_lower, bb_middle = self.calculate_bollinger_bands(prices_array)
            bb_position = (prices_array[-1] - bb_lower) / (bb_upper - bb_lower + 1e-8)
            bb_width = (bb_upper - bb_lower) / bb_middle
            features.extend([bb_position, bb_width])
        else:
            features.extend([0.5, 0])
        
        features.append(self.calculate_trend_strength(prices_array))
        
        if len(prices_array) >= 50:
            recent_high = np.max(prices_array[-50:])
            recent_low = np.min(prices_array[-50:])
            dist_to_high = (recent_high - prices_array[-1]) / prices_array[-1]
            dist_to_low = (prices_array[-1] - recent_low) / prices_array[-1]
            features.extend([dist_to_high, dist_to_low])
        else:
            features.extend([0, 0])
        
        if len(prices_array) >= 10:
            price_range = (np.max(prices_array[-10:]) - np.min(prices_array[-10:])) / prices_array[-1]
            features.append(price_range)
        else:
            features.append(0)
        
        return np.array(features)
    
    def calculate_rsi(self, prices, period=14):
        if len(prices) < period + 1:
            return 50.0
        
        deltas = np.diff(prices[-period-1:])
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.mean(gains)
        avg_loss = np.mean(losses)
        
        if avg_loss == 0:
            return 100.0
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def calculate_macd(self, prices, fast=12, slow=26, signal=9):
        if len(prices) < slow:
            return 0.0, 0.0
        
        ema_fast = self.ema(prices, fast)
        ema_slow = self.ema(prices, slow)
        macd_line = ema_fast - ema_slow
        
        macd_norm = macd_line / prices[-1]
        signal_norm = macd_norm * 0.85
        
        return macd_norm, signal_norm
    
    def ema(self, prices, period):
        prices = np.array(prices)
        if len(prices) < period:
            return np.mean(prices)
        
        alpha = 2 / (period + 1)
        ema = np.mean(prices[:period])
        
        for price in prices[period:]:
            ema = alpha * price + (1 - alpha) * ema
        
        return ema
    
    def calculate_bollinger_bands(self, prices, period=20, num_std=2):
        if len(prices) < period:
            mid = prices[-1]
            return mid, mid, mid
        
        recent = prices[-period:]
        middle = np.mean(recent)
        std = np.std(recent)
        
        upper = middle + (num_std * std)
        lower = middle - (num_std * std)
        
        return upper, lower, middle
    
    def calculate_trend_strength(self, prices, period=20):
        if len(prices) < period:
            return 0.0
        
        recent = prices[-period:]
        x = np.arange(len(recent))
        
        slope = (len(recent) * np.sum(x * recent) - np.sum(x) * np.sum(recent)) / \
                (len(recent) * np.sum(x**2) - np.sum(x)**2 + 1e-8)
        
        normalized_slope = slope / (np.mean(recent) + 1e-8)
        
        return normalized_slope
    
    def prepare_training_data(self, lookback=30):
        if len(self.history) < self.min_train_size:
            return None, None
        
        prices = [h['price'] for h in self.history]
        X, y = [], []
        
        for i in range(lookback, len(prices) - 5):
            window_prices = prices[max(0, i-100):i]
            features = self.calculate_technical_indicators(window_prices)
            X.append(features)
            
            future_prices = prices[i:i+5]
            future_return = (np.mean(future_prices) - prices[i]) / prices[i]
            y.append(future_return)
        
        if len(X) < 50:
            return None, None
        
        return np.array(X), np.array(y)
    
    def train_models(self):
        X, y = self.prepare_training_data()
        
        if X is None or len(X) < 50:
            return False

        X_scaled = self.scaler.fit_transform(X)
        
        self.gb_model = GradientBoostingRegressor(
            n_estimators=150,
            learning_rate=0.08,
            max_depth=6,
            min_samples_split=8,
            min_samples_leaf=4,
            subsample=0.85,
            random_state=42
        )
        
        self.rf_model = RandomForestRegressor(
            n_estimators=120,
            max_depth=12,
            min_samples_split=8,
            min_samples_leaf=4,
            max_features='sqrt',
            random_state=42,
            n_jobs=-1
        )
        
        self.gb_model.fit(X_scaled, y)
        self.rf_model.fit(X_scaled, y)
        
        self.trained = True
        return True
    
    def predict_next_move(self):
        if not self.trained or len(self.price_window) < 30:
            return 0.0
        
        prices = list(self.price_window)
        features = self.calculate_technical_indicators(prices)
        features_scaled = self.scaler.transform(features.reshape(1, -1))
        
        pred_gb = self.gb_model.predict(features_scaled)[0]
        pred_rf = self.rf_model.predict(features_scaled)[0]
        
        prediction = 0.65 * pred_gb + 0.35 * pred_rf
        
        return prediction
    
    def calculate_position_allocation(self, prediction, epoch=0):
        prices = list(self.price_window)
        
        rsi = self.calculate_rsi(np.array(prices))
        
        if len(prices) >= 20:
            returns = np.diff(prices[-20:]) / np.array(prices[-20:-1])
            volatility = np.std(returns)
        else:
            volatility = 0.01
        
        if len(prices) >= 10:
            momentum = (prices[-1] - prices[-10]) / prices[-10]
        else:
            momentum = 0
        
        if len(prices) >= 20:
            bb_upper, bb_lower, bb_middle = self.calculate_bollinger_bands(np.array(prices))
            bb_pos = (prices[-1] - bb_lower) / (bb_upper - bb_lower + 1e-8)
        else:
            bb_pos = 0.5
        
        base_allocation = 0.5
        
        if abs(prediction) > self.confidence_threshold:
            if prediction > 0:
                signal_strength = min(abs(prediction) * 16, 0.36)
                base_allocation = 0.5 + signal_strength
            else:
                signal_strength = min(abs(prediction) * 16, 0.36)
                base_allocation = 0.5 - signal_strength
        
        if rsi > 75:
            base_allocation = min(base_allocation, 0.55)
        elif rsi > 65 and base_allocation > 0.7:
            base_allocation *= 0.9
        elif rsi < 25:
            base_allocation = max(base_allocation, 0.45)
        elif rsi < 35 and base_allocation < 0.3:
            base_allocation = base_allocation + (0.4 - base_allocation) * 0.3
        
        if volatility > 0.025:
            reduction = min((volatility - 0.025) / 0.025, 0.3)
            base_allocation = 0.5 + (base_allocation - 0.5) * (1 - reduction)
        
        if abs(momentum) > 0.03:
            if momentum > 0 and base_allocation > 0.55:
                base_allocation = min(base_allocation * 1.08, 0.85)
            elif momentum < 0 and base_allocation < 0.45:
                base_allocation = max(base_allocation * 0.92, 0.15)
        
        if bb_pos > 0.95:
            base_allocation = min(base_allocation, 0.5)
        elif bb_pos < 0.05:
            base_allocation = max(base_allocation, 0.5)
        
        if len(prices) >= 50:
            sma_50 = np.mean(prices[-50:])
            if prices[-1] > sma_50 * 1.015 and base_allocation > 0.5:
                base_allocation = min(base_allocation * 1.06, 0.87)
            elif prices[-1] < sma_50 * 0.985 and base_allocation < 0.5:
                base_allocation = max(base_allocation * 0.94, 0.13)
        
        epoch_factor = (epoch % 7) / 100.0
        base_allocation += epoch_factor * 0.01
        
        final_allocation = np.clip(base_allocation, 0.15, 0.85)
        
        return final_allocation

bot = AdvancedTradingBot()

def make_decision(epoch: int, price: float):
    
    bot.history.append({"epoch": epoch, "price": price})
    bot.price_window.append(price)
    
    if len(bot.history) < 30:
        return {'Asset B': 0.5, 'Cash': 0.5}
    
    if not bot.trained and len(bot.history) >= bot.min_train_size:
        bot.train_models()
    
    elif bot.trained and len(bot.history) % bot.retrain_interval == 0:
        bot.train_models()
    
    if not bot.trained:
        prices = list(bot.price_window)
        if len(prices) >= 20:
            sma_short = np.mean(prices[-10:])
            sma_long = np.mean(prices[-20:])
            rsi = bot.calculate_rsi(np.array(prices))
            
            if sma_short > sma_long and rsi < 70:
                allocation = 0.7
            elif sma_short < sma_long and rsi > 30:
                allocation = 0.3
            else:
                allocation = 0.5
        else:
            allocation = 0.5
    else:
        prediction = bot.predict_next_move()
        allocation = bot.calculate_position_allocation(prediction, epoch)
    
    return {
        'Asset B': float(round(allocation, 6)),
        'Cash': float(round(1 - allocation, 6))
    }
