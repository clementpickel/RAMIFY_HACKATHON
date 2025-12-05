import numpy as np
from collections import deque
import random

class HyperAggressiveTradingBot:
    def __init__(self):
        self.price_window = deque(maxlen=150)
        self.last_action = 0.5
        self.trend_strength = 0
        random.seed(None)
        
    def calculate_allocation(self, prices, epoch):
        if len(prices) < 20:
            return 0.5
        
        p = np.array(prices)
        
        # Multi-timeframe SMAs including ultra-short for early detection
        sma_2 = np.mean(p[-2:])
        sma_3 = np.mean(p[-3:])
        sma_4 = np.mean(p[-4:])
        sma_5 = np.mean(p[-5:])
        sma_7 = np.mean(p[-7:])
        sma_10 = np.mean(p[-10:])
        sma_15 = np.mean(p[-15:]) if len(p) >= 15 else sma_10
        sma_20 = np.mean(p[-20:])
        sma_30 = np.mean(p[-30:]) if len(p) >= 30 else sma_20
        sma_50 = np.mean(p[-50:]) if len(p) >= 50 else sma_30
        
        mom_1 = (p[-1] - p[-2]) / p[-2]
        mom_2 = (p[-1] - p[-3]) / p[-3] if len(p) > 2 else 0
        mom_3 = (p[-1] - p[-4]) / p[-4] if len(p) > 3 else 0
        mom_5 = (p[-1] - p[-6]) / p[-6] if len(p) > 5 else 0
        mom_10 = (p[-1] - p[-11]) / p[-11] if len(p) > 10 else 0
        mom_20 = (p[-1] - p[-21]) / p[-21] if len(p) > 20 else 0
        mom_50 = (p[-1] - p[-51]) / p[-51] if len(p) > 50 else 0
        
        returns = np.diff(p[-20:]) / p[-20:-1]
        vol = np.std(returns)
        mean_ret = np.mean(returns)
        
        # Volatility breakout detection
        vol_short = np.std(np.diff(p[-10:]) / p[-10:-1]) if len(p) >= 10 else vol
        vol_ratio = vol_short / vol if vol > 0.001 else 1.0
        vol_expanding = vol_ratio > 1.3  # Volatility expanding - potential breakout
        vol_contracting = vol_ratio < 0.7  # Volatility contracting - compression
        
        # Breakout detection - new highs/lows
        recent_high = np.max(p[-20:])
        recent_low = np.min(p[-20:])
        at_new_high = p[-1] >= recent_high * 0.999
        at_new_low = p[-1] <= recent_low * 1.001
        
        # Price position relative to range
        price_range = recent_high - recent_low
        if price_range > 0:
            range_position = (p[-1] - recent_low) / price_range
        else:
            range_position = 0.5
        
        # RSI calculation
        deltas = np.diff(p[-14:]) if len(p) >= 14 else np.diff(p)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        rsi = 50.0
        if np.mean(losses) > 0:
            rs = np.mean(gains) / np.mean(losses)
            rsi = 100 - (100 / (1 + rs))
        
        allocation = 0.5
        
        # Ultra early trend detection
        ultra_early_up = (sma_2 > sma_3 > sma_4 > sma_5 and mom_1 > 0.002 and mom_2 > 0.004)
        ultra_early_down = (sma_2 < sma_3 < sma_4 < sma_5 and mom_1 < -0.002 and mom_2 < -0.004)
        
        # Hyper trend detection - even more aggressive
        mega_uptrend = (
            sma_2 > sma_3 > sma_5 > sma_7 > sma_10 > sma_15 > sma_20 and
            mom_5 > 0.012 and
            mom_10 > 0.008 and
            mom_1 > 0
        )
        
        mega_downtrend = (
            sma_2 < sma_3 < sma_5 < sma_7 < sma_10 < sma_15 < sma_20 and
            mom_5 < -0.012 and
            mom_10 < -0.008 and
            mom_1 < 0
        )
        
        # Explosive acceleration detection
        accel_1 = mom_1 - mom_2
        accel_2 = mom_2 - mom_3
        explosive_up = (accel_1 > 0.003 and accel_2 > 0.002 and mom_1 > 0.003)
        explosive_down = (accel_1 < -0.003 and accel_2 < -0.002 and mom_1 < -0.003)
        
        # Maximum exposure on strongest signals + breakouts
        if explosive_up or (mega_uptrend and mom_20 > 0.03) or (at_new_high and vol_expanding and mom_5 > 0.01):
            allocation = 1.0
            self.trend_strength = 10
        elif explosive_down or (mega_downtrend and mom_20 < -0.03) or (at_new_low and vol_expanding and mom_5 < -0.01):
            allocation = 0.0
            self.trend_strength = -10
        elif mega_uptrend:
            allocation = 1.0
            self.trend_strength = 8
        elif mega_downtrend:
            allocation = 0.0
            self.trend_strength = -8
        elif ultra_early_up:
            allocation = 0.98
            self.trend_strength = 5
        elif ultra_early_down:
            allocation = 0.02
            self.trend_strength = -5
        
        elif sma_3 > sma_5 > sma_7 > sma_10 > sma_15 > sma_20:
            if mom_5 > 0.01:
                allocation = 0.99
            elif mom_5 > 0.007:
                allocation = 0.96
            elif mom_5 > 0.004:
                allocation = 0.92
            else:
                allocation = 0.88
        
        elif sma_3 < sma_5 < sma_7 < sma_10 < sma_15 < sma_20:
            if mom_5 < -0.01:
                allocation = 0.01
            elif mom_5 < -0.007:
                allocation = 0.04
            elif mom_5 < -0.004:
                allocation = 0.08
            else:
                allocation = 0.12
        
        elif sma_5 > sma_10 > sma_20:
            allocation = 0.88 if mom_10 > 0.005 else 0.78
        
        elif sma_5 < sma_10 < sma_20:
            allocation = 0.12 if mom_10 < -0.005 else 0.22
        
        elif sma_10 > sma_20:
            allocation = 0.80 if mom_10 > 0 else 0.70
        
        elif sma_10 < sma_20:
            allocation = 0.20 if mom_10 < 0 else 0.30
        
        # Ultra-reactive to immediate momentum - EXTREME
        if mom_1 > 0.006:
            allocation = 1.0  # Instant full allocation on strong move
        elif mom_1 > 0.004:
            allocation = min(allocation * 1.35, 1.0)
        elif mom_1 > 0.0025:
            allocation = min(allocation * 1.2, 1.0)
        elif mom_1 > 0.0015:
            allocation = min(allocation * 1.1, 1.0)
        elif mom_1 < -0.006:
            allocation = 0.0  # Instant exit on strong drop
        elif mom_1 < -0.004:
            allocation = max(allocation * 0.65, 0.0)
        elif mom_1 < -0.0025:
            allocation = max(allocation * 0.8, 0.0)
        elif mom_1 < -0.0015:
            allocation = max(allocation * 0.9, 0.0)
        
        # Multi-level momentum confirmation with compound boost
        if mom_2 > 0.006 and mom_3 > 0.008 and mom_5 > 0.01:
            allocation = 1.0  # Triple confirmation = max allocation
        elif mom_2 > 0.004 and mom_3 > 0.006:
            allocation = min(allocation * 1.3, 1.0)
        elif mom_2 > 0.003 and mom_3 > 0.004:
            allocation = min(allocation * 1.18, 1.0)
        elif mom_2 < -0.006 and mom_3 < -0.008 and mom_5 < -0.01:
            allocation = 0.0  # Triple negative = full exit
        elif mom_2 < -0.004 and mom_3 < -0.006:
            allocation = max(allocation * 0.7, 0.0)
        elif mom_2 < -0.003 and mom_3 < -0.004:
            allocation = max(allocation * 0.82, 0.0)
        
        if rsi > 85 and allocation > 0.85:
            allocation *= 0.95
        elif rsi < 15 and allocation < 0.15:
            allocation = allocation + (0.5 - allocation) * 0.3
        
        # Volatility boost - contracting vol = bigger moves coming
        if vol_contracting and abs(allocation - 0.5) > 0.3:
            allocation = 0.5 + (allocation - 0.5) * 1.5
        elif vol < 0.012 and abs(allocation - 0.5) > 0.2:
            allocation = 0.5 + (allocation - 0.5) * 1.8
        
        # Range position amplification
        if range_position > 0.85 and mom_5 > 0.008:  # High in range + momentum up
            allocation = min(allocation * 1.2, 1.0)
        elif range_position < 0.15 and mom_5 < -0.008:  # Low in range + momentum down
            allocation = max(allocation * 0.8, 0.0)
        
        if len(p) >= 50:
            if p[-1] > sma_50 * 1.015 and mom_50 > 0.02:
                allocation = min(allocation * 1.35, 1.0)
            elif p[-1] < sma_50 * 0.985 and mom_50 < -0.02:
                allocation = max(allocation * 0.65, 0.0)
        
        if mean_ret > 0.0005 and allocation > 0.5:
            allocation = min(allocation * 1.15, 1.0)
        elif mean_ret < -0.0005 and allocation < 0.5:
            allocation = max(allocation * 0.85, 0.0)
        
        # Enhanced acceleration response
        if len(p) >= 30:
            accel = mom_5 - mom_10
            accel_short = mom_3 - mom_5
            
            if accel > 0.015 and accel_short > 0.005:
                allocation = min(allocation * 1.35, 1.0)
            elif accel > 0.01 and allocation > 0.5:
                allocation = min(allocation * 1.25, 1.0)
            elif accel > 0.005:
                allocation = min(allocation * 1.12, 1.0)
            elif accel < -0.015 and accel_short < -0.005:
                allocation = max(allocation * 0.65, 0.0)
            elif accel < -0.01 and allocation < 0.5:
                allocation = max(allocation * 0.75, 0.0)
            elif accel < -0.005:
                allocation = max(allocation * 0.88, 0.0)
        
        noise = ((epoch * 7 + 13) % 19) / 300.0
        allocation += noise * 0.015
        
        return np.clip(allocation, 0.0, 1.0)

bot = HyperAggressiveTradingBot()

def make_decision(epoch: int, price: float):
    global bot
    
    bot.price_window.append(price)
    
    if len(bot.price_window) < 20:
        return {'Asset B': 0.5, 'Cash': 0.5}
    
    prices = list(bot.price_window)
    allocation = bot.calculate_allocation(prices, epoch)
    
    bot.last_action = allocation
    
    return {
        'Asset B': float(round(allocation, 6)),
        'Cash': float(round(1 - allocation, 6))
    }
