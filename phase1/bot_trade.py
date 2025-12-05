import numpy as np
from collections import deque
import random

class MeanReversionTradingBot:
    def __init__(self):
        self.price_window = deque(maxlen=100)
        self.last_action = 0.5
        random.seed(None)
        
    def calculate_allocation(self, prices, epoch):
        if len(prices) < 20:
            return 0.7  # Start with slight bias to asset
        
        p = np.array(prices)
        
        # Simple trend detection
        sma_5 = np.mean(p[-5:])
        sma_10 = np.mean(p[-10:])
        sma_20 = np.mean(p[-20:])
        sma_50 = np.mean(p[-50:]) if len(p) >= 50 else sma_20
        
        # Momentum
        mom_5 = (p[-1] - p[-6]) / p[-6] if len(p) > 5 else 0
        mom_10 = (p[-1] - p[-11]) / p[-11] if len(p) > 10 else 0
        
        # Base: mostly invested since asset has positive return
        allocation = 0.8
        
        # Slight adjustments based on short-term trend
        if sma_5 > sma_10 > sma_20:
            allocation = 0.95
        elif sma_5 > sma_10:
            allocation = 0.9
        elif sma_5 < sma_10 < sma_20:
            allocation = 0.6
        elif sma_5 < sma_10:
            allocation = 0.7
        
        # Momentum boost
        if mom_5 > 0.01:
            allocation = min(allocation * 1.1, 1.0)
        elif mom_5 < -0.01:
            allocation = max(allocation * 0.9, 0.5)
        
        # Longer-term trend
        if len(p) >= 50 and p[-1] > sma_50:
            allocation = min(allocation * 1.05, 1.0)
        elif len(p) >= 50 and p[-1] < sma_50:
            allocation = max(allocation * 0.95, 0.6)
        
        # Noise
        noise = ((epoch * 11 + 17) % 23) / 400.0
        allocation = np.clip(allocation + noise - 0.03, 0.0, 1.0)
        
        return float(allocation)

bot = MeanReversionTradingBot()

def make_decision(epoch: int, price: float):
    global bot
    
    bot.price_window.append(price)
    
    if len(bot.price_window) < 20:
        return {'Asset A': 0.5, 'Cash': 0.5}
    
    prices = list(bot.price_window)
    allocation = bot.calculate_allocation(prices, epoch)
    
    bot.last_action = allocation
    
    return {
        'Asset A': float(round(allocation, 6)),
        'Cash': float(round(1 - allocation, 6))
    }
