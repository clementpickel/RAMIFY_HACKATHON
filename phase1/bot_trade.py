import numpy as np
from collections import deque

class AdaptiveRobustTradingBot:
    def __init__(self):
        self.price_window = deque(maxlen=200)
        self.er_window = deque(maxlen=50)
        
    def calculate_allocation(self, prices):
        if len(prices) < 30:
            return 0.5
        
        p = np.array(prices)
        
        returns = np.diff(p) / p[:-1]
        vol_20 = np.std(returns[-20:]) if len(returns) >= 20 else 0.01
        vol_20 = max(vol_20, 0.002)
        
        mom_5 = (p[-1] / p[-6] - 1) if len(p) > 5 else 0
        mom_10 = (p[-1] / p[-11] - 1) if len(p) > 10 else 0
        mom_20 = (p[-1] / p[-21] - 1) if len(p) > 20 else 0
        
        sig_5 = mom_5 / (vol_20 * np.sqrt(5))
        sig_10 = mom_10 / (vol_20 * np.sqrt(10))
        sig_20 = mom_20 / (vol_20 * np.sqrt(20))
        
        composite_signal = (0.5 * sig_5) + (0.3 * sig_10) + (0.2 * sig_20)
        
        change_10 = np.abs(p[-1] - p[-11])
        path_10 = np.sum(np.abs(np.diff(p[-11:])))
        er_10 = change_10 / path_10 if path_10 > 0 else 0
        
        self.er_window.append(er_10)
        avg_er = np.mean(self.er_window) if len(self.er_window) > 10 else 0.5
        
        # Regime Detection
        if avg_er > 0.35:
            # TRENDING ASSET (Asset B)
            # Aggressive Trend Following
            allocation = 1 / (1 + np.exp(-100 * (composite_signal - 0.1)))
            
            # Strict ER filter (Best for Asset B)
            if er_10 > 0.4:
                if composite_signal > 0: allocation = 1.0
                else: allocation = 0.0
        else:
            # CHOPPY ASSET (Asset A)
            # Conservative / Mean Reversion
            # For now, let's just be VERY strict with trend
            if er_10 > 0.6: # Only trade if trend is extremely clear
                allocation = 1 / (1 + np.exp(-20 * (composite_signal - 0.1)))
                if composite_signal > 0: allocation = 1.0
                else: allocation = 0.0
            else:
                allocation = 0.0 # Stay out of chop
            
        return float(np.clip(allocation, 0.0, 1.0))

bot = AdaptiveRobustTradingBot()

def make_decision(epoch: int, price: float):
    bot.price_window.append(price)
    
    if len(bot.price_window) < 30:
        return {'Asset A': 0.5, 'Cash': 0.5}
    
    prices = list(bot.price_window)
    allocation = bot.calculate_allocation(prices)
    
    return {
        'Asset A': float(round(allocation, 6)),
        'Cash': float(round(1 - allocation, 6))
    }
