import numpy as np
from collections import deque

class AdaptiveRobustTradingBot:
    def __init__(self):
        self.price_window = deque(maxlen=200)
        self.last_action = 0.5
        
    def calculate_allocation(self, prices, epoch):
        if len(prices) < 30:
            return 0.5
        
        p = np.array(prices)
        
        # 1. Volatility & Regime
        returns = np.diff(p) / p[:-1]
        vol_20 = np.std(returns[-20:]) if len(returns) >= 20 else 0.01
        # Cap min volatility to avoid division by zero or huge signals
        vol_20 = max(vol_20, 0.002)
        
        # Normalize momentum by volatility (Sharpe-like signal)
        mom_5 = (p[-1] / p[-6] - 1) if len(p) > 5 else 0
        mom_10 = (p[-1] / p[-11] - 1) if len(p) > 10 else 0
        mom_20 = (p[-1] / p[-21] - 1) if len(p) > 20 else 0
        
        # Z-Scores (Signal Strength)
        sig_5 = mom_5 / (vol_20 * np.sqrt(5))
        sig_10 = mom_10 / (vol_20 * np.sqrt(10))
        sig_20 = mom_20 / (vol_20 * np.sqrt(20))
        
        # Aggregate Signal (Trend)
        trend_signal = (0.5 * sig_5) + (0.3 * sig_10) + (0.2 * sig_20)
        
        # Efficiency Ratio (Trend Purity)
        change_10 = np.abs(p[-1] - p[-11])
        path_10 = np.sum(np.abs(np.diff(p[-11:])))
        er_10 = change_10 / path_10 if path_10 > 0 else 0
        
        # Mean Reversion Signal (Bollinger Band / Deviation)
        sma_20 = np.mean(p[-20:])
        dev_20 = (p[-1] - sma_20) / (vol_20 * p[-1]) # Normalized deviation
        
        # Allocation Logic
        allocation = 0.5
        
        if er_10 > 0.25: 
            # TREND REGIME
            # Use sigmoid on trend_signal
            # Steepness 4, offset 0.0
            allocation = 1 / (1 + np.exp(-4 * trend_signal))
            
            # Boost if very clean trend
            if er_10 > 0.5:
                if trend_signal > 0.5: allocation = 1.0
                elif trend_signal < -0.5: allocation = 0.0
                
        else:
            # MEAN REVERSION REGIME (Choppy)
            # Fade the move: if price is high (dev_20 > 0), sell. If low, buy.
            # dev_20 is roughly Z-score of price vs SMA
            
            if dev_20 > 2.0: allocation = 0.0 # Top of band
            elif dev_20 < -2.0: allocation = 1.0 # Bottom of band
            else:
                # Linear mapping between -2 and 2
                # -2 -> 1.0, 2 -> 0.0
                allocation = 0.5 - (dev_20 * 0.25)
        
        # Clip
        allocation = np.clip(allocation, 0.0, 1.0)
        
        return float(allocation)

bot = AdaptiveRobustTradingBot()

def make_decision(epoch: int, price: float):
    global bot
    
    bot.price_window.append(price)
    
    if len(bot.price_window) < 30:
        return {'Asset A': 0.5, 'Cash': 0.5}
    
    prices = list(bot.price_window)
    allocation = bot.calculate_allocation(prices, epoch)
    
    bot.last_action = allocation
    
    return {
        'Asset A': float(round(allocation, 6)),
        'Cash': float(round(1 - allocation, 6))
    }
