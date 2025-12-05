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
        
        # Normalize momentum by volatility (Sharpe-like signal)
        mom_5 = (p[-1] / p[-6] - 1) if len(p) > 5 else 0
        mom_10 = (p[-1] / p[-11] - 1) if len(p) > 10 else 0
        mom_20 = (p[-1] / p[-21] - 1) if len(p) > 20 else 0
        
        # Z-Scores (Signal Strength)
        # We expect momentum to be roughly proportional to vol * sqrt(time)
        # Signal = Momentum / (Vol * sqrt(Period))
        sig_5 = mom_5 / (vol_20 * np.sqrt(5)) if vol_20 > 0 else 0
        sig_10 = mom_10 / (vol_20 * np.sqrt(10)) if vol_20 > 0 else 0
        sig_20 = mom_20 / (vol_20 * np.sqrt(20)) if vol_20 > 0 else 0
        
        # Aggregate Signal (Weighted average of timeframes)
        # Giving more weight to recent momentum but confirming with longer term
        composite_signal = (0.5 * sig_5) + (0.3 * sig_10) + (0.2 * sig_20)
        
        # Efficiency Ratio (Trend Purity)
        # ER = |Price_change| / Sum(|Individual_changes|)
        change_10 = np.abs(p[-1] - p[-11])
        path_10 = np.sum(np.abs(np.diff(p[-11:])))
        er_10 = change_10 / path_10 if path_10 > 0 else 0
        
        # Allocation Logic
        # Map composite signal to 0-1. 
        # A Z-score of +2 is strong buy, -2 is strong sell.
        # We want to be aggressive, so maybe +1 is enough for full allocation.
        
        base_allocation = 0.5 + (composite_signal * 0.25) # -2 -> 0.0, +2 -> 1.0 roughly
        
        # Boost with Efficiency Ratio if trend is aligned
        if composite_signal > 0.5 and er_10 > 0.6:
            base_allocation += 0.3 # Strong clean uptrend
        elif composite_signal < -0.5 and er_10 > 0.6:
            base_allocation -= 0.3 # Strong clean downtrend
            
        # Volatility Regime Adjustment
        # If vol is very low, we can leverage up (in this context, go to 1.0 or 0.0 faster)
        # If vol is very high, we might want to be careful, OR ride the crash/pump.
        # For a hackathon bot, usually high vol = opportunity if direction is right.
        
        # Let's use a sigmoid-like function for smoother transition
        # Steepness 8, offset 0.1 for more aggression
        allocation = 1 / (1 + np.exp(-8 * (composite_signal - 0.1))) # Shifted sigmoid
        
        # Apply ER filter
        if er_10 > 0.4: # Lower threshold for clean trend
            if composite_signal > 0: allocation = 1.0
            else: allocation = 0.0
            
        # Clip
        allocation = np.clip(allocation, 0.0, 1.0)
        
        return float(allocation)

bot = AdaptiveRobustTradingBot()

def make_decision(epoch: int, price: float):
    global bot
    
    bot.price_window.append(price)
    
    if len(bot.price_window) < 30:
        return {'Asset B': 0.5, 'Cash': 0.5}
    
    prices = list(bot.price_window)
    allocation = bot.calculate_allocation(prices, epoch)
    
    bot.last_action = allocation
    
    return {
        'Asset B': float(round(allocation, 6)),
        'Cash': float(round(1 - allocation, 6))
    }
