import numpy as np
from collections import deque

p = {'A': deque(maxlen=150), 'B': deque(maxlen=150)}

def make_decision(epoch, priceA, priceB):
    p['A'].append(priceA)
    p['B'].append(priceB)
    if len(p['A']) < 30: return {'Asset A': 0.0, 'Asset B': 0.0, 'Cash': 1.0}
    
    best_asset = None
    best_score = -999
    
    for asset, prices in [('A', p['A']), ('B', p['B'])]:
        arr = np.array(prices)
        ret = np.diff(arr) / arr[:-1]
        vol = max(np.std(ret[-20:]), 0.001)
        
        m2 = (arr[-1] / arr[-3] - 1) / (vol * 1.414)
        m3 = (arr[-1] / arr[-4] - 1) / (vol * 1.732)
        m5 = (arr[-1] / arr[-6] - 1) / (vol * 2.236)
        m10 = (arr[-1] / arr[-11] - 1) / (vol * 3.162)
        m20 = (arr[-1] / arr[-21] - 1) / (vol * 4.472)
        
        signal = 0.35 * m2 + 0.3 * m3 + 0.2 * m5 + 0.1 * m10 + 0.05 * m20
        
        change = abs(arr[-1] - arr[-11])
        path = sum(abs(np.diff(arr[-11:])))
        er = change / path if path > 0 else 0
        
        recent_trend = (arr[-1] / arr[-4] - 1)
        vol_boost = 1 / (vol + 0.01)
        
        score = signal * (er ** 0.6) * (1 + abs(recent_trend) * 15) * (vol_boost ** 0.3)
        
        if score > best_score:
            best_score = score
            best_asset = asset
    
    if best_score > 0.03 and best_asset:
        return {'Asset A': 1.0 if best_asset == 'A' else 0.0, 
                'Asset B': 1.0 if best_asset == 'B' else 0.0, 
                'Cash': 0.0}
    return {'Asset A': 0.0, 'Asset B': 0.0, 'Cash': 1.0}


