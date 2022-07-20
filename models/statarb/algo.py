import pandas as pd
import numpy as np
import statsmodels.api as sm

from ira.analysis.timeseries import infer_series_frequency
from ira.analysis.kalman import kalman_regression_estimator
from ira.strategies.helpers import generate_bands_signals
from ira.simulator.utils import shift_signals
from ira.analysis.tools import scols

from utils import ksmooth, merge_columns_by_op
from typing import Union


class PairsStrategy:
    def __init__(self, data: pd.DataFrame, sX: str, sY: str, period: int):
        self.data = data
        self.sX = sX
        self.sY = sY
        self.period = period
        self.timeframe = infer_series_frequency(data[:100])

    def positions(self, index, price, direction):
        px, py = price
        b = self.data.beta.iat[index]
        # -b*x, y
        return [-direction * b, +direction]
    
    def zscore(self, xs, period):
        m = xs.rolling(window=period).mean()
        s = xs.rolling(window=period).std()
        return (xs - m) / s
    
    def get_signals(self, entry, exit, period=None, accurate_time=True):
        priceX, priceY = self.data[self.sX], self.data[self.sY]
        signals = generate_bands_signals(
            scols(priceX, priceY), 
            self.zscore(self.data.spread, self.period if period is None else period), 
            entry, 
            exit, 
            size_func=self.positions
        )
        return shift_signals(signals, self.timeframe - pd.Timedelta('1s')) if accurate_time else signals
    
    
class PairsPreparation:
    """
    Class for preparation pairs model
    """
    def __init__(self, closes: pd.DataFrame, end_of_train: Union[str, pd.Timestamp]):
        self.closes = closes
        self.end_of_train = end_of_train
        
    def half_life(self, xs, min_period=5):
        xs_lag = xs.shift(1).bfill()
        xs_ret = xs.diff().bfill()
        res = sm.OLS(xs_ret, sm.add_constant(xs_lag)).fit()
        return max(int(-np.log(2) / res.params[1]), min_period)

    def get_trader(self, smbX: str, smbY: str, delta=1e-3, pv=0.01, mv=1):
        x, y = self.closes[smbX], self.closes[smbY]
        xa = ksmooth(x, pv, mv).x
        ya = ksmooth(y, pv, mv).x
        gamma = delta / (1 - delta)
        r = kalman_regression_estimator(xa, ya, gamma, 1, False)

        df = scols(x, y)
        beta = pd.Series(r[0][0], index=df.index)
        df['beta'] = beta
        df['spread'] = df[smbY] - (df[smbX] * beta)
        smoothing_period = self.half_life(df['spread'][:self.end_of_train])
        return PairsStrategy(df, smbX, smbY, smoothing_period)
