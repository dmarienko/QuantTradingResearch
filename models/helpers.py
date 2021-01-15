import pandas as pd
import numpy as np
from datetime import datetime
import copy
import matplotlib.pyplot as plt

from tools.utils.utils import mstruct, red, green, yellow, blue, magenta, cyan, white
from tools.charting.plot_helpers import sbp
from tools.analysis.data import retain_columns_and_join
from tools.analysis.tools import scols, srows, ohlc_resample, roll


def tracking_error(benchmark, tracker, mode='returns'):
    """
    Tracking error in percents
    """
    # we want to compare only common date intervals so bit filtering here
    if mode.startswith('price'):
        # converting to returns
        f = pd.concat((benchmark.pct_change(), tracker.pct_change()), axis=1, keys=['X', 'Y']).dropna()
    else:
        # data already contains returns so not need to convert it
        f = pd.concat((benchmark, tracker), axis=1, keys=['X', 'Y']).dropna()
    return 100 * np.std(f.X - f.Y, ddof=1)


def prices_to_returns(prices):
    return prices.pct_change()[1:] # drop first inf


def returns_to_prices(rets):
    return (rets + 1).cumprod() - 1


def norm(x):
    return x / x.iloc[0]


class Model:
    """
    Abstract class for any tracking models
    """
    def __init__(self, description=None):
        self.description = description
    
    def fit(self, x, y, **kwargs):
        return self
    
    def predict(self, x, y=None, **kwargs):
        return None
    

class TrackingModel:
    def __init__(self, data, index_name, train_date):
        self.data = data
        self.closes = retain_columns_and_join(data, 'close')
        self.index = data[index_name]
        self.index_closes = self.index.close
        self.X_price = self.closes[self.closes.columns[~self.closes.columns.str.match(index_name)]]
        self.Y_price = self.closes[index_name]
        self.X_ret, self.Y_ret = prices_to_returns(self.X_price), prices_to_returns(self.Y_price)
        self.train_date = pd.Timestamp(train_date)
        self.selection = None
        
    def select(self, selection):
        n = copy.copy(self)
        n.selection = set(n.data.keys()) & set(selection)
        return n
        
    def get_data(self, mode='prices', where='all data'):
        x, y = (self.X_price, self.Y_price) if mode.startswith('price') else (self.X_ret, self.Y_ret)
        if where.startswith('train'):
            x, y = x[:self.train_date], y[:self.train_date]
        elif where.startswith('test'):
            x, y = x[self.train_date:], y[self.train_date:]
        return (x[self.selection], y) if self.selection is not None else (x, y)
        
    def train_set(self, mode='prices'):
        return self.get_data(mode, 'train')
    
    def test_set(self, mode='prices'):
        return self.get_data(mode, 'test')
    
    def estimate(self, tracker: Model, on='prices', **kwargs):
        xn, yn = self.train_set(mode=on)
        m = tracker.fit(xn, yn)
        yn_h = m.predict(xn, y=yn, **kwargs) if isinstance(m, Model) else m.predict(xn, **kwargs)
        yn_h = pd.Series(yn_h, index=xn.index) if isinstance(yn_h, np.ndarray) else yn_h
        
        xt, yt = self.test_set(mode=on)
        yt_h = m.predict(xt, y=yt, **kwargs) if isinstance(m, Model) else m.predict(xt, **kwargs)
        yt_h = pd.Series(yt_h, index=xt.index) if isinstance(yt_h, np.ndarray) else yt_h
        
        return mstruct(
            model=m,
            description=tracker.description,
            mode=on,
            train = mstruct(
                x = xn, y = yn, yh = yn_h, w = tracking_error(yn, yn_h, mode=on)
            ),
            test = mstruct(
                x = xt, y = yt, yh = yt_h, w = tracking_error(yt, yt_h, mode=on)
            )
        )

    
def plot_results(m: mstruct):
    """
    Plot results for train/test periods with it's tracking errors
    """
    yn, yn_h = (m.train.y, m.train.yh) if m.mode.startswith('price') else (norm(returns_to_prices(m.train.y)), norm(returns_to_prices(m.train.yh)))
    yt, yt_h = (m.test.y, m.test.yh) if m.mode.startswith('price') else (norm(returns_to_prices(m.test.y)), norm(returns_to_prices(m.test.yh)))
    sbp(13, 1, c=2);
    plt.plot(yn, lw=1, c='blue', label='Index')
    plt.plot(yn_h, lw=1, ls='--', c='g', label='Tracking Model')
    plt.legend(loc=2)
    plt.title(f'{m.description}: InSample: $\omega$: {m.train.w: 0.2f}')
    
    sbp(13, 3, c=1);
    plt.plot(yt, lw=1, c='blue', label='Index')
    plt.plot(yt_h, lw=1, ls='--', c='g', label='Tracking Model')
    plt.legend(loc=2)
    plt.title(f'{m.description}: OutOfSample: $\omega$: {m.test.w: 0.2f}')
    print(yellow(f' -> TE (train): {m.train.w:.2f}%'), ' | ', green(f'TE (test): {m.test.w:.2f}%'))
    
    
def arbitrage_trading(kfm, Z_entry, Z_exit, traded_size):
    """
    Simple arbitrage strategy for KF
    """
    
    from ira.simulator.utils import shift_signals
    # z-score
    z = kfm.model.err / kfm.model.var
    
    # take position on weighted portfolio
    shorts, longs = z >= Z_entry, z <= -Z_entry
    shorts_exits, longs_exits = ((z <= -Z_exit) & (z > -Z_entry)), ((z >= Z_exit) & (z < Z_entry))
    
    # we need to normalize portfolio weights
    b_a_w = kfm.model.b.div(kfm.model.b.sum(axis=1), axis=0)
    
    signals = pd.concat((
         -b_a_w[longs],  b_a_w[longs_exits] * 0, 
          b_a_w[shorts], b_a_w[shorts_exits] * 0,
    ), axis=0).sort_index()

    # take opposite position on Index
    signals['^RUI'] = 0
    signals.loc[longs, '^RUI'] = +1
    signals.loc[shorts, '^RUI'] = -1
    
    # we will trade 1000 shares per stock and execute on daily close
    signals = shift_signals(np.floor(traded_size * signals), hours=15, minutes=59)
    return signals.astype(int)
    