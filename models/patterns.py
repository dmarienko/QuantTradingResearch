import pandas as pd
import numpy as np
from ira.analysis.timeseries import find_movements


def movements_tail_corrected(h, percentage):
    trends = find_movements(h.close, np.inf, use_prev_movement_size_for_percentage=False,
                            pcntg=percentage/100, t_window=np.inf, 
                            drop_weekends_crossings=True, drop_out_of_market=False, result_as_frame=True, silent=True)
    # attach tail
    u, d = trends.UpTrends.dropna(), trends.DownTrends.dropna()
    t_ends = [u.end[-1], d.end[-1]]
    n_last = np.argmax(t_ends)
    _empt = {'start_price':np.nan, 'end_price':np.nan, 'delta':np.nan, 'end':np.nan}
    e0, t0 = h.close[-1], h.index[-1]

    if n_last == 0: # last is uptrend
        s0 = u.end_price[-1]
        _y = {'start_price': s0, 'end_price': e0, 'delta':abs(s0-e0), 'end':t0}
        _x = _empt
    else: # last is downtrend
        s0 = d.end_price[-1]
        _x = {'start_price': s0, 'end_price': e0, 'delta':abs(s0-e0), 'end':t0}
        _y = _empt


    _r = pd.concat((
        pd.DataFrame({t_ends[n_last]: _x}).T,
        pd.DataFrame({t_ends[n_last]: _y}).T
    ), axis=1, keys=['UpTrends', 'DownTrends'])
    return pd.concat((trends, _r), axis=0)


def piecewise_linear(date, h5, threshold, normalize=True, _tf=pd.Timedelta('5Min')):
    t = pd.Timestamp(date) if isinstance(date, str) else date
    if isinstance(t, (list, tuple)):
        dh = pd.Timestamp(t[1]) - pd.Timestamp(t[0])
        h = h5[pd.Timestamp(t[0]) : pd.Timestamp(t[1]) ]
    else: 
        dh = pd.Timedelta('24H')
        h = h5[t : t + dh]
    
    trends = movements_tail_corrected(h, threshold)
    
    pw = pd.concat((trends.UpTrends.dropna(), trends.DownTrends.dropna()), axis=0).sort_index()
    d0 = pd.Timestamp(pw.index[0])

    x0 = (pw.index - d0).values / _tf
    x1 = (pw.end.astype('datetime64[ns]') - d0).values / _tf
    y0, y1 = pw.start_price.values, pw.end_price
    
    n_max = int(dh / _tf)
    j = 0
    yl = {}
    for i in range(n_max + 1):
        yl[d0 + i * _tf] = np.interp(i, [x0[j], x1[j]], [y0[j], y1[j]])
        if i + 1 > x1[j]:
            j += 1
    yl = pd.Series(yl, name='y')
    return yl / yl.iloc[0] if normalize else yl

