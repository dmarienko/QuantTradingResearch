from typing import Union, Tuple, List
import types

import numpy as np
import pandas as pd
import statsmodels.api as sm
from collections import OrderedDict

from statsmodels.regression.linear_model import OLS
from datetime import timedelta
from .tools import (
        column_vector, shift, sink_nans_down,
        lift_nans_up, nans, rolling_sum, isscalar, apply_to_frame, ohlc_resample
        )


try:
    from numba import njit
except:
    print('numba package is not found !')

    def njit(f):
        return f


def __wrap_dataframe_decorator(func):
    def wrapper(*args, **kwargs):
        if isinstance(args[0], (pd.Series, pd.DataFrame)):
            return apply_to_frame(func, *args, **kwargs)
        else:
            return func(*args)

    return wrapper


def __empty_smoother(x, *args, **kwargs):
    return column_vector(x)


def smooth(x, stype: Union[str, types.FunctionType], *args, **kwargs) -> pd.Series:
    """
    Smooth series using either given function or find it by name from registered smoothers
    """
    smoothers = {'sma': sma, 'ema': ema, 'tema': tema, 'dema': dema, 'zlema': zlema, 'kama': kama}

    f_sm = __empty_smoother
    if isinstance(stype, str):
        if stype in smoothers:
            f_sm = smoothers.get(stype)
        else:
            raise ValueError("Smoothing method '%s' is not supported !" % stype)

    if isinstance(stype, types.FunctionType):
        f_sm = stype

    # smoothing
    x_sm = f_sm(x, *args, **kwargs)

    return x_sm if isinstance(x_sm, pd.Series) else pd.Series(x_sm.flatten(), index=x.index)


def find_movements(x, threshold, pcntg=0.75, t_window: Union[List, Tuple, int]=10,
                   drop_out_of_market=True,
                   drop_weekends_crossings=True,
                   silent=False,
                   use_prev_movement_size_for_percentage=True,
                   result_as_frame=False, collect_log=False):
    """
    Tries to find all movements in timeseies x (should be pandas Series object) which have absolute magnitude >= threshold
    and lasts not more than t_window bars.
    If need to drop all movements covering out of market time (from 16:00 till 9:30 next day) set drop_out_of_market to True

    # Example:
    # -----------------

    import pandas as pd
    import numpy as np
    import matplotlib
    import matplotlib.pyplot as plt
    from pylab import *

    z = 50 + np.random.normal(0, 0.2, 1000).cumsum()
    x = pd.Series(z, index=pd.date_range('1/1/2000 16:00:00', periods=len(z), freq='30s'))

    i_drops, i_grows, _, _ = find_movements(x, threshold=1, t_window=120, pcntg=.75)

    plt.figure(figsize=(15,10))

    # plot series
    plt.plot(x)

    # plot movements
    plt.plot(x.index[i_drops].T, x[i_drops].T, 'r--', lw=1.2);
    plt.plot(x.index[i_grows].T, x[i_grows].T, 'w--', lw=1.2);

    # or new version (after 2018-08-31)
    trends = find_movements(x, threshold=1, t_window=120, pcntg=.75, result_as_indexes=False)
    u, d = trends.UpTrends.dropna(), trends.DownTrends.dropna()
    plt.plot([u.index, u.end], [u.start_price, u.end_price], 'w--', lw=0.7, marker='.', markersize=5);
    plt.plot([d.index, d.end], [d.start_price, d.end_price], 'r--', lw=0.7);

    plt.draw()
    plt.show()

    # -----------------

    :param x: pandas Series object
    :param threshold: movement minimal magnitude threshold
    :param pcntg: percentage of previous movement (if use_prev_movement_size_for_percentage is True) that considered as start of new movement (1 == 100%)
    :param use_prev_movement_size_for_percentage: False if use percentage from previous price extremum (otherwise it uses prev. movement) [True]
    :param t_window: movement's length filter in bars or range: 120 or (0, 100) or (100, np.inf) etc
    :param drop_out_of_market: True if need to drop movements between sessions
    :param drop_weekends_crossings: True if need to drop movemets crossing weekends (for intraday data)
    :param silent: if True it doesn't show progress bar [False by default]
    :param result_as_frame: if False (default) result returned as tuple of indexes otherwise as DataFrame
    :param collect_log: True if need to collect track of tops/bottoms at times when they appeared
    :return: tuple with indexes of (droping movements, growing movements, droping magnitudes, growing magnitudes)
    """

    # check input arguments
    if not isinstance(x, pd.Series):
        x = pd.Series(x)

    # drop nans (not sure about 0 as replacement)
    if x.hasnans:
        x = x.fillna(0)

    mi, mx, direction = 0, 0, 0
    i_drops, i_grows = [], []
    log_rec = OrderedDict()
    timeline = x.index

    # check filter values
    if isscalar(t_window):
        t_window = [0, t_window]
    elif len(t_window) != 2 or t_window[0] >= t_window[1]:
        raise ValueError("t_window must have 2 ascending elements")

    if not silent: print(' -[', end='')
    n_p_len = max(int(len(x) / 100), 1)

    for i in range(1, len(x)):
        v = x.iat[i]

        if direction <= 0:
            if v < x.iat[mi]:
                mi = i
                direction = -1
            else:
                # floating up
                if use_prev_movement_size_for_percentage:
                    l_mv = pcntg * (x.iat[mx] - x.iat[mi])
                else:
                    l_mv = pcntg * x.iat[mi]

                # check condition    
                if (v - x.iat[mi] >= threshold) or (l_mv < v - x.iat[mi]):
                    i_drops.append([mx, mi])
                    if collect_log:
                        log_rec[timeline[i]] = {'Type': '-', 'Time': timeline[mi], 'Price': x.iat[mi]}
                    mx = i
                    direction = 1

        if direction >= 0:
            if v > x.iat[mx]:
                mx = i
                direction = +1
            else:
                if use_prev_movement_size_for_percentage:
                    l_mv = pcntg * (x.iat[mx] - x.iat[mi])
                else:
                    l_mv = pcntg * x.iat[mx]

                if (x.iat[mx] - v >= threshold) or (l_mv < x.iat[mx] - v):
                    i_grows.append([mi, mx])
                    if collect_log:
                        log_rec[timeline[i]] = {'Type': '+', 'Time': timeline[mx], 'Price': x.iat[mx]}
                    mi = i
                    direction = -1

        if not silent and not (i % n_p_len): print(':', end='')

    if not silent: print(']-')
    i_drops = np.array(i_drops)
    i_grows = np.array(i_grows)

    # Nothing is found 
    if len(i_drops) == 0 or len(i_grows) == 0: 
        if not silent:
            print("\n\t[WARNING] find_movements: No trends found for given conditions !")
        return pd.DataFrame({'UpTrends':[], 'DownTrends':[]}) if result_as_frame else ([], [], [], [])

    # retain only movements equal or exceed specified threshold
    if not np.isinf(threshold):
        if i_drops.size:
            i_drops = i_drops[abs(x[i_drops[:, 1]].values - x[i_drops[:, 0]].values) >= threshold, :]
        if i_grows.size:
            i_grows = i_grows[abs(x[i_grows[:, 1]].values - x[i_grows[:, 0]].values) >= threshold, :]

    # retain only movements which shorter than specified window
    __drops_len = abs(i_drops[:, 1] - i_drops[:, 0])
    __grows_len = abs(i_grows[:, 1] - i_grows[:, 0])
    if i_drops.size: i_drops = i_drops[(__drops_len >= t_window[0]) & (__drops_len <= t_window[1]), :]
    if i_grows.size: i_grows = i_grows[(__grows_len >= t_window[0]) & (__grows_len <= t_window[1]), :]

    # filter out all movements which cover period from 16:00 till 9:30 next day
    if drop_out_of_market and (isinstance(x, pd.Series) and isinstance(x.index, pd.DatetimeIndex)):
        if i_drops.size:
            h = x.index.hour[i_drops]
            i_drops = i_drops[~((h[:, 0] <= 16) & (h[:, 1] >= 9))]
        if i_grows.size:
            h = x.index.hour[i_grows]
            i_grows = i_grows[~((h[:, 0] <= 16) & (h[:, 1] >= 9))]

    # drop crossed weekend if required (we would not want to drop them when use daily prices)
    # drop if start < Sunday and end is Sunday. Drop if start and end are different weeks and start is not Sunday.
    if drop_weekends_crossings:
        if i_drops.size and (isinstance(x, pd.Series) and isinstance(x.index, pd.DatetimeIndex)):
            d = x.index.dayofweek[i_drops]
            w = x.index.week[i_drops]
            i_drops = i_drops[~(((d[:, 0] < 6) & (d[:, 1] == 6)) | (w[:,0] != w[:,1]) & (d[:, 0] != 6))]
        if i_grows.size and (isinstance(x, pd.Series) and isinstance(x.index, pd.DatetimeIndex)):
            d = x.index.dayofweek[i_grows]
            w = x.index.week[i_grows]
            i_grows = i_grows[~(((d[:, 0] < 6) & (d[:, 1] == 6)) | (w[:,0] != w[:,1]) & (d[:, 0] != 6))]

    # drops and grows magnitudes
    v_drops = []
    if i_drops.size:
        v_drops = abs(x[i_drops[:, 1]].values - x[i_drops[:, 0]].values)

    v_grows = []
    if i_grows.size:
        v_grows = abs(x[i_grows[:, 1]].values - x[i_grows[:, 0]].values)

    # how to return results
    if not result_as_frame:
        # just raw indexes (by default)
        return i_drops, i_grows, v_drops, v_grows
    else:
        i_d, i_g = x.index[i_drops], x.index[i_grows]
        x_d, x_g = x[i_drops], x[i_grows]

        d = pd.DataFrame(OrderedDict({
            'start_price': x_d[:, 0],
            'end_price': x_d[:, 1],
            'delta': v_drops,
            'end': i_d[:, 1]
        }), index=i_d[:, 0])

        g = pd.DataFrame(OrderedDict({
            'start_price': x_g[:, 0],
            'end_price': x_g[:, 1],
            'delta': v_grows,
            'end': i_g[:, 1]
        }), index=i_g[:, 0])

        trends = pd.concat((g, d), axis=1, keys=['UpTrends', 'DownTrends'])
        if collect_log:
            return trends, pd.DataFrame.from_dict(log_rec, orient='index')
        
        return trends


def infer_series_frequency(series):
    """
    Infer frequency of given timeseries

    :param series: Series, DataFrame or DatetimeIndex object
    :return: timedelta for found frequency
    """

    if not isinstance(series, (pd.DataFrame, pd.Series, pd.DatetimeIndex)):
        raise ValueError("infer_series_frequency> Only DataFrame, Series of DatetimeIndex objects are allowed")

    times_index = (series if isinstance(series, pd.DatetimeIndex) else series.index).to_pydatetime()
    if times_index.shape[0] < 2:
        raise ValueError("Series must have at least 2 points to determ frequency")

    values = np.array(sorted([(x.total_seconds()) for x in np.diff(times_index)]))
    diff = np.concatenate(([1], np.diff(values)))
    idx = np.concatenate((np.where(diff)[0], [len(values)]))
    freqs = dict(zip(values[idx[:-1]], np.diff(idx)))
    return timedelta(seconds=max(freqs, key=freqs.get))


def running_view(arr, window, axis=-1):
    """
    Produces running view (lagged matrix) from given array.

    Example:

    > running_view(np.array([1,2,3,4,5,6]), 3)

    array([[1, 2, 3],
           [2, 3, 4],
           [3, 4, 5],
           [4, 5, 6]])

    :param arr: array of numbers
    :param window: window length
    :param axis:
    :return: lagged matrix
    """
    shape = list(arr.shape)
    shape[axis] -= (window-1)
    return np.lib.index_tricks.as_strided(arr, shape + [window], arr.strides + (arr.strides[axis],))


def detrend(y, order):
    """
    Removes linear trend from the series y.
    detrend computes the least-squares fit of a straight line to the data
    and subtracts the resulting function from the data.

    :param y:
    :param order:
    :return:
    """
    if order == -1: return y
    return OLS(y, np.vander(np.linspace(-1, 1, len(y)), order + 1)).fit().resid


def moving_detrend(y, order, window):
    """
    Removes linear trend from the series y by using sliding window.
    :param y: series (ndarray or pd.DataFrame/Series)
    :param order: trend's polinome order
    :param window: sliding window size
    :return: (residual, rsquatred, betas)
    """
    yy = running_view(column_vector(y).T[0], window=window)
    n_pts = len(y)
    resid = nans((n_pts))
    r_sqr = nans((n_pts))
    betas = nans((n_pts, order + 1))
    for i, p in enumerate(yy):
        n = len(p)
        lr = OLS(p, np.vander(np.linspace(-1, 1, n), order + 1)).fit()
        r_sqr[n - 1 + i] = lr.rsquared
        resid[n - 1 + i] = lr.resid[-1]
        betas[n - 1 + i, :] = lr.params

    # return pandas frame if input is series/frame
    if isinstance(y, (pd.DataFrame, pd.Series)):
        r = pd.DataFrame({'resid': resid, 'r2': r_sqr}, index=y.index, columns=['resid', 'r2'])
        betas_fr = pd.DataFrame(betas, index=y.index, columns=['b%d' % i for i in range(order+1)])
        return pd.concat((r, betas_fr), axis=1)

    return resid, r_sqr, betas


def moving_ols(y, x, window):
    """
    Function for calculating moving linear regression model using sliding window
        y = B*x + err
    returns array of betas, residuals and standard deviation for residuals
    residuals = y - yhat, where yhat = betas * x

    Example:

    x = pd.DataFrame(np.random.randn(100, 5))
    y = pd.Series(np.random.randn(100).cumsum())
    m = moving_ols(y, x, 5)
    lr_line = (x * m).sum(axis=1)

    :param y: dependent variable (vector)
    :param x: exogenous variables (vector or matrix)
    :param window: sliding windowsize
    :return: array of betas, residuals and standard deviation for residuals
    """
    # if we have any indexes
    idx_line = y.index if isinstance(y, (pd.Series, pd.DataFrame)) else None
    x_col_names = x.columns if isinstance(y, (pd.Series, pd.DataFrame)) else None

    x = column_vector(x)
    y = column_vector(y)
    nx = len(x)
    if nx != len(y):
        raise ValueError('Series must contain equal number of points')

    if y.shape[1] != 1:
        raise ValueError('Response variable y must be column array or series object')

    if window > nx:
        raise ValueError('Window size must be less than number of observations')

    betas = nans(x.shape);
    err = nans((nx));
    sd = nans((nx));

    for i in range(window, nx + 1):
        ys = y[(i - window):i]
        xs = x[(i - window):i, :]
        lr = OLS(ys, xs).fit()
        betas[i - 1, :] = lr.params
        err[i - 1] = y[i - 1] - (x[i - 1, :] * lr.params).sum()
        sd[i - 1] = lr.resid.std()

    # convert to datafra?e if need
    if x_col_names is not None and idx_line is not None:
        _non_empy = lambda c, idx: c if c else idx
        _bts = pd.DataFrame({
            _non_empy(c, i): betas[:, i] for i, c in enumerate(x_col_names)
        }, index=idx_line)
        return pd.concat((_bts, pd.DataFrame({'error': err, 'stdev': sd}, index=idx_line)), axis=1)
    else:
        return betas, err, sd


def holt_winters_second_order_ewma(x, span, beta) -> tuple:
    """
    The Holt-Winters second order method (aka double exponential smoothing) attempts to incorporate the estimated
    trend into the smoothed data, using a {b_{t}} term that keeps track of the slope of the original signal.
    The smoothed signal is written to the s_{t} term.

    :param x: series values (DataFrame, Series or numpy array)
    :param span: number of data points taken for calculation
    :param beta: trend smoothing factor, 0 < beta < 1
    :return: tuple of smoothed series and smoothed trend
    """
    if span < 0: raise ValueError("Span value must be positive")

    if isinstance(x, (pd.DataFrame, pd.Series)):
        x = x.values

    x = np.reshape(x, (x.shape[0], -1))
    alpha = 2.0 / (1 + span)
    r_alpha = 1 - alpha
    r_beta = 1 - beta
    s = np.zeros(x.shape)
    b = np.zeros(x.shape)
    s[0, :] = x[0,:]
    for i in range(1, x.shape[0]):
        s[i,:] = alpha * x[i,:] + r_alpha*(s[i-1,:] + b[i-1,:])
        b[i,:] = beta * (s[i,:] - s[i-1,:]) + r_beta * b[i-1,:]
    return s, b


def sma(x, period):
    """
    Classical simple moving average

    :param x: input data (as np.array or pd.DataFrame/Series)
    :param period: period of smoothing
    :return: smoothed values
    """
    if period <= 0:
        raise ValueError('Period must be positive and greater than zero !!!')

    x = column_vector(x)
    x, ix = sink_nans_down(x, copy=True)
    s = rolling_sum(x, period) / period
    return lift_nans_up(s, ix)

@njit
def _calc_kama(x, period, fast_span, slow_span):
    x = x.astype(np.float64)
    for i in range(0, x.shape[1]):
        nan_start = np.where(~np.isnan(x[:, i]))[0][0]
        x_s = x[:, i][nan_start:]
        if period >= len(x_s):
            raise ValueError('Wrong value for period. period parameter must be less than number of input observations')
        abs_diff = np.abs(x_s - shift(x_s, 1))
        er = np.abs(x_s - shift(x_s, period)) / rolling_sum(np.reshape(abs_diff, (len(abs_diff), -1)), period)[:,0]
        sc = np.square((er * (2.0 / (fast_span + 1) - 2.0 / (slow_span + 1.0)) + 2 / (slow_span + 1.0)))
        ama = nans(sc.shape)

        # here ama_0 = x_0
        ama[period - 1] = x_s[period - 1]
        for n in range(period, len(ama)):
            ama[n] = ama[n - 1] + sc[n] * (x_s[n] - ama[n - 1])

        # drop 1-st kama value (just for compatibility with ta-lib)
        ama[period - 1] = np.nan

        x[:, i] = np.concatenate((nans(nan_start), ama))

    return x

def kama(x, period, fast_span=2, slow_span=30):
    """
    Kaufman Adaptive Moving Average

    :param x: input data (as np.array or pd.DataFrame/Series)
    :param period: period of smoothing
    :param fast_span: fast period (default is 2 as in canonical impl)
    :param slow_span: slow period (default is 30 as in canonical impl)
    :return: smoothed values
    """
    x = column_vector(x)
    return _calc_kama(x, period, fast_span, slow_span)

@njit
def _calc_ema(x, span, init_mean=True, min_periods=0):
    alpha = 2.0 / (1 + span)
    x = x.astype(np.float64)
    for i in range(0, x.shape[1]):
        nan_start = np.where(~np.isnan(x[:, i]))[0][0]
        x_s = x[:, i][nan_start:]
        a_1 = 1 - alpha
        s = np.zeros(x_s.shape)

        start_i = 1
        if init_mean:
            s += np.nan
            if span - 1 >= len(s):
                x[:,:] = np.nan
                continue
            s[span - 1] = np.mean(x_s[:span])
            start_i = span
        else:
            s[0] = x_s[0]

        for n in range(start_i, x_s.shape[0]):
            s[n] = alpha * x_s[n] + a_1 * s[n - 1]

        if min_periods > 0:
            s[:min_periods - 1] = np.nan

        x[:, i] = np.concatenate((nans(nan_start), s))

    return x


def ema(x, span, init_mean=True, min_periods=0) -> np.ndarray:
    """
    Exponential moving average

    :param x: data to be smoothed
    :param span: number of data points for smooth
    :param init_mean: use average of first span points as starting ema value (default is true)
    :param min_periods: minimum number of observations in window required to have a value (0)
    :return:
    """
    x = column_vector(x)
    return _calc_ema(x, span, init_mean, min_periods)


def zlema(x: np.ndarray, n: int, init_mean=True):
    """
    'Zero lag' moving average
    :type x: np.array
    :param x:
    :param n:
    :param init_mean: True if initial ema value is average of first n points
    :return:
    """
    return ema(2 * x - shift(x, n), n, init_mean=init_mean)


def dema(x, n: int, init_mean=True):
    """
    Double EMA

    :param x:
    :param n:
    :param init_mean: True if initial ema value is average of first n points
    :return:
    """
    e1 = ema(x, n, init_mean=init_mean)
    return 2 * e1 - ema(e1, n, init_mean=init_mean)


def tema(x, n: int, init_mean=True):
    """
    Triple EMA

    :param x:
    :param n:
    :param init_mean: True if initial ema value is average of first n points
    :return:
    """
    e1 = ema(x, n, init_mean=init_mean)
    e2 = ema(e1, n, init_mean=init_mean)
    return 3 * e1 - 3 * e2 + ema(e2, n, init_mean=init_mean)


def bidirectional_ema(x, span, smoother='ema'):
    """
    EMA function is really appropriate for stationary data, i.e., data without trends or seasonality.
    In particular, the EMA function resists trends away from the current mean that it’s already “seen”.
    So, if you have a noisy hat function that goes from 0, to 1, and then back to 0, then the EMA function will return
    low values on the up-hill side, and high values on the down-hill side.
    One way to circumvent this is to smooth the signal in both directions, marching forward,
    and then marching backward, and then average the two.

    :param x: data
    :param span: span for smoothing
    :param smoother: smoothing function (default 'ema' or 'tema')
    :return: smoohted data
    """
    if smoother == 'tema':
        fwd = tema(x, span, init_mean=False)        # take TEMA in forward direction
        bwd = tema(x[::-1], span, init_mean=False)  # take TEMA in backward direction
    else:
        fwd = ema(x, span=span, init_mean=False)        # take EMA in forward direction
        bwd = ema(x[::-1], span=span, init_mean=False)  # take EMA in backward direction
    return (fwd + bwd[::-1]) / 2.


def series_halflife(series):
    """
    Tries to find half-life time for this series.

    Example:
    >>> series_halflife(np.array([1,0,2,3,2,1,-1,-2,0,1]))
    >>> 2.0

    :param series: series data (np.array or pd.Series)
    :return: half-life value rounded to integer
    """
    series = column_vector(series)
    if series.shape[1] > 1:
        raise ValueError("Nultimple series is not supported")

    lag = series[1:]
    dY = -np.diff(series, axis=0)
    m = OLS(dY, sm.add_constant(lag, prepend=False))
    reg = m.fit()

    return np.ceil(-np.log(2) / reg.params[0])


def rolling_std_with_mean(x, mean, window):
    """
    Calculates rolling standard deviation for data from x and already calculated mean series
    :param x: series data
    :param mean: calculated mean
    :param window: window
    :return: rolling standard deviation
    """
    return np.sqrt((((x - mean) ** 2).rolling(window=window).sum() / (window - 1)))


def bollinger(x, window=14, nstd=2, mean='sma', as_frame=False):
    """
    Bollinger Bands indicator

    :param x: input data
    :param window: lookback window
    :param nstd: number of standard devialtions for bands
    :param mean: method for calculating mean: sma, ema, tema, dema, zlema, kama
    :param as_frame: if true result is returned as DataFrame
    :return: mean, upper and lower bands
    """
    rolling_mean = smooth(x, mean, window)
    rolling_std = rolling_std_with_mean(x, rolling_mean, window)

    upper_band = rolling_mean + (rolling_std * nstd)
    lower_band = rolling_mean - (rolling_std * nstd)

    _bb = rolling_mean, upper_band, lower_band
    return pd.concat(_bb, axis=1, keys=['Median', 'Upper', 'Lower']) if as_frame else _bb


def bollinger_atr(x, window=14, atr_window=14, natr=2, mean='sma', atr_mean='ema', as_frame=False):
    """
    Bollinger Bands indicator where ATR is used for bands range estimating
    :param x: input data
    :param window: window size for averaged price
    :param atr_window: atr window size
    :param natr:  number of ATRs for bands
    :param mean: method for calculating mean: sma, ema, tema, dema, zlema, kama
    :param atr_mean:  method for calculating mean for atr: sma, ema, tema, dema, zlema, kama
    :param as_frame: if true result is returned as DataFrame
    :return: mean, upper and lower bands
    """
    if not (isinstance(x, pd.DataFrame) and sum(x.columns.isin(['open', 'high', 'low', 'close'])) == 4):
        raise ValueError("Input series must be DataFrame within 'open', 'high', 'low' and 'close' columns defined !")

    b, _, _ = bollinger(x.close, window, 0, mean, as_frame=False)
    a = natr * atr(x, atr_window, atr_mean)
    _bb = b, b + a, b - a

    return pd.concat(_bb, axis=1, keys=['Median', 'Upper', 'Lower']) if as_frame else _bb


def macd(x, fast=12, slow=26, signal=9, method='ema', signal_method='ema'):
    """
    Moving average convergence divergence (MACD) is a trend-following momentum indicator that shows the relationship
    between two moving averages of prices. The MACD is calculated by subtracting the 26-day slow moving average from the
    12-day fast MA. A nine-day MA of the MACD, called the "signal line", is then plotted on top of the MACD,
    functioning as a trigger for buy and sell signals.

    :param x: input data
    :param fast: fast MA period
    :param slow: slow MA period
    :param signal: signal MA period
    :param method: used moving averaging method (sma, ema, tema, dema, zlema, kama)
    :param signal_method: used method for averaging signal (sma, ema, tema, dema, zlema, kama)
    :return: macd signal
    """
    x_diff = smooth(x, method, fast) - smooth(x, method, slow)

    # averaging signal
    return smooth(x_diff, signal_method, signal).rename('macd')


def atr(x, window=14, smoother='sma'):
    """
    Average True Range indicator

    :param x: input series
    :param window: smoothing window size
    :param smoother: smooting method: sma, ema, zlema, tema, dema, kama
    :return:
    """
    if not (isinstance(x, pd.DataFrame) and sum(x.columns.isin(['open', 'high', 'low', 'close'])) == 4):
        raise ValueError("Input series must be DataFrame within 'open', 'high', 'low' and 'close' columns defined !")

    h_l = abs(x['high'] - x['low'])
    h_pc = abs(x['high'] - x['close'].shift(1))
    l_pc = abs(x['low'] - x['close'].shift(1))
    tr = pd.concat((h_l, h_pc, l_pc), axis=1).max(axis=1)

    # smoothing
    return smooth(tr, smoother, window).rename('atr')


def rolling_atr(x, window, periods, smoother=sma):
    """
    Average True Range indicator calculated on rolling window

    :param x:
    :param window: windiw size as Timedelta or string
    :param periods: number periods for smoothing (applied if > 1)
    :param smoother: smoother (sma is default)
    :return: ATR
    """
    if not (isinstance(x, pd.DataFrame) and sum(x.columns.isin(['open', 'high', 'low', 'close'])) == 4):
        raise ValueError("Input series must be DataFrame within 'open', 'high', 'low' and 'close' columns defined !")

    window = pd.Timedelta(window) if isinstance(window, str) else window
    tf_orig = pd.Timedelta(infer_series_frequency(x))
    
    if window < tf_orig:
        raise ValueError('window size must be great or equal to OHLC series timeframe !!!')

    wind_delta = window + tf_orig
    n_min_periods = wind_delta // tf_orig
    _c_1 = x.rolling(wind_delta, min_periods=n_min_periods).close.apply(lambda y: y[0])
    _l = x.rolling(window, min_periods=n_min_periods - 1).low.apply(lambda y: np.nanmin(y))
    _h = x.rolling(window, min_periods=n_min_periods - 1).high.apply(lambda y: np.nanmax(y))

    # calculate TR
    _tr = pd.concat((abs(_h - _l), abs(_h - _c_1), abs(_l - _c_1)), axis=1).max(axis=1)

    if smoother and periods > 1:
        _tr = smooth(_tr.ffill(), smoother, periods * max(1, (n_min_periods - 1)))

    return _tr


def trend_detector(data, period, nstd, avg='sma', k_ext=1, exit_on_mid=False,
                   use_atr=False, atr_period=12, atr_avg='kama') -> pd.DataFrame:
    """
    Trend detector method

    :param data: input series/frame
    :param period: bb period
    :param nstd: bb num of stds
    :param avg: averaging ma type
    :param k_ext: extending factor
    :param exit_on_mid: trend is over when x crosses middle of bb
    :param use_atr: true if we use bollinger_atr for trend detecting
    :param atr_period: ATR period (used only when use_atr is True)
    :param atr_avg: ATR smoother (used only when use_atr is True)
    :return: frame
    """
    # flatten list lambda
    flatten = lambda l: [item for sublist in l for item in sublist]

    # just taking close prices
    x = data.close if isinstance(data, pd.DataFrame) else data

    if use_atr:
        midle, smax, smin = bollinger_atr(data, period, atr_period, nstd, avg, atr_avg)
    else:
        midle, smax, smin = bollinger(x, period, nstd, avg)

    trend = (((x > smax.shift(1)) + 0.0) - ((x < smin.shift(1)) + 0.0)).replace(0, np.nan)

    # some special case if we want to exit when close is on the opposite side of median price
    if exit_on_mid:
        lom, him = ((x < midle).values, (x > midle).values)
        t = 0;
        _t = trend.values.tolist()
        for i in range(len(trend)):
            t0 = _t[i]
            t = t0 if np.abs(t0) == 1 else t
            if (t > 0 and lom[i]) or (t < 0 and him[i]):
                t = 0
            _t[i] = t
        trend = pd.Series(_t, trend.index)
    else:
        trend = trend.fillna(method='ffill').fillna(0.0)

    # making resulting frame
    m = x.to_frame().copy()
    m['trend'] = trend
    m['blk'] = (m.trend.shift(1) != m.trend).astype(int).cumsum()
    m['x'] = abs(m.trend) * (smax * (-m.trend + 1) - smin * (1 + m.trend)) / 2
    _g0 = m.reset_index().groupby(['blk', 'trend'])
    m['x'] = flatten(abs(_g0['x'].apply(np.array).transform(np.minimum.accumulate).values))
    m['utl'] = m.x.where(m.trend > 0)
    m['dtl'] = m.x.where(m.trend < 0)

    # signals
    tsi = pd.DatetimeIndex(_g0['time'].apply(lambda x: x.values[0]).values)
    m['uts'] = m.loc[tsi].utl
    m['dts'] = m.loc[tsi].dtl

    return m.filter(items=['uts', 'dts', 'trend', 'utl', 'dtl'])


def denoised_trend(x: pd.DataFrame, period: int, window=0, mean: str='kama', bar_returns: bool=True) -> pd.Series:
    """
    Returns denoised trend (T_i).

    ----

    R_i = C_i - O_i

    D_i = R_i - R_{i - period}

    P_i = sum_{k=i-period-1}^{i} abs(R_k)

    T_i = D_i * abs(D_i) / P_i

    ----

    :param x: OHLC dataset (must contain .open and .close columns)
    :param period: period of filtering
    :param window: smothing window size (default 0)
    :param mean: smoother
    :param bar_returns: if True use R_i = close_i - open_i
    :return: trend with removed noise
    """
    if bar_returns:
        ri = x.close - x.open
        di = x.close - x.open.shift(period)
    else:
        ri = x.close - x.close.shift(1)
        di = x.close - x.close.shift(period)
        period -= 1

    abs_di = abs(di)
    si = abs(ri).rolling(window=period+1).sum()
    # for open - close there may be gaps
    if bar_returns:
        si = np.max(np.concatenate((abs_di[:, np.newaxis], si[:, np.newaxis]), axis=1), axis=1)
    filtered_trend = abs_di * (di / si)
    filtered_trend = filtered_trend.replace([np.inf, -np.inf], 0.0)

    if window > 0 and mean is not None:
        filtered_trend = smooth(filtered_trend, mean, window)

    return filtered_trend


def rolling_percentiles(x, window, pctls=(0, 1, 2, 3, 5, 10, 15, 25, 45, 50, 55, 75, 85, 90, 95, 97, 98, 99, 100)):
    """
    Calculates percentiles from x on rolling window basis

    :param x: series data
    :param window: window size
    :param pctls: percentiles
    :return: calculated percentiles as DataFrame indexed by time.
             Every pctl. is denoted as Qd (where d is taken from pctls)
    """
    r = nans((len(x), len(pctls)))
    i = window - 1

    for v in running_view(x, window):
        r[i, :] = np.percentile(v, pctls)
        i += 1

    return pd.DataFrame(r, index=x.index, columns=['Q%d' % q for q in pctls])


def trend_locker(y, order, window, lock_forward_window=1, use_projections=False, as_frame=True):
    """
    Trend locker indicator based on OLS.

    :param y: series data
    :param order: OLS order (1 - linear, 2 - squared etc)
    :param window: rolling window for regression
    :param lock_forward_window: how many forward points to lock (default is 1)
    :param use_projections: if need to get regression projections as well (False)
    :param as_frame: true if need to force converting result to DataFrame (True)
    :return: (residuals, projections, r2, betas)
    """

    if lock_forward_window < 1:
        raise ValueError('lock_forward_window must be positive non zero integer')
        
    n = window + lock_forward_window
    yy = running_view(column_vector(y).T[0], window=n)
    n_pts = len(y)
    resid = nans((n_pts, lock_forward_window))
    proj = nans((n_pts, lock_forward_window)) if use_projections else None
    r_sqr = nans(n_pts)
    betas = nans((n_pts, order + 1))

    for i, p in enumerate(yy):
        x = np.vander(np.linspace(-1, 1, n), order + 1)
        lr = OLS(p[:window], x[:window, :]).fit()
        
        r_sqr[window - 1 + i] = lr.rsquared
        betas[window - 1 + i, :] = lr.params

        pl = p[-lock_forward_window:]
        xl = x[-lock_forward_window:, :]
        fwd_prj = np.sum(lr.params * xl, axis=1)
        fwd_data = pl - fwd_prj
        
        # store forward data
        np.fill_diagonal(resid[window + i : n + i + 1, :], fwd_data)

        # if we asked for projections
        if use_projections:
            np.fill_diagonal(proj[window + i : n + i + 1, :], fwd_prj)
            
    if as_frame and not isinstance(y, pd.Series):
        y = pd.Series(y, name='X')
    
    # return pandas frame if input is series/frame
    if isinstance(y, pd.Series):
        y_idx = y.index
        f_res = pd.DataFrame(data=resid, index=y_idx, columns=['R%d' % i for i in range(1, lock_forward_window + 1)])
        f_prj = None
        if use_projections:
            f_prj = pd.DataFrame(data=proj, index=y_idx, columns=['L%d' % i for i in range(1, lock_forward_window + 1)])
        r = pd.DataFrame({'r2': r_sqr}, index=y_idx, columns=['r2'])
        betas_fr = pd.DataFrame(betas, index=y_idx, columns=['b%d' % i for i in range(order + 1)])
        return pd.concat((y, f_res, f_prj, r, betas_fr), axis=1)

    return resid, proj, r_sqr, betas


def __slope_ols(x):
    x = x[~np.isnan(x)]
    xs = 2 * (x - min(x)) / (max(x) - min(x)) - 1
    m = OLS(xs, np.vander(np.linspace(-1, 1, len(xs)), 2)).fit()
    return m.params[0]


def __slope_angle(p, t):
    return 180 * np.arctan(p / t) / np.pi


def rolling_series_slope(x, period, method='ols', scaling='transform', n_bins=5):
    """
    Rolling slope indicator. May be used as trend indicator

    :param x: time series
    :param period: period for OLS window
    :param n_bins: number of bins used for scaling
    :param method: method used for metric of regression line slope: ('ols' or 'angle')
    :param scaling: how to scale slope 'transform' / 'binarize' / nothing
    :return: series slope metric
    """

    def __binarize(_x, n, limits=(None, None), center=False):
        n0 = n // 2 if center else 0
        _min = np.min(_x) if limits[0] is None else limits[0]
        _max = np.max(_x) if limits[1] is None else limits[1]
        return np.floor(n * (_x - _min) / (_max - _min)) - n0

    def __scaling_transform(x, n=5, need_round=True, limits=None):
        if limits is None:
            _lmax = max(abs(x))
            _lmin = -_lmax
        else:
            _lmax = max(limits)
            _lmin = min(limits)

        if need_round:
            ni = np.round(np.interp(x, (_lmin, _lmax), (-2 * n, +2 * n))) / 2
        else:
            ni = np.interp(x, (_lmin, _lmax), (-n, +n))
        return pd.Series(ni, index=x.index)

    if method == 'ols':
        slp_meth = lambda z: __slope_ols(z)
        _lmts = (-1, 1)
    elif method == 'angle':
        slp_meth = lambda z: __slope_angle(z[-1] - z[0], len(z))
        _lmts = (-90, 90)
    else:
        raise ValueError('Unknown Method %s' % method)

    _min_p = period
    if isinstance(period, str):
        _min_p = pd.Timedelta(period).days

    roll_slope = x.rolling(period, min_periods=_min_p).apply(slp_meth)

    if scaling == 'transform':
        return __scaling_transform(roll_slope, n=n_bins, limits=_lmts)
    elif scaling == 'binarize':
        return __binarize(roll_slope, n=(n_bins - 1) * 4, limits=_lmts, center=True) / 2

    return roll_slope


def adx(ohlc, period, smoother=kama):
    """
    Average Directional Index.

    ADX = 100 * MA(abs((+DI - -DI) / (+DI + -DI)))

    Where:
    -DI = 100 * MA(-DM) / ATR
    +DI = 100 * MA(+DM) / ATR

    +DM: if UPMOVE > DWNMOVE and UPMOVE > 0 then +DM = UPMOVE else +DM = 0
    -DM: if DWNMOVE > UPMOVE and DWNMOVE > 0 then -DM = DWNMOVE else -DM = 0

    DWNMOVE = L_{t-1} - L_t
    UPMOVE = H_t - H_{t-1}

    :param ohlc: DataFrame with ohlc data
    :param period: indicator period
    :param smoother: smoothing function (kama is default)
    :return: adx, DIp, DIm
    """
    if not (isinstance(ohlc, pd.DataFrame) and sum(ohlc.columns.isin(['open', 'high', 'low', 'close'])) == 4):
        raise ValueError("Input series must be DataFrame within 'open', 'high', 'low' and 'close' columns defined !")

    h, l = ohlc['high'], ohlc['low']
    _atr = atr(ohlc, period, smoother=smoother)

    Mu, Md = h.diff(), -l.diff()
    DMp = Mu * (((Mu > 0) & (Mu > Md)) + 0)
    DMm = Md * (((Md > 0) & (Md > Mu)) + 0)
    DIp = 100 * smooth(DMp, smoother, period) / _atr
    DIm = 100 * smooth(DMm, smoother, period) / _atr
    _adx = 100 * smooth(abs((DIp - DIm) / (DIp + DIm)), smoother, period)

    return _adx.rename('ADX'), DIp.rename('DIp'), DIm.rename('DIm')


def rsi(x, periods, smoother=sma):
    """
    U = X_t - X_{t-1}, D = 0 when X_t > X_{t-1}
    D = X_{t-1} - X_t, U = 0 when X_t < X_{t-1}
    U = 0, D = 0,            when X_t = X_{t-1}

    RSI = 100 * E[U, n] / (E[U, n] + E[D, n])

    """
    xx = pd.concat((x, x.shift(1)), axis=1, keys=['c', 'p'])
    df = (xx.c - xx.p)
    mu = smooth(df.where(df > 0, 0), smoother, periods)
    md = smooth(abs(df.where(df < 0, 0)), smoother, periods)

    return 100 * mu / (mu + md)


def pivot_point(data, method='classic'):
    """
    Pivot points indicator  based on daily data
    it supports 'classic', 'woodie' and 'camarilla' species
    """
    freq_sec = pd.Timedelta(data.index.freq).total_seconds()
    if np.isnan(freq_sec) or freq_sec < 3600 * 24:
        x = ohlc_resample(data, "1D", resample_tz="EET")
    elif freq_sec > 3600 * 24:
        raise ValueError("Input series must be resampled in days bars or less")
    else:
        x = data
    pp = pd.DataFrame()
    if method == 'classic':
        pvt = (x.high + x.low + x.close) / 3
        _range = x.high - x.low
        
        pp['R4'] = pvt + 3 * _range
        pp['R3'] = pvt + 2 * _range
        pp['R2'] = pvt + _range 
        pp['R1'] = pvt * 2 - x.low
        pp['P']  = pvt
        pp['S1'] = pvt * 2 - x.high
        pp['S2'] = pvt - _range
        pp['S3'] = pvt - 2 * _range
        pp['S4'] = pvt - 3 * _range
        
        # rearrange
        pp = pp[['R4','R3','R2','R1','P','S1','S2','S3','S4']]
        
    elif method == 'woodie':
        pvt = (x.high + x.low + x.open + x.open) / 4
        _range = x.high - x.low
        
        pp['R3'] = x.high + 2 * (pvt - x.low)
        pp['R2'] = pvt + _range
        pp['R1'] = pvt * 2 - x.low
        pp['P'] = pvt
        pp['S1'] = pvt * 2 - x.high
        pp['S2'] = pvt - _range
        pp['S3'] = x.low + 2 * (x.high - pvt)
        pp = pp[['R3','R2','R1','P','S1','S2','S3']]
        
    elif method == 'camarilla':
        """
            R4 = C + RANGE * 1.1/2
            R3 = C + RANGE * 1.1/4
            R2 = C + RANGE * 1.1/6
            R1 = C + RANGE * 1.1/12
            PP = (HIGH + LOW + CLOSE) / 3
            S1 = C - RANGE * 1.1/12
            S2 = C - RANGE * 1.1/6
            S3 = C - RANGE * 1.1/4
            S4 = C - RANGE * 1.1/2        
        """
        pvt = (x.high + x.low + x.close) / 3
        _range = x.high - x.low
        
        pp['R4'] = x.close + _range * 1.1 / 2
        pp['R3'] = x.close + _range * 1.1 / 4
        pp['R2'] = x.close + _range * 1.1 / 6
        pp['R1'] = x.close + _range * 1.1 / 12
        pp['P'] = pvt
        pp['S1'] = x.close - _range * 1.1 / 12
        pp['S2'] = x.close - _range * 1.1 / 6
        pp['S3'] = x.close - _range * 1.1 / 4
        pp['S4'] = x.close - _range * 1.1 / 2
        pp = pp[['R4','R3','R2','R1','P','S1','S2','S3','S4']]
    else:
        raise ValueError("Unknown method %s. Available methods is classic, woodie, camarilla" % method)

    pp.index = pp.index + pd.Timedelta('1D')

    return data.combine_first(pp).fillna(method='ffill')[pp.columns]

def intraday_min_max(data):
    """
    min and max values intraday
    :param data: ohlcv series
    :return: series with min and max values intraday
    """
    
    if not (isinstance(data, pd.DataFrame) and sum(data.columns.isin(['open', 'high', 'low', 'close'])) == 4):
        raise ValueError("Input series must be DataFrame within 'open', 'high', 'low' and 'close' columns defined !")

    def _day_min_max(d):
        _d_min = np.minimum.accumulate(d.low)
        _d_max = np.maximum.accumulate(d.high)
        return pd.concat((_d_min, _d_max), axis=1, keys=['Min', 'Max'])

    source_tz = data.index.tz
    if not source_tz:
        x = data.tz_localize('GMT')
    else:
        x = data

    x = x.tz_convert('EET')
    return x.groupby(x.index.date).apply(_day_min_max).tz_convert(source_tz)

@njit
def _laguerre_calc(xx, g):
    l0, l1, l2, l3, f = np.zeros(len(xx)), np.zeros(len(xx)), np.zeros(len(xx)), np.zeros(len(xx)), np.zeros(
        len(xx))
    for i in range(1, len(xx)):
        l0[i] = (1 - g) * xx[i] + g * l0[i - 1]
        l1[i] = -g * l0[i] + l0[i - 1] + g * l1[i - 1]
        l2[i] = -g * l1[i] + l1[i - 1] + g * l2[i - 1]
        l3[i] = -g * l2[i] + l2[i - 1] + g * l3[i - 1]
        f[i] = (l0[i] + 2 * l1[i] + 2 * l2[i] + l3[i]) / 6
    return f

def laguerre_filter(x, gamma=0.8):
    """
    Laguerre 4 pole IIR filter
    """
    return pd.Series(_laguerre_calc(x.values.flatten(), gamma), x.index)

@njit
def _lrsi_calc(xx, g):
    l0, l1, l2, l3, f = np.zeros(len(xx)), np.zeros(len(xx)), np.zeros(len(xx)), np.zeros(len(xx)), np.zeros(
        len(xx))
    for i in range(1, len(xx)):
        l0[i] = (1 - g) * xx[i] + g * l0[i - 1]
        l1[i] = -g * l0[i] + l0[i - 1] + g * l1[i - 1]
        l2[i] = -g * l1[i] + l1[i - 1] + g * l2[i - 1]
        l3[i] = -g * l2[i] + l2[i - 1] + g * l3[i - 1]

        _cu, _cd = 0, 0
        _d0 = l0[i] - l1[i]
        _d1 = l1[i] - l2[i]
        _d2 = l2[i] - l3[i]

        if _d0 >= 0:
            _cu = _d0
        else:
            _cd = np.abs(_d0)

        if _d1 >= 0:
            _cu += _d1
        else:
            _cd += np.abs(_d1)

        if _d2 >= 0:
            _cu += _d2
        else:
            _cd += np.abs(_d2)

        f[i] = 100 * _cu / (_cu + _cd) if (_cu + _cd) != 0 else 0

    return f

def lrsi(x, gamma=0.5):
    """
    Laguerre RSI
    """
    return pd.Series(_lrsi_calc(x.values.flatten(), gamma), x.index)

@njit
def calc_ema_time(t, vv, period, min_time_quant, with_correction=True):
    index = np.empty(len(vv) - 1, dtype=np.float64)
    values = np.empty(len(vv) - 1, dtype=np.float64)
    dt = np.diff(t)
    dt[dt == 0] = min_time_quant
    a = dt / period
    u = np.exp(-a)
    _ep = vv[0]

    if with_correction:
        v = (1 - u) / a
        c1 = v - u
        c2 = 1 - v
        for i in range(0, len(vv) - 1):
            _ep = u[i] * _ep + c1[i] * vv[i] + c2[i] * vv[i + 1]
            index[i] = t[i + 1]
            values[i] = _ep
    else:
        v = 1 - u
        for i in range(0, len(vv) - 1):
            _ep = _ep + v[i] * (vv[i + 1] - _ep)
            index[i] = t[i + 1]
            values[i] = _ep
    return index, values


def ema_time(x, period, min_time_quant=pd.Timedelta('1ms'), with_correction=True):
    t = x.index.values
    vv = x.values
    if not isinstance(x, pd.Series):
        raise ValueError('Input series must be instance of pandas Series class')

    if isinstance(period, str):
        period = pd.Timedelta(period)

    index, values = calc_ema_time(t.astype('float64'), vv, period.value, min_time_quant.value, with_correction)

    old_ser_name = 'UnknownSeries' if x.name is None else x.name
    res = pd.Series(values, pd.to_datetime(index), name='EMAT_%d_sec_%s' % (period.seconds, old_ser_name))
    res = res.loc[~res.index.duplicated(keep='first')]
    return res
