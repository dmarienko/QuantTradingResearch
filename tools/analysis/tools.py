import types
from typing import Union

import numpy as np
import pandas as pd
from numba import njit
from numpy.lib.stride_tricks import as_strided as stride


def column_vector(x):
    """
    Convert any vector to column vector. Matrices remain unchanged.
     
    :param x: vector 
    :return: column vector 
    """
    if isinstance(x, (pd.DataFrame, pd.Series)): x = x.values
    return np.reshape(x, (x.shape[0], -1))

@njit
def shift(xs: np.ndarray, n: int, fill=np.nan) -> np.ndarray:
    """
    Shift data in numpy array (aka lag function):

    shift(np.array([[1.,2.], 
                    [11.,22.], 
                    [33.,44.]]), 1)

    >> array([[ nan,  nan],
              [  1.,   2.],
              [ 11.,  22.]])

    :param xs: 
    :param n: 
    :param fill: value to use for  
    :return: 
    """
    e = np.empty_like(xs)
    if n >= 0:
        e[:n] = fill
        e[n:] = xs[:-n]
    else:
        e[n:] = fill
        e[:n] = xs[-n:]
    return e


def sink_nans_down(x_in, copy=False) -> (np.ndarray, np.ndarray):
    """
    Move all starting nans 'down to the bottom' in every column.

    NaN = np.nan
    x = np.array([[NaN, 1, NaN], 
                  [NaN, 2, NaN], 
                  [NaN, 3, NaN], 
                  [10,  4, NaN], 
                  [20,  5, NaN], 
                  [30,  6, 100], 
                  [40,  7, 200]])

    x1, nx = sink_nans_down(x)
    print(x1)

    >> [[  10.    1.  100.]
        [  20.    2.  200.]
        [  30.    3.   nan]
        [  40.    4.   nan]
        [  nan    5.   nan]
        [  nan    6.   nan]
        [  nan    7.   nan]]

    :param x_in: numpy 1D/2D array
    :param copy: set if need to make copy input to prevent being modified [False by default]
    :return: modified x_in and indexes
    """
    x = np.copy(x_in) if copy else x_in
    n_ix = np.zeros(x.shape[1])
    for i in range(0, x.shape[1]):
        f_n = np.where(~np.isnan(x[:, i]))[0]
        if len(f_n) > 0:
            if f_n[0] != 0:
                x[:, i] = np.concatenate((x[f_n[0]:, i], x[:f_n[0], i]))
            n_ix[i] = f_n[0]
    return x, n_ix


def lift_nans_up(x_in, n_ix, copy=False) -> np.ndarray:
    """
    Move all ending nans 'up to top' of every column.

    NaN = np.nan
    x = np.array([[NaN, 1, NaN], 
                  [NaN, 2, NaN], 
                  [NaN, 3, NaN], 
                  [10,  4, NaN], 
                  [20,  5, NaN], 
                  [30,  6, 100], 
                  [40, 7, 200]])

    x1, nx = sink_nans_down(x)
    print(x1)

    >> [[  10.    1.  100.]
        [  20.    2.  200.]
        [  30.    3.   nan]
        [  40.    4.   nan]
        [  nan    5.   nan]
        [  nan    6.   nan]
        [  nan    7.   nan]]

    x2 = lift_nans_up(x1, nx)
    print(x2)

    >> [[  nan    1.   nan]
        [  nan    2.   nan]
        [  nan    3.   nan]
        [  10.    4.   nan]
        [  20.    5.   nan]
        [  30.    6.  100.]
        [  40.    7.  200.]]

    :param x_in: numpy 1D/2D array
    :param n_ix: indexes for every column
    :param copy: set if need to make copy input to prevent being modified [False by default]
    :return: modified x_in
    """
    x = np.copy(x_in) if copy else x_in
    for i in range(0, x.shape[1]):
        f_n = int(n_ix[i])
        if f_n != 0:
            x[:, i] = np.concatenate((nans(f_n), x[:-f_n, i]))
    return x


def add_constant(x, const=1., prepend=True):
    """
    Adds a column of constants to an array

    Parameters
    ----------
    :param data: column-ordered design matrix
    :param prepend: If true, the constant is in the first column.  Else the constant is appended (last column)
    :param const: constant value to be appended (default is 1.0) 
    :return: 
    """
    x = column_vector(x)
    if prepend:
        r = (const * np.ones((x.shape[0], 1)), x)
    else:
        r = (x, const * np.ones((x.shape[0], 1)))
    return np.hstack(r)


def isscalar(x):
    """
    Returns true if x is scalar value
    
    :param x: 
    :return: 
    """
    return not isinstance(x, (list, tuple, dict, np.ndarray))

@njit
def nans(dims):
    """
    nans((M,N,P,...)) is an M-by-N-by-P-by-... array of NaNs.
    
    :param dims: dimensions tuple 
    :return: nans matrix 
    """
    return np.nan * np.ones(dims)

@njit
def rolling_sum(x:np.ndarray, n:int) -> np.ndarray:
    """
    Fast running sum for numpy array (matrix) along columns.

    Example:
    >>> rolling_sum(column_vector(np.array([[1,2,3,4,5,6,7,8,9], [11,22,33,44,55,66,77,88,99]]).T), n=5)
    
    array([[  nan,   nan],
       [  nan,   nan],
       [  nan,   nan],
       [  nan,   nan],
       [  15.,  165.],
       [  20.,  220.],
       [  25.,  275.],
       [  30.,  330.],
       [  35.,  385.]])

    :param x: input data
    :param n: rolling window size
    :return: rolling sum for every column preceded by nans
    """
    for i in range(0, x.shape[1]):
        ret = np.nancumsum(x[:,i])
        ret[n:] = ret[n:] - ret[:-n]
        x[:,i] = np.concatenate((nans(n - 1), ret[n - 1:]))
    return x


def apply_to_frame(func, x, *args, **kwargs):
    """
    Utility applies given function to x and converts result to incoming type 

    >>> from ira.analysis.timeseries import ema
    >>> apply_to_frame(ema, data['EURUSD'], 50)
    >>> apply_to_frame(lambda x, p1: x + p1, data['EURUSD'], 1)

    :param func: function to map
    :param x: input data
    :param args: arguments of func
    :param kwargs: named arguments of func (if it contains keep_names=True it won't change source columns names)
    :return: result of function's application
    """
    _keep_names = False
    if 'keep_names' in kwargs:
        _keep_names = kwargs.pop('keep_names')

    if func is None or not isinstance(func, types.FunctionType):
        raise ValueError(str(func) + ' must be callable object')

    xp = column_vector(func(x, *args, **kwargs))
    _name = None
    if not _keep_names:
        _name = func.__name__ + '_' + '_'.join([str(i) for i in args])

    if isinstance(x, pd.DataFrame):
        c_names = x.columns if _keep_names else ['%s_%s' % (c, _name) for c in x.columns]
        return pd.DataFrame(xp, index=x.index, columns=c_names)
    elif isinstance(x, pd.Series):
        return pd.Series(xp.flatten(), index=x.index, name=_name)

    return xp


def ohlc_resample(df, new_freq: str = '1H', vmpt: bool = False, resample_tz=None) -> Union[pd.DataFrame, dict]:
    """
    Resample OHLCV/tick series to new timeframe.

    Example:
    >>> d = pd.DataFrame({
    >>>          'open' : np.random.randn(30),
    >>>          'high' : np.random.randn(30),
    >>>          'low' : np.random.randn(30),
    >>>          'close' : np.random.randn(30)
    >>>         }, index=pd.date_range('2000-01-01 00:00', freq='5Min', periods=30))
    >>>
    >>> ohlc_resample(d, '15Min')
    >>>
    >>> # if we need to resample quotes
    >>> from ira.datasource import DataSource
    >>> with DataSource('kdb::dukas') as ds:
    >>>     quotes = ds.load_data(['EURUSD', 'GBPUSD'], '2018-05-07', '2018-05-11')
    >>> ohlc_resample(quotes, '1Min', vmpt=True)

    :param df: input ohlc or bid/ask quotes or dict
    :param new_freq: how to resample rule (see pandas.DataFrame::resample)
    :param vmpt: use volume weighted price for quotes (if false mid price will be used)
    :param resample_tz: timezone for resample. For example, to create daily bars in the EET timezone
    :return: resampled ohlc / dict
    """
    def __mx_rsmpl(d, freq: str, is_vmpt: bool = False, resample_tz=None) -> pd.DataFrame:
        _cols = d.columns
        _source_tz = d.index.tz

        # if we have bid/ask frame
        if 'ask' in _cols and 'bid' in _cols:
            # if sizes are presented we can calc vmpt if need
            if is_vmpt and 'askvol' in _cols and 'bidvol' in _cols:
                mp = (d.ask * d.bidvol + d.bid * d.askvol) / (d.askvol + d.bidvol)
                return mp.resample(freq).agg('ohlc')

            # if there is only asks and bids and we don't need vmpt
            result = _tz_convert(d[['ask', 'bid']].mean(axis=1), resample_tz, _source_tz)
            result = result.resample(freq).agg('ohlc')
            # Convert timezone to back if it changed
            return result if not resample_tz else result.tz_convert(_source_tz)

        # for OHLC case or just simple series
        if all([i in _cols for i in ['open', 'high', 'low', 'close']]) or isinstance(d, pd.Series):
            ohlc_rules = {'open': 'first',
                          'high': 'max',
                          'low': 'min',
                          'close': 'last',
                          'ask_vol': 'sum',
                          'bid_vol': 'sum',
                          'volume': 'sum'
                          }
            result = _tz_convert(d, resample_tz, _source_tz)
            result = result.resample(freq).apply(dict(i for i in ohlc_rules.items() if i[0] in d.columns)).dropna()
            # Convert timezone to back if it changed
            return result if not resample_tz else result.tz_convert(_source_tz)

        raise ValueError("Can't recognize structure of input data !")

    def _tz_convert(df, tz, source_tz):
        if tz:
            if not source_tz:
                df = df.tz_localize('GMT')
            return df.tz_convert(tz)
        else:
            return df

    if isinstance(df, (pd.DataFrame, pd.Series)):
        return __mx_rsmpl(df, new_freq, vmpt, resample_tz)
    elif isinstance(df, dict):
        return {k: __mx_rsmpl(v, new_freq, vmpt, resample_tz) for k, v in df.items()}
    else:
        raise ValueError('Type [%s] is not supported in ohlc_resample' % str(type(df)))


def round_up(x, step):
    """
    Round float to nearest greater value by step
    34.23 -> 34.50 etc

    :param x: value to round
    :param step: step
    :return: rounded value
    """
    return int(np.ceil(x / step)) * step


def round_down(x, step):
    """
    Round float to nearest lesser value by step
    34.67 -> 34.50 etc

    :param x: value to round
    :param step: step
    :return: rounded value
    """
    return (int(x / step)) * step


def roll(df: pd.DataFrame, w: int, **kwargs):
    """
    Rolling window on dataframe using multiple columns
    
    >>> roll(pd.DataFrame(np.random.randn(10,3), index=list('ABCDEFGHIJ')), 3).apply(print)
    
    or alternatively 
    
    >>> pd.DataFrame(np.random.randn(10,3), index=list('ABCDEFGHIJ')).pipe(roll, 3).apply(lambda x: print(x[2]))
    
    :param df: pandas DataFrame
    :param w: window size (only integers)
    :return: rolling window
    """
    if w > len(df):
        raise ValueError("Window size exceeds number of rows !")
        
    v = df.values
    d0, d1 = v.shape
    s0, s1 = v.strides
    a = stride(v, (d0 - (w - 1), w, d1), (s0, s0, s1))
    rolled_df = pd.concat({
        row: pd.DataFrame(values, columns=df.columns)
        for row, values in zip(df.index, a)
    })

    return rolled_df.groupby(level=0, **kwargs)


def drop_duplicated_indexes(df, keep='first'):
    """
    Drops duplicated indexes in dataframe/series
    Keeps either first or last occurence (parameter keep)
    """
    return df[~df.index.duplicated(keep=keep)]


def scols(*xs, keys=None, names=None, keep='all'):
    """
    Concat dataframes/series from xs into single dataframe by axis 1
    :param keys: keys of new dataframe (see pd.concat's keys parameter)
    :param names: new column names or dict with replacements
    :return: combined dataframe
    
    Example
    -------
    >>>  scols(
            pd.DataFrame([1,2,3,4,-4], list('abcud')),
            pd.DataFrame([111,21,31,14], list('xyzu')), 
            pd.DataFrame([11,21,31,124], list('ertu')), 
            pd.DataFrame([11,21,31,14], list('WERT')), 
            names=['x', 'y', 'z', 'w'])
    """
    r = pd.concat((xs), axis=1, keys=keys)
    if names:
        if isinstance(names, (list, tuple)):
            if len(names) == len(r.columns):
                  r.columns = names
            else:
                raise ValueError(f"if 'names' contains new column names it must have same length as resulting df ({len(r.columns)})")
        elif isinstance(names, dict):
            r = r.rename(columns=names)
    return r


def srows(*xs, keep='all', sort=True):
    """
    Concat dataframes/series from xs into single dataframe by axis 0
    :param sort: if true it sorts resulting dataframe by index (default)
    :param keep: how to deal with duplicated indexes. 
                 If set to 'all' it doesn't do anything (default). Otherwise keeps first or last occurences
    :return: combined dataframe
    
    Example
    -------
    >>>  srows(
            pd.DataFrame([1,2,3,4,-4], list('abcud')),
            pd.DataFrame([111,21,31,14], list('xyzu')), 
            pd.DataFrame([11,21,31,124], list('ertu')), 
            pd.DataFrame([11,21,31,14], list('WERT')), 
            sort=True, keep='last')
    """
    r = pd.concat((xs), axis=0)
    r = r.sort_index() if sort else r
    if keep != 'all':
        r = drop_duplicated_indexes(r, keep=keep)
    return r


def retain_columns_and_join(data: dict, columns):
    """
    Retains given columns from every value of data dictionary and concatenate them into single data frame    

    from ira.datasource import DataSource
    from ira.analysis.tools import retain_columns_and_join 

    ds = DataSource('yahoo::daily')
    data = ds.load_data(['aapl', 'msft', 'spy'], '2000-01-01', 'now')

    closes = retain_columns_and_join(data, 'close')
    hi_lo = retain_columns_and_join(data, ['high', 'low'])

    :param data: dictionary with dataframes  
    :param columns: columns names need to be retained 
    :return: data frame 
    """
    if not isinstance(data, dict):
        raise ValueError('Data must be passed as dictionary')

    return pd.concat([data[k][columns] for k in data.keys()], axis=1, keys=data.keys())
