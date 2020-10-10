
import numpy as np
import pandas as pd


def absmaxdd(data):
    """

    Calculates the maximum absolute drawdown of series data.

    Args:
        data: vector of doubles. Data may be presented as list,
        tuple, numpy array or pandas series object.

    Returns:
        (max_abs_dd, d_start, d_peak, d_recovered, dd_data)

    Where:
        - max_abs_dd: absolute maximal drawdown value
        - d_start: index from data array where drawdown starts
        - d_peak: index when drawdown reach it's maximal value
        - d_recovered: index when DD is fully recovered
        - dd_data: drawdown series

    Example:

    mdd, ds, dp, dr, dd_data = absmaxdd(np.random.randn(1,100).cumsum())


    \(c\) 2016, http://www.appliedalpha.com

    """

    if not isinstance(data, (list, tuple, np.ndarray, pd.Series)):
        raise TypeError('Unknown type of input series');

    datatype = type(data)

    if datatype is pd.Series:
        indexes = data.index
        data = data.values
    elif datatype is not np.ndarray:
        data = np.array(data)

    dd = np.maximum.accumulate(data) - data
    mdd = dd.max()
    d_peak = dd.argmax()

    if mdd == 0:
        return 0, 0, 0, 0, [0]

    zeros_ixs = np.where(dd == 0)[0]
    zeros_ixs = np.insert(zeros_ixs, 0, 0)
    zeros_ixs = np.append(zeros_ixs, dd.size)

    d_start = zeros_ixs[zeros_ixs < d_peak][-1]
    d_recover = zeros_ixs[zeros_ixs > d_peak][0]

    if d_recover >= data.__len__():
        d_recover = data.__len__() - 1

    if datatype is pd.Series:
        dd = pd.Series(dd, index=indexes)

    return mdd, d_start, d_peak, d_recover, dd


def max_drawdown_pct(returns):
    """
    Finds the maximum drawdown of a strategy returns in percents

    :param returns: pd.Series or np.ndarray daily returns of the strategy, noncumulative 
    :return: maximum drawdown in percents 
    """
    if len(returns) < 1:
        return np.nan
    
    if isinstance(returns, pd.Series):
        returns = returns.values

    # drop nans
    returns[np.isnan(returns) | np.isinf(returns)] = 0.0

    cumrets = 100*(returns + 1).cumprod(axis=0)
    max_return = np.fmax.accumulate(cumrets)
    return np.nanmin((cumrets - max_return) / max_return)


def dd_freq_stats(draw_down_series):
    """
    Calculates drawdown frequencies statistics

    :param draw_down_series:
    :return: table with drawdown freqeuencies statistics

            Occurencies    AvgMagnitude        Max        Min
    duration
    1             1456.0    1019.695288  165927.04       0.18
    2              707.0    1639.920325   29392.71      27.14
    3              446.0    2244.420987   27785.23      28.96
    4              316.0    2543.957310   15941.31     145.70
    5              240.0    3124.423792   27746.15     242.41
    ...

    Example:

    t_pnl = pnl_data.get_field_data('Total_PnL')[0];
    t_pnl = t_pnl[t_pnl.columns[0]]
    c_pnl = t_pnl.cumsum(skipna=True).between_time('17:00:01', '16:00').dropna()

    mdd,start,peak,recover,dd_ser = absmaxdd(c_pnl)
    dd_stat = dd_freq_stats(dd_ser)
    max(dd_stat['Max'])==mdd

    \(c\) 2016, http://www.appliedalpha.com

    """
    f2 = pd.DataFrame(draw_down_series.values, columns=['dd'], index=range(0, len(draw_down_series)))
    f2['t'] = f2['dd'] > 0
    fst = f2.index[f2['t'] & ~ f2['t'].shift(1).fillna(False)]
    lst = f2.index[f2['t'] & ~ f2['t'].shift(-1).fillna(False)]

    dd_periods = pd.DataFrame(
        data=np.array([(j - i + 1, max(draw_down_series.iloc[i:j + 1])) for i, j in zip(fst, lst)],
                      dtype=[('duration', np.uint8), ('m', np.float64)]), columns=['duration', 'm']
    )

    dx = dd_periods.groupby(by='duration').agg([len, np.average, np.max, np.min]). \
        rename(columns={'len': 'Occurencies', 'average': 'AvgMagnitude', 'amax': 'Max', 'amin': 'Min'})

    return dx.xs('m', axis=1, drop_level=True)


def flat_periods(pnl_series, tolerance):
    """
    Flat PnL periods finder.

    TODO: need to port it from Matlab
    TODO: also add statistics of found flat periods

    function [p] = flatperiods(data, tolerance)
        p = [];
        s0 = 1;
        sUp = data(s0) + tolerance/2;
        sDw = data(s0) - tolerance/2;
        for si = 2:length(data)
            if data(si) > sUp || data(si) < sDw
                if s0 ~= si-1 % skip single point
                    p = [p;  [s0 si-1]];
                end
                sUp = data(si) + tolerance/2;
                sDw = data(si) - tolerance/2;
                s0 = si;
                continue
            end
        end
    end

    :param pnl_series: cumulative PnL series (pandas/numpy/list should be accepted)
    :param tolerance: channel size considered as flat period
    :return: should return set (or list) of indexes where flat periods are found
    """
    raise NotImplementedError("Not yet implemented !!!")