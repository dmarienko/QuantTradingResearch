import types
import pandas as pd
import numpy as np
from typing import Union

from pandas.core.generic import NDFrame
from itertools import product


def make_forward_returns_matrix(x: pd.DataFrame, n_forward_bars=1, use_open_close=True, use_usd_rets=False, shift=-1):
    """
    Forward returns matrix generator
    --------------------------------
    :param x: OHLC data frame
    :param n_forward_bars: number of bars used for forward returns calculations (F1 - 1 bar, F2 - 2 bars, ... )
    :param use_open_close: return = close/open - 1 (True)
    :param use_usd_rets: use raw difference as returns instead of percentage (False)
    :param shift: number of bars to be shifted (default is -1 - one bar back)
    """
    n_forward_bars = max(abs(n_forward_bars),1)
    d = {}
    
    for i in range(1, n_forward_bars + 1):
        n_ = 'F%d' % i
        if use_open_close:
            if use_usd_rets:
                d.update({n_ : x.close.shift(-i+1) - x.open})
            else:
                d.update({n_ : x.close.shift(-i+1) / x.open - 1})
        else:
            if use_usd_rets:
                d.update({n_ : x.close.shift(-i) - x.close})
            else:
                d.update({n_ : x.close.shift(-i) / x.close - 1})
                
    # we need forward return at current moment so shift back it
    return pd.DataFrame(d, index=x.index).shift(shift)


def retain_columns_and_join(data: dict, columns):
    """
    Retains given columns from every value of data dictionary and concatenate them into single data frame    

    closes = retain_columns_and_join(data, 'close')
    hi_lo = retain_columns_and_join(data, ['high', 'low'])

    :param data: dictionary with dataframes  
    :param columns: columns names need to be retained 
    :return: data frame 
    """
    if not isinstance(data, dict):
        raise ValueError('Data must be passed as dictionary')

    return pd.concat([data[k][columns] for k in data.keys()], axis=1, keys=data.keys())


def permutate_params(parameters: dict, conditions:Union[types.FunctionType, list, tuple]=None) -> list([dict]):
    """
    Generate list of all permutations for given parameters and it's possible values

    Example:

    >>> def foo(par1, par2):
    >>>     print(par1)
    >>>     print(par2)
    >>>
    >>> # permutate all values and call function for every permutation
    >>> [foo(**z) for z in permutate_params({
    >>>                                       'par1' : [1,2,3],
    >>>                                       'par2' : [True, False]
    >>>                                     }, conditions=lambda par1, par2: par1<=2 and par2==True)]
    
    1
    True
    2
    True

    :param conditions: list of filtering functions
    :param parameters: dictionary
    :return: list of permutations
    """
    if conditions is None:
        conditions = []
    elif isinstance(conditions, types.FunctionType):
        conditions = [conditions]
    elif isinstance(conditions, (tuple, list)):
        if not all([isinstance(e, types.FunctionType) for e in conditions]):
            raise ValueError('every condition must be a function')
    else:
        raise ValueError('conditions must be of type of function, list or tuple')

    args = []
    vals = []
    for (k, v) in parameters.items():
        args.append(k)
        vals.append([v] if not isinstance(v, (list, tuple)) else v)
    d = [dict(zip(args, p)) for p in product(*vals)]
    result = []
    for params_set in d:
        conditions_met = True
        for cond_func in conditions:
            func_param_args = cond_func.__code__.co_varnames
            func_param_values = [params_set[arg] for arg in func_param_args]
            if not cond_func(*func_param_values):
                conditions_met = False
                break
        if conditions_met:
            result.append(params_set)
    return result