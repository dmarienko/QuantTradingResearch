import pandas as pd
import numpy as np
import os
from os.path import join
from tqdm.auto import tqdm
from ira.datasource import DataSource 
from ira.analysis.kalman import kf_smoother, kalman_regression_estimator
from ira.analysis.tools import scols


def norm(xs):
    """
    Just small helper
    """
    return xs / xs.iloc[0] if isinstance(xs, (pd.Series, pd.DataFrame)) else xs / xs[0]


def load_ohlc(timeframe, start='2022-06-01', end='2022-07-05'):
    """
    Data loader from Kdb/Q
    """
    md = {}
    tf_sec = pd.Timedelta(timeframe).seconds
    with DataSource('kdb::ftx.quotes', join(os.getcwd(), 'dsconfig.json')) as ds:
        for s in tqdm(ds.series_list()):
            if not s[0].isdigit():
                md[s] = ds.load_data(s, start, end, timeframe=tf_sec)[s] 
    return md


def ksmooth(x, pv, mv):
    """
    Kalman filter for smoothing series x
    """
    s, cvr = kf_smoother(x, pv, mv)
    return pd.DataFrame(np.array([s,cvr]).T, index=x.index, columns=['x', 'cvr'])


def merge_columns_by_op(x: pd.DataFrame, y: pd.DataFrame, op):
    """
    Merge 2 dataframes into one and performing operation on intersected columns
    
    merge_columns_by_op(
        pd.DataFrame({'A': [1,2,3], 'B': [100,200,300]}), 
        pd.DataFrame({'B': [5,6,7], 'C': [10,20,30]}), 
        lambda x,y: x + y 
    )
    
        B 	    A   C
    0 	105 	1 	10
    1 	206 	2 	20
    2 	307 	3 	30
    
    """
    if x is None or x.empty: return y
    if y is None: return x
    r = []
    uc = set(x.columns & y.columns)
    for c in uc:
        r.append(op(x[c], y[c]))

    for c in set(x.columns) - uc:
        r.append(x[c])

    for c in set(y.columns) - uc:
        r.append(y[c])

    return scols(*r)