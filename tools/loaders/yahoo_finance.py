import pandas as pd
import numpy as np
from datetime import datetime
from tqdm.notebook import tqdm
import requests

from tools.utils.utils import mstruct, red, green, yellow, blue, magenta, cyan, white


def load_yahoo_daily_data(symbols, start, end=None, use_adj=True):
    """
    Loads Yahoo OHLC daily data for requested symbols and returns result as dict
    """
    symbols = [symbols] if not isinstance(symbols, (list, tuple, np.ndarray)) else symbols
    res_data = {}
    for s in tqdm(symbols): 
        d = load_yahoo_ohlc_data(s, start, end, use_adj=use_adj, timeframe='1d')
        if d is not None:
            res_data[s] = d
    print(f'\n> Loaded {len(res_data)} symbols')
    return res_data


def load_yahoo_ohlc_data(symbol, start, end=None, use_field=None, use_adj=True, drop_time=True, timeframe='1d'):
    """
    Loading daily data from Yahoo
    """
    if end is None:
        end = datetime.now().strftime('%Y-%m-%d')
    
    replacements = {'BRKB': 'BRK-B'}
    url = "https://query1.finance.yahoo.com/v8/finance/chart/{}".format(replacements.get(symbol, symbol))
#     print('Loading %s ...' % symbol, end='')

    r = requests.get(url, params={
        'interval': timeframe,
        'period1': int(pd.to_datetime(start).timestamp()),
        'period2': int(pd.to_datetime(end).timestamp()),
    })
    data = r.json()
    try:
        meta = data['chart']['result'][0]['meta']
        ds = data['chart']['result'][0]
        qts = ds['indicators']['quote'][0]
        a_close = ds['indicators']['adjclose'][0] if 'adjclose' in ds['indicators'] else {}

        df = pd.DataFrame.from_dict({**qts, **a_close})
        df.index = pd.to_datetime(ds['timestamp'], unit="s", yearfirst=True)
        df.sort_index(inplace=True)
        try:
            df.index = df.index + pd.Timedelta('%dS' % meta['gmtoffset'])
        except Exception as e:
            print(">>> Error: %s !!!" % e)

        if drop_time:
            df.index = pd.DatetimeIndex(df.index.date)

        # use only adjusted closes
        if use_adj:
            af = df['adjclose'] / df['close']
            df['open'] = df['open'] * af
            df['high'] = df['high'] * af
            df['low'] = df['low'] * af
            df['close'] = df['adjclose']

#         print('[loaded %d bars]' % len(df))
        return df if use_field is None else df[use_field]
    except Exception as e:
        print(red(symbol), end=' ')
        return None
    