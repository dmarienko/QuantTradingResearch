import pandas as pd
import numpy as np

import requests
from io import StringIO
import json
from datetime import datetime as dt, timezone

from blockchain import statistics


def load_blockchain_series(rep, alias=None, timespan='all', api_code='c785d32f-461a-4f27-a1bf-7422544a8ca5'):
    print(' > Loading [%s] ...' % rep)
    x = statistics.get_chart(rep, time_span=timespan, api_code=api_code)
    sx = pd.Series({pd.to_datetime(dt.fromtimestamp(v.x, timezone.utc).date()):v.y for v in x.values})
    if alias is not None:
        sx = sx.rename(alias)
    return sx


def load_blockchain_series_old(rep, alias=None, timespan='all', api_code='c785d32f-461a-4f27-a1bf-7422544a8ca5'):
    url = 'https://api.blockchain.info/charts/%s?timespan=%s&format=csv' % (rep, timespan)
    if api_code is not None:
        url = url + '&api_code=' + api_code
    print('Loading [%s] ...' % rep)
    r = requests.get(url)
    if r.status_code == 200:
        c_name = rep if alias is None else alias
        data = pd.read_csv(StringIO(r.text), names=['time', c_name], header=None, parse_dates=True, index_col='time')
        return data
    else:
        raise ValueError("Can't load data: %s" % r)

        
def load_coindesk_ohlc():
    n_date = dt.now().date().strftime('%Y-%m-%d')
    r = requests.get('https://api.coindesk.com/v1/bpi/historical/ohlcv.json?start=2012-08-01&end=%s' % n_date)
    data = json.loads(r.text)['bpi']
    d0 = pd.DataFrame.from_dict(data, orient='index')
    d0.index = pd.DatetimeIndex(d0.index).rename('time')
    return d0


def load_blockchain_data(timespan='all'):
    return pd.concat([load_blockchain_series(r, a, timespan) for r,a in 
                      [('market-price', 'close'),
                       ('trade-volume', 'volume'),
                       ('n-transactions', 'tr_per_day'),
                       ('n-unique-addresses', 'uniq_addr'),
                       ('output-volume', 'outvolume'),
                       ('estimated-transaction-volume-usd', 'e_tr_vol_usd'),
                       ('estimated-transaction-volume', 'e_tr_vol'),
                       ('miners-revenue', 'revenue'),
                       ('utxo-count', 'utxo_count'),
                      ]], axis=1)


def load_bitcoinity_series(rep, exchange=None, alias=None):
    url = "https://data.bitcoinity.org/export_data.csv?data_type=%s&r=day&t=l&timespan=all" % rep
    print('Loading [%s] ...' % rep)
    
    r = requests.get(url)
    if r.status_code == 200:
        data = pd.read_csv(StringIO(r.text), header=0, parse_dates=True, index_col='Time')
        data.index = data.index.rename('time')
            
        if exchange is not None and exchange in data.columns:
            if alias is None:
                alias = exchange
                
            data = data[exchange].rename(alias)
        else:
            alias = rep if alias is None else alias
            if data.shape[1]==1:
                data = data.rename(columns={data.columns[0]:alias})
        return data
    else:
        raise ValueError("Can't load data: %s" % rep)
