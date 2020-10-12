from tools.utils.utils import mstruct, green, red, yellow
import numpy as np
import pandas as pd

from binance.client import Client
from dateutil import parser
# from ira.utils.MongoController import MongoController
from tqdm.notebook import tqdm
import pytz, time, datetime
import sqlite3
from os.path import join


def binance_client(api_key, secret_key):
    return Client(api_key=api_key, api_secret=secret_key)


def __get_database_path(exchange, timeframe, path='./'):
    return join(path, f'{exchange}_{timeframe.upper()}.db')


def update_binance_data(client, symbol, timeframe='1M', step='4W', timeout_sec=5, path='./'):
    # Load binance data from API endpoint and store data to mongo db
    with sqlite3.connect(__get_database_path('binance', timeframe, path)) as db:
    
        m_table = symbol
        table_exists = db.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{m_table}'").fetchone() is not None
        if table_exists:
            ranges = pd.read_sql_query(f"SELECT min(time) as Start, max(time) as End FROM {m_table}", db)
            start = ranges.End[0]
        else:
            start = '1 Jan 2017 00:00:00'

        tD = pd.Timedelta(timeframe)
        now = (pd.Timestamp(datetime.datetime.now(pytz.UTC).replace(second=0)) - tD).strftime('%d %b %Y %H:%M:%S')
        tlr = pd.DatetimeIndex([start]).append(pd.date_range(start, now, freq=step).append(pd.DatetimeIndex([now])))

        print(f' >> Loading {green(symbol)} {yellow(timeframe)} for [{red(start)}  -> {red(now)}]')
        df = pd.DataFrame()
        s = tlr[0]
        for e in tqdm(tlr[1:]):
            if s + tD < e:
                _start, _stop = (s + tD).strftime('%d %b %Y %H:%M:%S'), e.strftime('%d %b %Y %H:%M:%S')
                nerr = 0
                while nerr < 3:
                    try:
                        chunk = client.get_historical_klines(symbol, timeframe.lower(), _start, _stop)
                        nerr = 100
                    except e as Exception:
                        nerr +=1
                        print(red(str(e)))
                        time.sleep(10)

                if chunk:
        #             print(_start, _stop)
                    data = pd.DataFrame(chunk, columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_av', 'trades', 'tb_base_av', 'tb_quote_av', 'ignore' ])
                    data.index = pd.to_datetime(data['timestamp'].rename('time'), unit='ms')
                    data = data.drop(columns=['timestamp', 'close_time']).astype(float).astype({
                        'ignore': bool,
                        'trades': int,
                    })
                    df = df.append(data)
                    # store to db
                    data.to_sql(m_table, db, if_exists='append', index_label='time')
                    db.commit()
                s = e
                time.sleep(timeout_sec)

    return df


def load_binance_data(symbols, timeframe, start='2017-01-01', end='2100-01-01', path='../data'):
    """
    Loads OHLCV data for symbols (or symbol) from SQLite3 cache
    """
    data = {}
    symbols = [symbols] if isinstance(symbols, str) else symbols
    with sqlite3.connect(__get_database_path('binance', timeframe, path)) as db:
        for s in symbols:
            ds = pd.read_sql_query(f"SELECT * FROM {s.upper()} where time >= '{start}' and time <= '{end}'", db, index_col='time')
            ds.index = pd.DatetimeIndex(ds.index)
            data[s] = ds
    return data