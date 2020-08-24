import datetime
import time
import bitfinex
import pickle
import pandas as pd
import os
from nomics import Nomics


def fetch_data(start, stop, symbol, interval, tick_limit, step):
    # Create api instance
    api_v2 = bitfinex.bitfinex_v2.api_v2()
    data = []
    res = api_v2.candles(symbol=symbol, interval=interval,
                         limit=tick_limit, start=start,
                         end=stop)

    time.sleep(1)
    res.reverse()

    data.extend(res)
    return data

selected_crypto_symbol = pd.read_csv("selected_crypto.csv").symbol.values

# Set step size
time_step = 60000 * 60 * 24 * 9

# Define the start date 
t_start = datetime.datetime(2019, 5, 15, 0, 0)
t_start = time.mktime(t_start.timetuple()) * 1000

# Define the end date
t_stop = datetime.datetime(2019, 6, 13, 0, 0)
t_stop = time.mktime(t_stop.timetuple()) * 1000

# Define query parameters
bin_size = '1h' # This will return hour data
limit = 1000    # We want the maximum of 1000 data points

index = 0
biggest = 0
for crypto in selected_crypto_symbol:
    pair = crypto + 'usd'

    pair_data = fetch_data(start=t_start, stop=t_stop, symbol=pair,
                           interval=bin_size, tick_limit=limit, 
                           step=time_step)

    print(index / len(selected_crypto_symbol))
    index += 1

    if len(pair_data) == 0:
        continue

    with open("test_pump_eos/" + pair + "_data.pkl", "wb") as f: 
        pickle.dump(pair_data, f, protocol=pickle.HIGHEST_PROTOCOL)
