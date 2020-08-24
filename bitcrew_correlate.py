import csv
import bisect
from datetime import datetime, timedelta
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller
import scipy
from scipy.signal import correlate
import seaborn as sns
import os
import time

def correct_timestamp(time_btc, time_coin, val_coin):
    missing_time_stamps = [item for item in time_btc if item not in time_coin]

    for m in missing_time_stamps:
        m_position = bisect.bisect(time_coin, m)
        time_coin.insert(m_position, m)
        val_coin.insert(m_position, np.nan)

    s = pd.Series(val_coin)
    s = s.interpolate()
    val_coin = s.tolist()

    print(time_btc, time_coin, val_coin)

    return time_coin, val_coin
 

crypto_list = [i[:3] for i in os.listdir("bigdata_hour_2") if i[-4:] == ".pkl"]

print("[.] Crypto list:", crypto_list)
print("[.] Crypto list len:", len(crypto_list))

crypto_data = {}
crypto_timestamp = {}

with open("bigdata_hour_3/btcusd_data.pkl", "rb") as f:
    x = pickle.load(f)
    timestamp_btc = [row[0] for row in x]

timestamp_range = len(timestamp_btc)

for cryptocurrency in crypto_list:
    print("bigdata_hour_3/" + cryptocurrency + "usd_data.pkl")
    with open("bigdata_hour_3/" + cryptocurrency + "usd_data.pkl", "rb") as f:
        x = pickle.load(f)

    timestamp = [row[0] for row in x]
    data = [row[1] for row in x]

    crypto_timestamp_range = len(timestamp) 

    if crypto_timestamp_range >= 0.8 * timestamp_range:
        timestamp, data = correct_timestamp(timestamp_btc, timestamp, data)
        crypto_timestamp[cryptocurrency] = timestamp 
        crypto_data[cryptocurrency] = data 


with open("crypto_data.pkl", "wb") as f:
    pickle.dump(crypto_data, f, protocol=pickle.HIGHEST_PROTOCOL)

crypto_list = list(crypto_data.keys()) # update crypto_list

index = 0
cryptocurrency_correlation = {}
correlation_dataframe = pd.DataFrame(columns= crypto_list)
correlation_timedelta_dataframe = pd.DataFrame(columns= crypto_list)

count = 0

for cryptocurrency_1 in crypto_list:
    data = []
    arg_d = []
    for cryptocurrency_2 in crypto_list:
        count += 1
        start = time.time()

        try:
            flat_crypto_1 = crypto_data[cryptocurrency_1]
            a = (flat_crypto_1 - np.mean(flat_crypto_1))/(np.std(flat_crypto_1)*len(flat_crypto_1))
            flat_crypto_2 = crypto_data[cryptocurrency_2]
            b = (flat_crypto_2 - np.mean(flat_crypto_2))/np.std(flat_crypto_2)
        except ZeroDivisionError as e:
            print(str(100*count/len(crypto_list)**2) + "%", time.time() - start, cryptocurrency_1, cryptocurrency_2)
            data.append(0)
            arg_d.append(0)
            continue

        if cryptocurrency_1 == cryptocurrency_2:
            data.append(0)
            arg_d.append(0)
        else:

            nsamples = len(flat_crypto_1)
            dt = np.arange(1-nsamples, nsamples)

            xcorr = correlate(a, b, method='fft', mode='same')

            max_pos = xcorr.argmax()
            min_pos = xcorr.argmin()

            max_val = xcorr[max_pos]
            min_val = xcorr[min_pos]

            if max_val > -min_val:
                data.append(max_val)
                #recovered_time_shift_max = dt[max_pos]
                #arg_d.append(recovered_time_shift_max)
                arg_d.append(max_pos)
            else:
                data.append(min_val)
                #recovered_time_shift_min = dt[min_pos]
                #arg_d.append(recovered_time_shift_min)
                arg_d.append(min_pos)
        
        print(str(100*count/len(crypto_list)**2) + "%", time.time() - start, cryptocurrency_1, cryptocurrency_2)

    correlation_dataframe.loc[index] = data
    correlation_timedelta_dataframe.loc[index] = arg_d
    index += 1

correlation_dataframe.to_csv("correlation_data.csv", index=False)
correlation_timedelta_dataframe.to_csv("correlation_shift.csv", index=False)
