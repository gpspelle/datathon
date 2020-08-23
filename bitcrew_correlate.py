import csv
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

def calc_correlation(x, y, t=1):
    return numpy.corrcoef(numpy.array([x[:-t], x[t:]]))


selected_crypto_symbol = pd.read_csv("selected_crypto.csv").symbol.values

crypto_list = [i[:-4] for i in os.listdir("cryptocurrency_data") if "usd.csv" in i and (i[:3] in selected_crypto_symbol or i[:4] in selected_crypto_symbol)]

print("[.] Crypto list:", crypto_list)
print("[.] Crypto list len:", len(crypto_list))

crypto_data = {}
count = 0
last_time = pd.read_csv("cryptocurrency_data/"+ "btcusd" +  ".csv").tail(1).time.values[0]/10000
now = datetime(2020, 8, 18)
one_week = (now - timedelta(days = 3))
last_week_timestamp =  datetime.timestamp(one_week)/10
time_range = np.arange(last_time, last_week_timestamp, -6)
timestamp_dataframe = pd.DataFrame(time_range, columns = ['time'])
for cryptocurrency in crypto_list:
    count+=1
    dataframe = pd.read_csv("cryptocurrency_data/"+ cryptocurrency + '.csv')
    dataframe.time = dataframe.time.map(lambda x: x/10000)
    dataframe = timestamp_dataframe.join(dataframe, how='left', lsuffix="_caller", rsuffix="_other")
    dataframe.interpolate(inplace = True)
    crypto_data[cryptocurrency] = dataframe['close']

index = 0
cryptocurrency_correlation = {}
correlation_dataframe = pd.DataFrame(columns= crypto_list)
correlation_timedelta_dataframe = pd.DataFrame(columns= crypto_list)
begin_of_time = datetime.fromtimestamp(0)

count = 0

for cryptocurrency_1 in crypto_list:
    data = []
    arg_d = []
    for cryptocurrency_2 in crypto_list:
        count += 1
        start = time.time()

        flat_crypto_1 = crypto_data[cryptocurrency_1].values
        a = (flat_crypto_1 - np.mean(flat_crypto_1))/(np.std(flat_crypto_1)*len(flat_crypto_1))
        flat_crypto_2 = crypto_data[cryptocurrency_2].values
        b = (flat_crypto_2 - np.mean(flat_crypto_2))/np.std(flat_crypto_2)

        if cryptocurrency_1 == cryptocurrency_2:
            data.append(1)
            arg_d.append(0)
        else:

            nsamples = len(flat_crypto_1)
            dt = np.arange(1-nsamples, nsamples)

            xcorr = correlate(a, b, method='fft')

            max_pos = xcorr.argmax()
            min_pos = xcorr.argmin()

            max_val = xcorr[max_pos]
            min_val = xcorr[min_pos]

            if max_val > -min_val:
                data.append(max_val)
                recovered_time_shift_max = dt[max_pos]
                arg_d.append(recovered_time_shift_max)
            else:
                data.append(min_val)
                recovered_time_shift_min = dt[min_pos]
                arg_d.append(recovered_time_shift_min)
        
        print("Not found", str(100*count/len(crypto_list)**2) + "%", time.time() - start, cryptocurrency_1, cryptocurrency_2)

    correlation_dataframe.loc[index] = data
    correlation_timedelta_dataframe.loc[index] = arg_d
    index += 1

correlation_dataframe.to_csv("correlation_data.csv", index=False)
correlation_timedelta_dataframe.to_csv("correlation_shift.csv", index=False)


