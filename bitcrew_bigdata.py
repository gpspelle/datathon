import csv
from datetime import datetime, timedelta
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller
from scipy.stats import pearsonr
import scipy
import seaborn as sns
import os
import time

crypto_list = [i[:-4] for i in os.listdir("cryptocurrency_data") if "usd.csv" in i]
crypto_data = {}
count = 0
last_time = pd.read_csv("cryptocurrency_data/"+ "btcusd" +  ".csv").tail(1).time.values[0]/10000
one_week = (datetime.now() - timedelta(days = 30))
last_week_timestamp =  datetime.timestamp(one_week)/10
time_range = np.arange(last_time, last_week_timestamp, -6)
timestamp_dataframe = pd.DataFrame(time_range, columns = ['time'])
for cryptocurrency in crypto_list:
    count+=1
    dataframe = pd.read_csv("cryptocurrency_data/"+ cryptocurrency + '.csv')
    dataframe.time = dataframe.time.map(lambda x: x/10000)
    dataframe = timestamp_dataframe.join(dataframe, how='left', lsuffix="_caller", rsuffix="_other")
    dataframe.interpolate(inplace = True)
    # log_return = np.log(dataframe['close']).diff().dropna()
    # if adfuller(log_return)[1] < 0.01:
    print(cryptocurrency + " é estacionária", str(100*count/len(crypto_list)) + "%")
        # crypto_data[cryptocurrency] = pd.DataFrame(log_return)
    crypto_data[cryptocurrency] = dataframe['close']
index = 0
cryptocurrency_correlation = {}
correlation_dataframe = pd.DataFrame(columns= crypto_list)
correlation_timedelta_dataframe = pd.DataFrame(columns= crypto_list)
begin_of_time = datetime.fromtimestamp(0)

count = 0
calculated = {}

for cryptocurrency_1 in crypto_list:
    data = []
    arg_d = []
    for cryptocurrency_2 in crypto_list:
        count += 1
        start = time.time()

        if cryptocurrency_2 + cryptocurrency_1 in calculated:
            data.append(calculated[cryptocurrency_1 + cryptocurrency_2][0])
            arg_d.append(calculated[cryptocurrency_1 + cryptocurrency_2][1])
        else:
            flat_crypto_1 = crypto_data[cryptocurrency_1].values
            a = (flat_crypto_1 - np.mean(flat_crypto_1))/(np.std(flat_crypto_1)*len(flat_crypto_1))
            flat_crypto_2 = crypto_data[cryptocurrency_2].values
            b = (flat_crypto_2 - np.mean(flat_crypto_2))/np.std(flat_crypto_2)

            if cryptocurrency_1 == cryptocurrency_2:
                data.append(1)
                arg_d.append(0)
            else:
                d = np.correlate(a, b, mode = 'same')
                if max(d) > -min(d):
                    data.append(max(d))
                    arg_d.append((datetime.fromtimestamp(6*np.argmax(d)) - begin_of_time).total_seconds()/3600)
                else:
                    data.append(min(d))
                    arg_d.append((datetime.fromtimestamp(6*np.argmin(d)) - begin_of_time).total_seconds()/3600)
            
            calculated[cryptocurrency_1 + cryptocurrency_2] = [data[:-1], arg_d[:-1]]
            calculated[cryptocurrency_2 + cryptocurrency_1] = [data[:-1], arg_d[:-1]]

        print(str(100*count/len(crypto_list)**2) + "%", time.time() - start)

    correlation_dataframe.loc[index] = data
    correlation_timedelta_dataframe.loc[index] = arg_d
    index += 1
correlation_dataframe.to_csv("correlation_data.csv")

plt.suptitle("HeatMap of Correlation between Crypto-Currencies")
sns.heatmap(correlation_dataframe, yticklabels = crypto_list)
plt.savefig("HeatMap_of_Correlation_between_Crypto-Currencies.png", format='png')
plt.close()

plt.suptitle("Time Shift between cryptocurrencies heatmap")
sns.heatmap(correlation_timedelta_dataframe, yticklabels = crypto_list)
plt.savefig("HeatMap_of_Time_shift_between_Crypto-Currencies.png", format='png')
plt.close()

