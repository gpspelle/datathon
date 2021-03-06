import pandas as pd
import seaborn as sns
import os
from matplotlib import pyplot as plt
import pickle

with open("crypto_data.pkl", "rb") as f:
    crypto_data = pickle.load(f) 

crypto_list = list(crypto_data.keys())
correlate = pd.read_csv("correlation_data.csv")
shift = pd.read_csv("correlation_shift.csv")

# plot
sns.set_style('ticks')
fig, ax = plt.subplots()
# the size of A4 paper
fig.set_size_inches(11.7, 8.27)

plt.suptitle("HeatMap of Correlation between Crypto-Currencies")
sns.heatmap(correlate, yticklabels=crypto_list)
plt.savefig("HeatMap_of_Correlation_between_Crypto-Currencies.png", format='png')
plt.close()

# plot
sns.set_style('ticks')
fig, ax = plt.subplots()
# the size of A4 paper
fig.set_size_inches(11.7, 8.27)

plt.suptitle("Time Shift between cryptocurrencies heatmap")
sns.heatmap(shift, yticklabels=crypto_list)
plt.savefig("HeatMap_of_Time_shift_between_Crypto-Currencies.png", format='png')
plt.close()


