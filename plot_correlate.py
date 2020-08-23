import pandas as pd
import seaborn as sns
import os
from matplotlib import pyplot as plt

selected_crypto_symbol = pd.read_csv("selected_crypto.csv").symbol.values

crypto_list = [i[:-4] for i in os.listdir("cryptocurrency_data") if "usd.csv" in i and (i[:3] in selected_crypto_symbol or i[:4] in selected_crypto_symbol)]


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


