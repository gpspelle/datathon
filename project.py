import pandas as pd
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pickle
import os

from matplotlib import pyplot as plt

def calculate_derivate(serie, start, end):
    return (serie[end] - serie[start])/((end - start) * (serie[end] + serie[start]/2) )
    

def estimate(y0, x0, x1, m):
    return m * (x1 - x0) + y0


correlation_dataframe = pd.read_csv("correlation_data.csv")
timeshift_dataframe = pd.read_csv("correlation_shift.csv")


with open("crypto_data.pkl", "rb") as f:
    crypto_data = pickle.load(f)

crypto_list = list(crypto_data.keys())

for cryptocurrency in crypto_list:

    with open("crypto_data.pkl", "rb") as f:
        crypto_data = pickle.load(f)

    crypto_values = crypto_data[cryptocurrency]

    correlation_values = correlation_dataframe[cryptocurrency]
    correlation_shift_values = timeshift_dataframe[cryptocurrency]

    crypto_positive = []
    crypto_negative = []

    crypto_correlate = zip(correlation_values, crypto_list)

    print("[.] Filter only strongly negative and positive correlated with positive time shift")
    index = 0
    for correlate, crypto in crypto_correlate:

        if correlate > 0.75 and correlation_shift_values[index] > 0:
            crypto_positive.append([crypto, correlate, index, correlation_shift_values[index]])
        elif correlate < -0.75 and correlation_shift_values[index] > 0:
            crypto_negative.append([crypto, correlate, index, correlation_shift_values[index]])

        index += 1

    print(cryptocurrency, crypto_positive, crypto_negative)

    # a negative time shift means that the first input  is shifted to the left 
    # then we can't use negative time shift to predict the first one. because the first
    # one is "happenning" before

    # a positive time shift shows that the second line is showing the future
    # of the first one

    sum_ = 0
    print("[.] Shift the POSITIVE correlated")
    biggest = np.inf
    for crypto, correlate, index, shift in crypto_positive:
        crypto_data[crypto] = crypto_data[crypto][shift:]
        if len(crypto_data[crypto]) < biggest:
            biggest = len(crypto_data[crypto])
        sum_ += correlate
        

    print("[.] Shift the NEGATIVE correlated")
    for crypto, correlate, index, shift in crypto_negative:
        crypto_data[crypto] = crypto_data[crypto][shift:]
        if len(crypto_data[crypto]) < biggest:
            biggest = len(crypto_data[crypto])
        sum_ -= correlate


    if biggest == np.inf:
        continue

    crypto_approx = []
    space = biggest - 1 
    for time in range(space):

        slope = 0
        for crypto, correlate, index, _ in crypto_positive:
            derivate = calculate_derivate(crypto_data[crypto], time, time+1) * correlate
            slope += derivate

        for crypto, correlate, index, _ in crypto_negative:
            derivate = calculate_derivate(crypto_data[crypto], time, time+1) * correlate
            slope += derivate

        slope /= sum_ 
        y1 = estimate(crypto_values[time], time, time+1, slope) 
        crypto_approx.append(y1)

    fig, ax = plt.subplots()

    ax.plot(range(space), crypto_approx, label='approximate')
    ax.plot(range(space), crypto_values[:space], label='real')
    ax.legend()
    plt.suptitle("Cryptocurrency prediction value of " + cryptocurrency)
    plt.savefig("predictions/" + cryptocurrency + "-prediction.png", format='png')
    plt.close()
