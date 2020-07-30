# -*- coding: utf-8 -*-
"""
Created on Thu Jul 30 17:12:24 2020

@author: Generating white noise 
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats


def autocorrelation(data): 
    N = len(data)
    X = np.fft.fft(data - data.mean())
    R = np.real(np.fft.ifft(X * X.conj()))
    return R / N

def QQquantiles(data, plot = False):
    N = len(data)
    normal_samples = np.random.normal(data.mean(), data.std(), N)
    quant = np.linspace(0, 1, N)
    normal_quantiles = np.zeros(N)
    sample_quantiles = np.zeros(N)
    for i in range(N):
        normal_quantiles[i] = np.quantile(normal_samples, quant[i])
        sample_quantiles[i] = np.quantile(data, quant[i])
    if plot: plt.plot(normal_quantiles, sample_quantiles)
    return normal_quantiles, sample_quantiles
        

def markowRealization(size, scale): 
    x = np.zeros(size)
    x[0] = np.random.normal(0, scale)
    for i in range(1, size):
        x[i] = x[i - 1] + np.random.normal(0, scale)
    return x


def whitenoise(size, scale): 
    return np.random.normal(0, scale, size)


if __name__ == '__main__': 
    datas = pd.read_csv('S&P.csv', header = 0, index_col = 0)
    datas.index = pd.to_datetime(datas.index)
    data = np.array(datas['1980':'2000']['Close'])
    variation = (data[1:] - data[:-1]) / data[1:]
