# -*- coding: utf-8 -*-
"""
Created on Thu May  9 12:45:11 2024

@author: Xintang Zheng

"""

import numba as nb
import numpy as np
import pandas as pd

@nb.njit("float64[:](float64[:], int64)")
def ma_r(arr, window): # by zxt
    res = np.empty(len(arr))
    ma = 0
    for i, v in enumerate(arr):
        if i <= window:
            ma = ma*i/(i+1) + v/(i+1)
        else:
            ma += v/window
            ma -= arr[i-window-1]/window
        res[i] = ma
    return res


@nb.njit("float64[:](float64[:], int64)")
def ma(arr, window): # by zxt
    res = np.empty(len(arr))
    ma = 0
    for i, v in enumerate(arr):
        if i < window:
            ma = ma*i/(i+1) + v/(i+1)
        else:
            ma += v/window
            ma -= arr[i-window]/window
        res[i] = ma
    return res


wd = 15
a = np.cumsum(np.random.normal(0.05, 1, size=1440))
df = pd.DataFrame()
df['org'] = a
df['ma_r'] = ma_r(a, wd)
df['ma'] = ma(a, wd)
df.plot()