# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 13:29:25 2024

@author: Xintang Zheng

"""
# %% imports
import numba as nb
import numpy as np
import pandas as pd
import xicorpy
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed


# %%
@nb.njit("float64[:](float64[:])")
def advanced_forward_fill(arr):
    """
    Perform an advanced forward fill on a NumPy array.
    
    This function applies a forward fill (ffill) operation to fill missing values (np.nan)
    in the input array. If the initial elements of the array are missing, they are filled with 0.
    Subsequent missing values are filled with the previous non-missing value. If previous values
    are also missing, 0 is used instead.
    
    Parameters:
    -----------
    arr : np.ndarray
        A 1D NumPy array containing missing values represented by np.nan.
        
    Returns:
    --------
    np.ndarray
        A 1D NumPy array with missing values forward filled, and initial missing values filled with 0.
    
    Example:
    --------
    >>> arr = np.array([np.nan, np.nan, np.nan, 3, np.nan])
    >>> filled_arr = advanced_forward_fill(arr)
    >>> print(filled_arr)
    [0. 0. 0. 3. 3.]
    """
    if np.isnan(arr[0]):
        arr[0] = 0
    for i in range(1, len(arr)):
        if np.isnan(arr[i]):
            if np.isnan(arr[i - 1]):
                arr[i] = 0 
            else:
                arr[i] = arr[i - 1]
    return arr


# %%
@nb.njit("float64[:](float64[:], int64, int64)")
def rolling_mean_fut(arr, today_size, wd):
    wd_start_index = 0
    wd_end_index = wd - 1
    res = np.zeros(today_size)
    len_of_arr = len(arr)
    while wd_start_index < today_size and wd_end_index < len_of_arr:
        if wd_start_index == 0:
            ma = sum(arr[wd_start_index:wd_end_index+1]) / wd
        else:
            ma += arr[wd_end_index] / wd
            ma -= arr[wd_start_index-1] / wd
        res[wd_start_index] = ma
        wd_start_index += 1
        wd_end_index += 1
    return res


@nb.njit("float64[:](float64[:], int64)")
def msum(arr, window):
    res = np.empty(len(arr))
    msum = 0
    for i, v in enumerate(arr):
        if i <= window:
            msum = msum + v
        else:
            msum += v
            msum -= arr[i-window]
        res[i] = msum
    return res


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


@nb.njit("float64[:](float64[:], int64)")
def mstd(arr, window): # by zxt
    res = np.empty(len(arr))
    ma_1 = 0
    ma_2 = 0
    for i, v in enumerate(arr):
        if i < window:
            ma_1 = ma_1*i/(i+1) + v/(i+1)
            ma_2 = ma_2*i/(i+1) + v**2/(i+1)
        else:
            ma_1 += v/window
            ma_1 -= arr[i-window]/window
            ma_2 += v**2/window
            ma_2 -= arr[i-window]**2/window
        res[i] = np.sqrt(ma_2 - ma_1 ** 2)
    return res


@nb.njit("float64(float64, float64, float64)")
def skew_simple(ma1, ma2, ma3):
    std = np.sqrt(ma2 - ma1 ** 2)
    return 0 if std == 0 else (ma3 - 3 * ma1 * std ** 2 - ma1 ** 3) / std ** 3


@nb.njit("float64[:](float64[:], int64)")
def mskew(arr, window): # by zxt
    res = np.empty(len(arr))
    ma_1 = 0
    ma_2 = 0
    ma_3 = 0
    for i, v in enumerate(arr):
        if i < window:
            ma_1 = ma_1*i/(i+1) + v/(i+1)
            ma_2 = ma_2*i/(i+1) + v**2/(i+1)
            ma_3 = ma_3*i/(i+1) + v**3/(i+1)
        else:
            ma_1 += v/window
            ma_1 -= arr[i-window]/window
            ma_2 += v**2/window
            ma_2 -= arr[i-window]**2/window
            ma_3 += v**3/window
            ma_3 -= arr[i-window]**3/window
        res[i] = skew_simple(ma_1, ma_2, ma_3)
    return res


# %% moving min/max
@nb.njit #("(float64, int64)(float64[:], int64, int64)")
def get_min_and_arg_once(arr, wd_end_index, wd_start_index):
    v_min = 1e20
    for i in range(wd_start_index, wd_end_index+1):
        if arr[i] <= v_min:
            arg_min = i
            v_min = arr[i]
    return v_min, arg_min

@nb.njit("float64[:](float64[:], int64, int64)")
def get_moving_min(arr, len_today, len_wd):
    v_min_arr = np.empty(len_today, dtype=np.float64)
    len_his = len(arr)
    wd_end_index = len_his - len_today
    wd_start_index = max(wd_end_index - len_wd + 1, 0)
    res_index = 0
    # init
    v_min, arg_min = get_min_and_arg_once(arr, wd_end_index, wd_start_index)
    v_min_arr[res_index] = v_min
    wd_end_index += 1
    res_index += 1
    # rolling
    while wd_end_index < len_his:
        wd_start_index = max(wd_end_index - len_wd + 1, 0)
        if arr[wd_end_index] <= v_min:
            v_min = arr[wd_end_index]
            arg_min = wd_end_index
        elif wd_start_index > arg_min:
            v_min, arg_min = get_min_and_arg_once(arr, wd_end_index, wd_start_index)
        v_min_arr[res_index] = v_min
        wd_end_index += 1
        res_index += 1
    return v_min_arr


@nb.njit("float64[:](float64[:], int64, int64)")
def get_moving_argmin(arr, len_today, len_wd):
    v_argmin_arr = np.empty(len_today, dtype=np.float64)
    len_his = len(arr)
    wd_end_index = len_his - len_today
    wd_start_index = max(wd_end_index - len_wd + 1, 0)
    res_index = 0
    # init
    v_min, arg_min = get_min_and_arg_once(arr, wd_end_index, wd_start_index)
    v_argmin_arr[res_index] = arg_min - wd_end_index
    wd_end_index += 1
    res_index += 1
    # rolling
    while wd_end_index < len_his:
        wd_start_index = max(wd_end_index - len_wd + 1, 0)
        if arr[wd_end_index] <= v_min:
            v_min = arr[wd_end_index]
            arg_min = wd_end_index
        elif wd_start_index > arg_min:
            v_min, arg_min = get_min_and_arg_once(arr, wd_end_index, wd_start_index)
        v_argmin_arr[res_index] = arg_min - wd_end_index
        wd_end_index += 1
        res_index += 1
    return v_argmin_arr


@nb.njit #("(float64, int64)(float64[:], int64, int64)")
def get_max_and_arg_once(arr, wd_end_index, wd_start_index):
    v_max = -1e20
    for i in range(wd_start_index, wd_end_index+1):
        if arr[i] >= v_max:
            arg_max = i
            v_max = arr[i]
    return v_max, arg_max


@nb.njit("float64[:](float64[:], int64, int64)")
def get_moving_max(arr, len_today, len_wd):
    v_max_arr = np.empty(len_today, dtype=np.float64)
    len_his = len(arr)
    wd_end_index = len_his - len_today
    wd_start_index = max(wd_end_index - len_wd + 1, 0)
    res_index = 0
    # init
    v_max, arg_max = get_max_and_arg_once(arr, wd_end_index, wd_start_index)
    v_max_arr[res_index] = v_max
    wd_end_index += 1
    res_index += 1
    # rolling
    while wd_end_index < len_his:
        wd_start_index = max(wd_end_index - len_wd + 1, 0)
        if arr[wd_end_index] >= v_max:
            v_max = arr[wd_end_index]
            arg_max = wd_end_index
        elif wd_start_index > arg_max:
            v_max, arg_max = get_max_and_arg_once(arr, wd_end_index, wd_start_index)
        v_max_arr[res_index] = v_max
        wd_end_index += 1
        res_index += 1
    return v_max_arr


@nb.njit("float64[:](float64[:], int64, int64)")
def get_moving_argmax(arr, len_today, len_wd):
    v_argmax_arr = np.empty(len_today, dtype=np.float64)
    len_his = len(arr)
    wd_end_index = len_his - len_today
    wd_start_index = max(wd_end_index - len_wd + 1, 0)
    res_index = 0
    # init
    v_max, arg_max = get_max_and_arg_once(arr, wd_end_index, wd_start_index)
    v_argmax_arr[res_index] = arg_max - wd_end_index
    wd_end_index += 1
    res_index += 1
    # rolling
    while wd_end_index < len_his:
        wd_start_index = max(wd_end_index - len_wd + 1, 0)
        if arr[wd_end_index] >= v_max:
            v_max = arr[wd_end_index]
            arg_max = wd_end_index
        elif wd_start_index > arg_max:
            v_max, arg_max = get_max_and_arg_once(arr, wd_end_index, wd_start_index)
        v_argmax_arr[res_index] = arg_max - wd_end_index
        wd_end_index += 1
        res_index += 1
    return v_argmax_arr


# %%
def compute_xicor(row_pair):
    return xicorpy.compute_xi_correlation(*row_pair)[0][0]


def parallel_xi_correlation(dfx, dfy):
    with ProcessPoolExecutor() as executor:
        data_pairs = zip(dfx.values, dfy.values)
        results = executor.map(compute_xicor, data_pairs)
    return list(results)


def xi_correlation(dfx, dfy):
    data_pairs = zip(dfx.values, dfy.values)
    results = [compute_xicor(data_pair) for data_pair in data_pairs]
    return results


def calc_corr_daily(dfx, dfy, method='pearson'):
    correlations = []

    for (date, group_x), (_, group_y) in zip(dfx.groupby(dfx.index.date), dfy.groupby(dfy.index.date)):
        concat_data_x = group_x.values.flatten()
        concat_data_y = group_y.values.flatten()
        
        if method == 'pearson':
            corr_matrix = np.corrcoef(concat_data_x, concat_data_y)
            correlation = corr_matrix[0, 1]
        elif method == 'xi':
            correlation = xicorpy.compute_xi_correlation(concat_data_x, concat_data_y)[0][0]
        correlations.append((date, correlation))

    result_series = pd.Series(dict(correlations))
    result_series.index = pd.to_datetime(result_series.index) 
    return result_series


