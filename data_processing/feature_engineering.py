# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 15:50:27 2024

@author: Xintang Zheng

"""
# %% imports
import numpy as np
import pandas as pd
import traceback
from datetime import datetime, timedelta
from functools import reduce
from sklearn.preprocessing import QuantileTransformer


# %% old 
# =============================================================================
# # !!!: 会因为数值精度问题产生不一样的结果，例如SDcjbs_ratio_big，df_std可能是全0，也可能是e-15，后者可以计算出norm结果，前者返回None
# def normalization_slow(ogn_fct_df, outlier_n, fillna=True):
#     # a = ogn_fct_df.copy()
#     df_med = ogn_fct_df.median(axis=1)
#     diff_med = (ogn_fct_df.sub(df_med, axis=0)).abs().median(axis=1)
#     # ogn_fct_df = ogn_fct_df.clip(df_med - N * diff_med, df_med + N * diff_med, axis=0)
#     # ogn_fct_df = ogn_fct_df.clip(ogn_fct_df.quantile(0.01, axis=1), ogn_fct_df.quantile(0.99, axis=1), axis=0)
#     if_outlier = (ogn_fct_df.gt(df_med + outlier_n * diff_med, axis=0) 
#                   | ogn_fct_df.lt(df_med - outlier_n * diff_med, axis=0))
#     ogn_fct_df = ogn_fct_df.mask(if_outlier)
#     df_mean = ogn_fct_df.mean(axis=1)
#     ogn_fct_df = ogn_fct_df.sub(df_mean, axis=0)
#     # zscore标准化
#     try:
#         df_std = ogn_fct_df.std(axis=1).replace(0, np.nan)
#         std_fct_df = ogn_fct_df.div(df_std, axis=0).replace([np.inf, -np.inf], np.nan).fillna(0)
#     except:
#         return None #ogn_fct_df.replace([np.inf, -np.inf], np.nan).fillna(0)
#     return std_fct_df
# =============================================================================


# %% 
def normalization(ogn_fct_df, outlier_n, fillna=True):
    # 将DataFrame转换为NumPy数组
    ogn_fct_arr = ogn_fct_df.to_numpy(dtype=np.float64)

    # 计算中位数和绝对中位差
    df_med = np.nanmedian(ogn_fct_arr, axis=1, keepdims=True)
    diff_med = np.nanmedian(np.abs(ogn_fct_arr - df_med), axis=1, keepdims=True)

    # 标记并掩盖异常值
    upper_bound = df_med + outlier_n * diff_med
    lower_bound = df_med - outlier_n * diff_med
    mask = (ogn_fct_arr < lower_bound) | (ogn_fct_arr > upper_bound)
    ogn_fct_arr[mask] = np.nan

    # 计算均值和标准差，使用ddof=1来匹配pandas的std()
    df_mean = np.nanmean(ogn_fct_arr, axis=1, keepdims=True)
    df_std = np.nanstd(ogn_fct_arr, axis=1, ddof=1, keepdims=True)

    # 防止标准差为零导致的除以零问题
    df_std[df_std == 0] = np.nan

    # 直接在原数组上进行标准化操作
    ogn_fct_arr -= df_mean
    np.divide(ogn_fct_arr, df_std, out=ogn_fct_arr)

    # 处理NaN和无穷大值
    if fillna:
        np.nan_to_num(ogn_fct_arr, nan=0.0, posinf=0.0, neginf=0.0, copy=False)

    # 将结果转换回DataFrame，并保留原始索引和列名
    normalized_df = pd.DataFrame(ogn_fct_arr, index=ogn_fct_df.index, columns=ogn_fct_df.columns)

    return normalized_df


def neutralization(tm, ogn_fct_df, zxh_df, if_jy_df):
    try:
        # if tm > datetime(2021, 2, 16):
        #     breakpoint()
        if_jy = if_jy_df.loc[tm].values
        ogn_fct_on = ogn_fct_df.loc[tm].values[if_jy]
        zxh_meta_T = zxh_df.loc[tm].values.reshape(-1, if_jy.shape[0])
        zxh_met_T = zxh_meta_T[:, if_jy]
        zxh_met_T = zxh_met_T[np.abs(zxh_met_T).sum(axis=1) > 0, :]
        ogn_fct_on_met = ogn_fct_on.reshape(ogn_fct_on.shape[0], 1)
        zxh_met = zxh_met_T.T
        coef = np.linalg.inv((zxh_met_T).dot(zxh_met)).dot(zxh_met_T).dot(ogn_fct_on_met)
        ogn_fct_on_met_zxh = ogn_fct_on_met - zxh_met.dot(coef)
        ogn_fct_on_met_zxh = ogn_fct_on_met_zxh.ravel()
        fct_zxh_n = np.zeros(if_jy.shape)
        fct_zxh_n[if_jy] = ogn_fct_on_met_zxh
        # if tm > datetime(2021, 2, 16):
        #     breakpoint()
        return fct_zxh_n
    except:
        # breakpoint()
        # print(tm)
        traceback.print_exc()
        return np.zeros(if_jy_df.loc[tm].size)
    
    
def neutralization_multi(tm, ogn_fct_df, neu_data_list, if_jy_df, lookback_param):
    try:
        # if tm > datetime(2021, 2, 16):
        #     breakpoint()
        tm_start = tm - timedelta(**lookback_param)
        
        # 取前 n 个截面数据
        if_jy_slice, ogn_fct_on_met, neu_arr, neu_arr_T = prepare_data_for_neu(
            ogn_fct_df, neu_data_list, if_jy_df, tm_start, tm)
        
        if not any(if_jy_slice):
            return np.zeros(if_jy_df.loc[tm].size)
        
        # get coef
        coef = np.linalg.inv((neu_arr_T).dot(neu_arr)).dot(neu_arr_T).dot(ogn_fct_on_met)
        
        # 取 tm 的数据
        if_jy_slice, ogn_fct_on_met, neu_arr, neu_arr_T = prepare_data_for_neu(
            ogn_fct_df, neu_data_list, if_jy_df, tm, tm)
        
        # 计算残差
        ogn_fct_on_met_zxh = ogn_fct_on_met - neu_arr.dot(coef)
        ogn_fct_on_met_zxh = ogn_fct_on_met_zxh.ravel()
        fct_zxh_n = np.zeros(if_jy_slice.shape)
        fct_zxh_n[if_jy_slice] = ogn_fct_on_met_zxh
        return fct_zxh_n
    except:
        breakpoint()
        print(tm)
        traceback.print_exc()
        return np.zeros(if_jy_df.loc[tm].size)
    
    
def prepare_data_for_neu(ogn_fct_df, neu_data_list, if_jy_df, tm_start, tm_end, align=False):
    # align index
    if align:
        if_jy_index = if_jy_df.loc[tm_start:tm_end].index
        ogn_fct_index = ogn_fct_df.loc[tm_start:tm_end].index
        index_list = [if_jy_index, ogn_fct_index]
        for neu_data in neu_data_list:
            index_list.append(neu_data.loc[tm_start:tm_end].index)
        inner_index = reduce(lambda x, y: x.intersection(y), index_list)
    else:
        inner_index = ogn_fct_df.loc[tm_start:tm_end].index
    
    try:
        # 取前 n 个截面数据
        if_jy_slice = if_jy_df.loc[inner_index].values.reshape(-1)
        ogn_fct_slice = ogn_fct_df.loc[inner_index].values.reshape(-1)
        neu_arr = np.empty((if_jy_slice.shape[0], len(neu_data_list)))
        for i_n, neu_data in enumerate(neu_data_list):
            neu_arr[:, i_n] = neu_data.loc[inner_index].values.reshape(-1)
        
        # reshape ogn_fct
        ogn_fct_slice = ogn_fct_slice[if_jy_slice]
    except (IndexError, KeyError):
        return prepare_data_for_neu(ogn_fct_df, neu_data_list, if_jy_df, tm_start, tm_end, 
                                    align=True)
    ogn_fct_on_met = ogn_fct_slice.reshape(ogn_fct_slice.shape[0], 1)
    
    # reshape neu_arr
    neu_arr = neu_arr[if_jy_slice, :]
    neu_arr_T = neu_arr.T
    return if_jy_slice, ogn_fct_on_met, neu_arr, neu_arr_T
        
        
# %% with mask
def normalization_with_mask(data, to_mask, outlier_n):
    data_masked = data.mask(to_mask)
    data_normed = normalization(data_masked, outlier_n)
    data_normed_masked = data_normed.mask(to_mask)
    return data_normed_masked


# %% 分布变换
def quantile_transform_with_nan(arr, n_quantiles=1000, output_distribution='normal', random_state=None):
    """
    对数组中的非NaN值进行Quantile变换，并将变换后的值填回原数组中，保留NaN值。

    Parameters:
    arr (numpy array): 输入的数组，可以包含NaN值。
    n_quantiles (int): 用于QuantileTransformer的分位数数量。
    output_distribution (str): 输出分布类型，'normal'或'uniform'。
    random_state (int or None): 控制随机性。

    Returns:
    numpy array: 变换后的数组，保留了原始NaN值。
    """
    # 创建 QuantileTransformer 实例
    qt = QuantileTransformer(n_quantiles=n_quantiles, output_distribution=output_distribution, random_state=random_state)
    
    # 去掉 NaN 值，只对非 NaN 值进行变换
    non_nan_mask = ~np.isnan(arr)
    non_nan_values = arr[non_nan_mask].reshape(-1, 1)  # 提取非 NaN 值并转换为二维数组
    
    # 对非 NaN 值进行变换
    transformed_values = qt.fit_transform(non_nan_values).flatten()
    
    # 创建结果数组，先复制原始数组
    result = np.copy(arr)
    
    # 将变换后的值填回原数组对应位置
    result[non_nan_mask] = transformed_values
    
    return result
