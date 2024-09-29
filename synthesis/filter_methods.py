# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 17:14:09 2024

@author: Xintang Zheng

"""
# %% imports


# %%
def filter_func_v0(data):
    return data.apply(lambda r:
                      (r['sharpe_ratio'] > 1)
                      , 1)
        
        
def filter_func_v1(data):
    return data.apply(lambda r:
                      (r['sharpe_ratio'] > 1.5)
                      & (r['calmar_ratio'] > 4)
                      , 1)
        

def filter_func_v2(data):
    return data.apply(lambda r:
                      (r['sharpe_ratio'] > 3)
                      & (r['calmar_ratio'] > 4)
                      , 1)
        
        
def filter_func_v3(data):
    return data.apply(lambda r:
                      (r['sharpe_ratio'] > 2)
                      & (r['calmar_ratio'] > 3)
                      , 1)
        
        
def filter_func_v4(data):
    return data.apply(lambda r:
                      (r['sharpe_ratio'] > 2.5)
                      & (r['calmar_ratio'] > 4)
                      , 1)
        
        
def filter_func_v5(data):
    return data.apply(lambda r:
                      (r['sharpe_ratio'] > 3.5)
                      & (r['calmar_ratio'] > 7)
                      , 1)
        

def filter_func_v6(data):
    return data.apply(lambda r:
                      (r['sharpe_ratio'] > 2.5)
                      & (r['calmar_ratio'] > 4)
                      & (r['burke_ratio'] > 15)
                      , 1)
        
        
def filter_func_v7(data):
    return data.apply(lambda r:
                      (r['sharpe_ratio'] > 2.5)
                      & (r['calmar_ratio'] > 4)
                      & (r['burke_ratio'] > 15)
                      & (r['return_annualized'] / r['hsr'] > 0.3)
                      , 1)
        
        
def filter_func_v8(data):
    return data.apply(lambda r:
                      (r['sharpe_ratio'] > 2.5)
                      & (r['calmar_ratio'] > 4)
                      & (r['burke_ratio'] > 15)
                      & (r['hsr'] < 0.3)
                      , 1)
        
        
def filter_func_v9(data):
    return data.apply(lambda r:
                      (r['sharpe_ratio'] > 2.5)
                      & (r['calmar_ratio'] > 4)
                      & (r['burke_ratio'] > 15)
                      & (r['hsr'] < 0.2)
                      , 1)
        
        
def filter_func_v10(data):
    return data.apply(lambda r:
                      (r['sharpe_ratio'] > 3)
                      & (r['calmar_ratio'] > 4)
                      & (r['burke_ratio'] > 15)
                      , 1)
        

def filter_func_v11(data):
    return data.apply(lambda r:
                      (r['sharpe_ratio'] > 3.5)
                      & (r['calmar_ratio'] > 4)
                      & (r['burke_ratio'] > 15)
                      , 1)
        
        
def filter_func_v12(data):
    return data.apply(lambda r:
                      (r['sharpe_ratio'] > 2.5)
                      & (r['bin_125_sharpe_ratio'] > 2.5)
                      & (r['calmar_ratio'] > 4)
                      & (r['burke_ratio'] > 15)
                      , 1)
        
        
def filter_func_v13(data):
    return data.apply(lambda r:
                      (r['bin_125_sharpe_ratio'] > 2.5)
                      & (r['bin_125_calmar_ratio'] > 4)
                      & (r['bin_125_burke_ratio'] > 15)
                      , 1)
        
        
def filter_func_v6_gp1(data):
    return data.apply(lambda r:
                      (r['sharpe_ratio_1'] > 2.5)
                      & (r['calmar_ratio_1'] > 4)
                      & (r['burke_ratio_1'] > 15)
                      , 1)
        
        
def filter_func_rec_v0(data):
    return data.apply(lambda r:
                      (r['sharpe_ratio'] > 0)
                      & (r['calmar_ratio'] > 0)
                      & (r['burke_ratio'] > 10)
                      , 1)
        
        
def filter_func_rec_v1(data):
    return data.apply(lambda r:
                      (r['sharpe_ratio'] > 0)
                      & (r['bin_125_sharpe_ratio'] > 0)
                      & (r['calmar_ratio'] > 0)
                      & (r['burke_ratio'] > 10)
                      , 1)
        
        
def filter_func_rec_v2(data):
    return data.apply(lambda r:
                      (r['bin_125_sharpe_ratio'] > 0)
                      & (r['bin_125_calmar_ratio'] > 0)
                      & (r['bin_125_burke_ratio'] > 10)
                      , 1)

        
def filter_func_rec_v0_gp1(data):
    return data.apply(lambda r:
                      (r['sharpe_ratio_1'] > 0)
                      & (r['calmar_ratio_1'] > 0)
                      & (r['burke_ratio_1'] > 10)
                      , 1)
        
    
def filter_func_restrict_diff_with_lag_1_v0(data):
    return data.apply(lambda r:
                      (r['sharpe_ratio'] > 1.5)
                      & (r['diff_with_lag_1'] < 0.4)
                      , 1)
        
        
def filter_func_restrict_diff_with_lag_1_v1(data):
    return data.apply(lambda r:
                      (r['sharpe_ratio'] > 1.5)
                      & (r['diff_with_lag_1'] < 0.8)
                      , 1)

        
def filter_func_restrict_ic_v0(data):
    return data.apply(lambda r:
                      (r['sharpe_ratio'] > 1.5)
                      & (r['ic_240min_with_direction'] > 0.5)
                      , 1)

        
def filter_func_restrict_ic_v1(data):
    return data.apply(lambda r:
                      (r['sharpe_ratio'] > 1.5)
                      & (r['ic_240min_with_direction'] > 1)
                      & (r['quantile_performance'] > 0.7)
                      , 1)
        
        
# %% in pool
def filter_func_in_pool_v0(data):
    thresh = data['predict'].quantile(0.5)
    return data['predict'] >= thresh


def filter_func_in_pool_v1(data):
    thresh = data['predict'].quantile(0.9)
    return data['predict'] >= thresh


def filter_func_in_pool_v2(data):
    thresh = data['predict'].quantile(0.98)
    return data['predict'] >= thresh
