# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 17:14:09 2024

@author: Xintang Zheng

"""
# %% imports


# %%
def filter_func_v0(data):
    return data.apply(lambda r:
                      (r['pred1month_sharpe_ratio_0'] > 0.1)
                      , 1)
        
        
def filter_func_v1(data):
    return data.apply(lambda r:
                      (r['pred1month_sharpe_ratio_0'] > 0.04)
                      , 1)
        
        
def filter_func_v2(data):
    return data.apply(lambda r:
                      (r['r_pred1month_sharpe_ratio_0_div_hsr'] > 0.04)
                      , 1)