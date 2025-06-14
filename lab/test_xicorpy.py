# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 12:59:55 2024

@author: Xintang Zheng

"""
# %% imports
import pandas as pd


import warnings
warnings.filterwarnings("ignore")


from algo import parallel_xi_correlation, xi_correlation


# %%
x = pd.DataFrame({
    "x_1": [10, 8, 13, 9, 11, 14, 6, 4, 12, 7, 5],
    "x_2": [10, 8, 13, 9, 11, 14, 6, 4, 12, 7, 5],
    "x_3": [10, 8, 13, 9, 11, 14, 6, 4, 12, 7, 5],
    "x_4": [8, 8, 8, 8, 8, 8, 8, 19, 8, 8, 8],
})
y = pd.DataFrame({
    "y_1": [8.04, 6.95, 7.58, 8.81, 8.33, 9.96, 7.24, 4.26, 10.84, 4.82, 5.68],
    "y_2": [9.14, 8.14, 8.74, 8.77, 9.26, 8.1, 6.13, 3.1, 9.13, 7.26, 4.74],
    "y_3": [7.46, 6.77, 12.74, 7.11, 7.81, 8.84, 6.08, 5.39, 8.15, 6.42, 5.73],
    "y_4": [6.58, 5.76, 7.71, 8.84, 8.47, 7.04, 5.25, 12.5, 5.56, 7.91, 6.89],
})


# %%
if __name__=='__main__':
    # a = parallel_xi_correlation(x, y)
    a = xi_correlation(x, y)