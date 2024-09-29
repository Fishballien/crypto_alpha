# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 18:58:58 2024

@author: Xintang Zheng

"""
# %% imports 
from datetime import datetime
import numpy as np
import toml
import pickle


from dirs import PARAM_DIR, DATA_DIR, TWAP_PRICE_DIR
# from twap_price_processor import calc_twap_price_one_symbol
from twap_price_processor import process_multi_symbol


# %% 
process_name = 'twd30_sp15'


# %%
timeline_params = {
    'start_date': datetime(2021, 1, 15),
    'end_date': datetime(2024, 5, 1), #3.29
    'data_start_date': datetime(2021, 1, 1),
    'data_end_date': datetime(2024, 5, 2), #3.30
    }


eva_params = toml.load(PARAM_DIR / 'twap' / f'{process_name}.toml')


dtypes = {
    't': np.dtype([
        ('timestamp', 'M8[ms]'),
        ]),
    'tick': np.dtype([
        ('midprice', 'f8'),
        ]),
    'twap_price': np.dtype([
        ('twap_price', 'f8'),
        ]),
    }

    
with open('symbols.pickle', 'rb') as f:
    symbol_list = pickle.load(f)

            
# %%
if __name__=='__main__':
    max_workers = 100
    process_multi_symbol(process_name=process_name, symbol_list=symbol_list,
                         timeline_params=timeline_params, 
                         eva_params=eva_params, dtypes=dtypes,
                         data_dir=DATA_DIR, result_dir=TWAP_PRICE_DIR, 
                         max_workers=max_workers,
                         )
