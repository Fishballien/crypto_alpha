# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 18:58:58 2024

@author: Xintang Zheng

"""
# %% imports 
import sys
from datetime import datetime
import numpy as np
import toml
from functools import partial
from tqdm import tqdm
import pickle
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import warnings
warnings.filterwarnings("ignore")


from dirs import PARAM_DIR, DATA_DIR, PROCESSED_DATA_DIR
from factor_processor_by_re import process_one_factor_multi_symbol


# %% 
process_name = sys.argv[1]
# process_name = 'fma1_sp15_rp10'
# process_name = 'fma15_sp15'
# process_name = 'fma30_sp15'
# process_name = 'fma60_sp15'
# process_name = 'fma120_sp15'
# process_name = 'ma15_sp15'
# process_name = 'ma240_sp15'
# process_name = 'ma1440_sp15'
# process_name = 'ma1440_sp240'
# process_name = 'mskew240_sp15'
# process_name = 'mkurt240_sp15'
# process_name = 'mzscr240_sp15'
# process_name = 'diff1_mstd240_sp15'
# process_name = 'ma1200_sp15'
# process_name = 'mskew1200_sp15'
# process_name = 'mkurt1200_sp15'
# process_name = 'mzscr1200_sp15'
# process_name = 'diff1_mstd1200_sp15'


# %%
timeline_params = {
    'start_date': datetime(2021, 1, 15),
    'end_date': datetime(2024, 3, 29),
    'data_start_date': datetime(2021, 1, 1),
    'data_end_date': datetime(2024, 3, 30),
    }


eva_params = toml.load(PARAM_DIR / 'processing' / f'{process_name}.toml')


dtypes = {
    't': np.dtype([
        ('timestamp', 'M8[ms]'),
        ]),
    'factor': np.dtype([
        ('factor', 'f8'),
        ]),
    'factor_ma': np.dtype([
        ('factor_ma', 'f8'),
        ]),
    'factor_rp': np.dtype([
        ('factor_rp', 'f8'),
        ]),
    }


with open('factors.pickle', 'rb') as f:
    factor_name_list = pickle.load(f)
    
    
with open('symbols.pickle', 'rb') as f:
    symbol_list = pickle.load(f)
    
    
# %% test
# =============================================================================
# timeline_params = {
#     'start_date': datetime(2023, 10, 16),
#     'end_date': datetime(2023, 10, 16),
#     'data_start_date': datetime(2023, 10, 15),
#     'data_end_date': datetime(2023, 10, 17),
#     }
# 
# symbol_list = ['btcusdt', 'ethusdt']
# =============================================================================


# %%
def process_multi_factor_multi_symbol(process_name, factor_name_list, symbol_list,
                                      timeline_params=timeline_params, 
                                      eva_params=eva_params, dtypes=dtypes,
                                      max_workers=None):
    process_func = partial(process_one_factor_multi_symbol, process_name=process_name,
                           symbol_list=symbol_list,
                           timeline_params=timeline_params, 
                           eva_params=eva_params, dtypes=dtypes, 
                           data_dir=DATA_DIR,
                           result_dir=PROCESSED_DATA_DIR,
                           max_workers=max_workers)
    for factor_name in factor_name_list:
        process_func(factor_name=factor_name)
            
            
# %%
if __name__=='__main__':
    max_workers = 30
    process_multi_factor_multi_symbol(process_name, factor_name_list, symbol_list,
                                      timeline_params=timeline_params, 
                                      eva_params=eva_params, dtypes=dtypes,
                                      max_workers=max_workers,
                                      )
