# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 16:52:37 2024

@author: Xintang Zheng

"""
# %% imports
import pandas as pd
import numpy as np
import blosc2
import pickle
import copy
import os
import h5py


from speedutils import timeit


# %%
path = r'D:\crypto\multi_factor\factor_test_by_alpha\sample_data/bnbusdt.parquet'
path1 = r'D:\crypto\multi_factor\factor_test_by_alpha\sample_data/bnbusdt_s.parquet'
bl2_path = r'D:\crypto\multi_factor\factor_test_by_alpha\sample_data/bnbusdt.bl2'
pkl_path = r'D:\crypto\multi_factor\factor_test_by_alpha\sample_data/bnbusdt.pkl'
h5_path = r'D:\crypto\multi_factor\factor_test_by_alpha\sample_data/bnbusdt.h5'

raw_data = pd.read_parquet(path)
raw_data['t'] = pd.to_datetime(raw_data['timestamp'] / 1000, unit='ms')
raw_data.set_index('t', inplace=True)
agg_funcs = {col: 'last' if col == 'timestamp' else 'mean' for col in raw_data.columns}
data = raw_data.resample('1min').agg(agg_funcs) #.reset_index(drop=True)
data.to_parquet(path1)

# =============================================================================
# to_bl2_data = data
# dtypes = [(col, 'M8[ms]') if col == 'timestamp' else (col, 'f8') for col in to_bl2_data.columns]
# len_data = len(to_bl2_data)
# new_arr = np.empty(len_data, dtype=dtypes)
# for col in to_bl2_data.columns:
#     if col == 'timestamp':
#         new_arr[col] = pd.to_datetime(to_bl2_data['timestamp'] / 1000, unit='ms').values
#         # new_arr[col] = (raw_data[col] / 1000).astype(np.int64)
#     else:
#         new_arr[col] = to_bl2_data[col]
# out = blosc2.save_tensor(new_arr, str(bl2_path), mode='w')
# =============================================================================

'''
%timeit pd.read_parquet(path, columns=['timestamp', 'midprice'])
11.9 ms ± 99.9 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)

%timeit pd.read_parquet(path1, columns=['timestamp', 'midprice'])
25.6 ms ± 54.4 µs per loop (mean ± std. dev. of 7 runs, 10 loops each)

%timeit blosc2.unpack_array2(Path(bl2_path).read_bytes())
23.3 ms ± 284 µs per loop (mean ± std. dev. of 7 runs, 10 loops each)
'''

data_dict = {}
for i in range(1400):
    data_dict[i] = copy.deepcopy(data[['timestamp', 'midprice']])
    
with open(pkl_path, 'wb') as f:
    pickle.dump(data_dict, f)

def load_pkl(pkl_path):
    with open(pkl_path, 'rb') as f:
        a = pickle.load(f)
        
@timeit
def load_and_save_pkl(pkl_path, times):
    data_dict = {}
    for i in range(times):
        if os.path.exists(pkl_path):
            try:
                with open(pkl_path, 'rb') as f:
                    data_dict = pickle.load(f)
            except:
                data_dict = {}
        else:
            data_dict = {}
        data_dict[i] = copy.deepcopy(data[['timestamp', 'midprice']])
        with open(pkl_path, 'wb') as f:
            pickle.dump(data_dict, f)
            
@timeit
def load_and_save_h5(h5_path, times):
    for i in range(times):
        mode = 'w' if os.path.exists(pkl_path) else 'a'
        with h5py.File(h5_path, mode) as hf:
            hf.create_dataset(f'p_{i}', data=copy.deepcopy(data[['timestamp', 'midprice']]).values)