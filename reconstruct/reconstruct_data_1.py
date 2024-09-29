# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 16:19:02 2024

@author: Xintang Zheng

"""
# %% imports
import os
import pandas as pd
import gc
import pickle
from tqdm import tqdm
from functools import partial
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed


import warnings
warnings.filterwarnings("ignore")


from dirs import DATA_DIR, RE_DATA_DIR


# %%
data_dir = DATA_DIR
re_data_dir = RE_DATA_DIR
n_executor = 50


# %%
def process_one_symbol_one_day(symbol_file, date_dir, re_data_dir):
    symbol_name = symbol_file.split('.')[0]
    symbol_file_path = date_dir / symbol_file
    try:
        raw_data = pd.read_parquet(symbol_file_path)
    except: 
        return 0
    raw_data['t'] = pd.to_datetime(raw_data['timestamp'] / 1000, unit='ms')
    raw_data.set_index('t', inplace=True)
    agg_funcs = {col: 'last' if col == 'timestamp' else 'mean' for col in raw_data.columns}
    data = raw_data.resample('1min').agg(agg_funcs).reset_index(drop=True)
    del raw_data
    gc.collect()
    
    save_single_factor_func = partial(save_single_factor, data=data, symbol_name=symbol_name,
                                      re_data_dir=re_data_dir)
    
    if n_executor is None:
        for factor in data.columns:
            if factor == 'timestamp':
                continue
            save_single_factor_func(factor)
    else:
        all_tasks = []
        with ThreadPoolExecutor(max_workers=n_executor) as executor:
            for factor in data.columns:
                if factor == 'timestamp':
                    continue
                executor.submit(save_single_factor_func, factor)
            for task in as_completed(all_tasks):
                pass
    return 1


def save_single_factor(factor, data, symbol_name, re_data_dir):
    selected_data = data[['timestamp', factor]]
    
    target_dir = re_data_dir / factor
    target_dir.mkdir(parents=True, exist_ok=True)
    target_path = target_dir / f'{symbol_name}.pickle'
    if os.path.exists(target_path):
        try:
            with open(target_path, 'rb') as f:
                res = pickle.load(f)
        except:
            res = {}
    else:
        res = {}
    
    res[date] = selected_data
    with open(target_path, 'wb') as f:
        pickle.dump(res, f)


# %%
with open('reconstructed_dates.pkl', 'rb') as f:
    finished_dates = pickle.load(f)
sorted_dates = sorted(os.listdir(data_dir))
dates_to_re = [dt for dt in sorted_dates if dt not in finished_dates]
print(dates_to_re)

for date in dates_to_re:
    date_dir = data_dir / date
    process_one_symbol_func = partial(process_one_symbol_one_day, date_dir=date_dir, re_data_dir=re_data_dir)
    if os.path.isdir(date_dir):
        # if n_executor is None:
        for symbol_file in tqdm(os.listdir(date_dir), desc=date):
            process_one_symbol_func(symbol_file)
        # else:
        #     with ProcessPoolExecutor(max_workers=n_executor) as executor:
        #         all_tasks = [executor.submit(process_one_symbol_func, symbol_file) for symbol_file in os.listdir(date_dir)]
        #         for task in tqdm(as_completed(all_tasks), total=len(all_tasks), desc=date):
        #             pass
    finished_dates.append(date)
    with open('reconstructed_dates.pkl', 'wb') as f:
        pickle.dump(finished_dates, f)
        
            