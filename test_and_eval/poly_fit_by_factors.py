# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 10:37:19 2024

@author: Xintang Zheng

"""
# %% imports
import sys
import signal
import pickle
from tqdm import tqdm
import toml
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import warnings
warnings.filterwarnings("ignore")


from dirs import PROCESSED_DATA_DIR, TWAP_PRICE_DIR, RESULT_DIR, PARAM_DIR
from poly_fit import PolyFit


# %% init
factor_data_dir = PROCESSED_DATA_DIR
twap_data_dir = TWAP_PRICE_DIR
result_dir = RESULT_DIR
param_dir = PARAM_DIR

process_name = 'ma15_sp240'


# %% params
poly_name = 'poly_1stp_1y_3_sp240'
poly_params = toml.load(param_dir / 'poly' / f'{poly_name}.toml')
rolling_params_fitted = {
    'fstart': datetime(2021, 1, 1),
    'pstart': datetime(2022, 1, 1),
    'puntil': datetime(2024, 3, 29),
    'end_by': 'time'
    }


# %% load factors
with open('factors.pickle', 'rb') as f:
    factor_name_list = pickle.load(f)
    
    
# %% multi
def poly_fit_multi_factors(poly_fit, process_name, factor_name_list, executor=None):
    if executor is None:
        for factor_name in tqdm(factor_name_list, desc='multi_factors'):
        # for factor_name in factor_name_list:
            poly_fit.rolling_fit(process_name, factor_name)
            # print(factor_name)
    else:
        all_tasks = [executor.submit(poly_fit.rolling_fit, process_name, factor_name)
                     for factor_name in factor_name_list]
        num_of_success = 0
        for task in tqdm(as_completed(all_tasks), total=len(all_tasks), desc='multi_factors'):
            res = task.result()
            if res:
                num_of_success += 1
        print(f'num_of_success: {num_of_success}, num_of_failed: {len(factor_name_list)-num_of_success}')
        
        
# %% signal
def signal_handler(sig, frame):
    if executor:
        executor.shutdown(wait=False)
    sys.exit(0)

        
# %% main
if __name__=='__main__':
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGILL, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)
    
    poly_fit = PolyFit(poly_name, poly_params, rolling_params_fitted)
    executor = ProcessPoolExecutor(max_workers=30)
    poly_fit_multi_factors(poly_fit, process_name, factor_name_list, 
                           executor=executor,
                          )
    executor.shutdown()