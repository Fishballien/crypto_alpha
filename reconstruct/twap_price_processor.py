# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 18:58:58 2024

@author: Xintang Zheng

"""
# %% draft
'''
单个币种单个因子回测：
    1. 读数据：
        pandas -> numpy 读不到则当日为空array
        计算valid dates: [n-1, n+1] 都有数据的天
    2. return计算：
        对valid dates: 
            计算rolling未来半小时平均价格
            计算平均价格未来半小时收益率
    3. feature engineering
        计算过去一分钟平均
        计算过去一分钟平均的过去10天内分位数
    4. resample到1min? 15min?
    5. 计算z-score
    6. 计算每日收益率 -> 合并到该factor对应的文件里
'''
# %% TODOs
'''
1. 添加换手计算
2. 添加统计值计算（+/-）
'''
# %% imports 
from datetime import datetime
import pandas as pd
import numpy as np
import toml
from tqdm import tqdm
import traceback
import os
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed



from timeutils import MIN_SEC, DATA_FREQ, DAY_SEC, timestr_to_seconds
from algo import rolling_mean_fut, ma, get_moving_min, get_moving_max
from dirs import PARAM_DIR, TWAP_PRICE_DIR
from feval import TwapProcessor


# %% 
eval_name = 'v0_fma15_twd15_pp15_sp15_rp10'
# factor_name = 'ACTmidpos'
symbol = 'btcusdt'


# %%
# =============================================================================
# timeline_params = {
#     'start_date': datetime(2023, 10, 16),
#     'end_date': datetime(2023, 10, 16),
#     'data_start_date': datetime(2023, 10, 15),
#     'data_end_date': datetime(2023, 10, 17),
#     }
# 
# 
# eva_params = toml.load(PARAM_DIR / f'{eval_name}.toml')
# 
# 
# dtypes = {
#     't': np.dtype([
#         ('timestamp', 'f8'),
#         ]),
#     'tick': np.dtype([
#         ('midprice', 'f8'),
#         ]),
#     'twap_price': np.dtype([
#         ('twap_price', 'f8'),
#         ]),
#     }
# =============================================================================


# %% load data
def load_data_and_check(eva):
    is_valid_arr = load_data(eva)
    get_valid_dates(eva, is_valid_arr)
            

def load_data(eva):
    ''' load info '''
    symbol = eva.symbol
    ''' load params '''
    param = eva.eva_params
    valid_prop_lmt = param['valid_prop_lmt']
    ''' load dtypes '''
    dtypes = eva.dtypes
    t_dtype, tick_dtype = dtypes['t'], dtypes['tick']
    ''' load data or container '''
    t_data = eva.dataset["t"]
    tick_data = eva.dataset["tick"]
    ''' load dates '''
    dates_for_data = eva.timeline["dates_for_data"]
    ''' init '''
    factor = eva.factor
    data_col_list = ['timestamp', 'midprice']
    target_col_list = ['timestamp', 'midprice']
    dtype_list = [t_dtype, tick_dtype]
    data_list = [t_data, tick_data]
    is_valid_arr = np.zeros(len(dates_for_data), dtype=np.int32)
    ''' loop ''' 
    # for i_d, dt in enumerate(tqdm(dates_for_data, desc='load_data')):
    for i_d, dt in enumerate(dates_for_data):
        data = factor.read_oneday(dt, symbol, columns=data_col_list)
        is_not_valid = (data is None 
                        or (not check_if_enough_valid_points(data, 'midprice', valid_prop_lmt))
                        )
        if is_not_valid:
            t_data[dt] = np.zeros(0, dtype=t_dtype)
            tick_data[dt] = np.zeros(0, dtype=tick_dtype)
            continue
        today_size = len(data)
        for data_col, target_col, dtype, target_data in zip(data_col_list, target_col_list,
                                                            dtype_list, data_list):
            x_data_today = np.empty(today_size, dtype=dtype)
            if data_col == 'timestamp':
                x_data_today[target_col] = data[data_col] / 1e3
            else:
                x_data_today[target_col] = data[data_col].ffill().bfill()
            target_data[dt] = x_data_today
        is_valid_arr[i_d] = 1
    return is_valid_arr


def get_valid_dates(eva, is_valid_arr):
    dates_for_data_ordinal = eva.timeline["dates_for_data_ordinal"]
    # valid dates
    dates = eva.timeline["dates"]
    valid_dates = eva.timeline['valid_dates'] # 设为defaultdict(list)
    for date in dates:
        ordinal = dates_for_data_ordinal[date]
        is_valid = is_valid_arr[ordinal - 1] and is_valid_arr[ordinal] and is_valid_arr[ordinal + 1]
        if is_valid:
            valid_dates.append(date)
    # valid dates for data
    dates_for_data = eva.timeline["dates_for_data"]
    valid_dates_for_data = eva.timeline['valid_dates_for_data'] # 设为defaultdict(list)
    invalid_dates_for_data = eva.timeline['invalid_dates_for_data']
    for date in dates_for_data:
        ordinal = dates_for_data_ordinal[date]
        is_valid = is_valid_arr[ordinal - 1] and is_valid_arr[ordinal]
        if is_valid:
            valid_dates_for_data.append(date)
        else:
            invalid_dates_for_data.append(date)
        
        
def check_if_enough_valid_points(data, col_name, valid_prop_lmt):
    nan_prop = data[col_name].isna().mean()
    if 1 - nan_prop < valid_prop_lmt:
        return False
    return True


# %% calc twap return
def calc_twap_and_resample(eva):
    calc_twap_price(eva)
    resample(eva)
    
    
def calc_twap_price(eva):
    ''' load params '''
    param = eva.eva_params
    twap_wd = param.get('twap_wd')
    twap_wd_adj_by_freq = int(twap_wd * MIN_SEC / DATA_FREQ) if twap_wd is not None else 1
    ''' load dtypes '''
    dtypes = eva.dtypes
    twap_price_dtype = dtypes['twap_price']
    tick_dtype = dtypes['tick']
    ''' load data or container '''
    tick_data = eva.dataset['tick']
    twap_price_data = eva.dataset['twap_price']
    ''' load dates '''
    valid_dates_for_data = eva.timeline['valid_dates_for_data']
    dates_for_data = eva.timeline["dates_for_data"]
    dates_for_data_ordinal = eva.timeline["dates_for_data_ordinal"]
    ''' loop '''
    len_of_dates = len(dates_for_data) 
    # for date in tqdm(valid_dates_for_data, desc='calc_twap_price'):
    for date in valid_dates_for_data:
        ordinal = dates_for_data_ordinal[date]
        tick_today = tick_data[date]
        today_size = tick_today.size
        tick_fut = np.hstack([(tick_data[dates_for_data[i_d]]
                              if i_d < len_of_dates
                              else np.zeros(0, dtype=tick_dtype))
                              for i_d in range(ordinal, ordinal+2)])
        twap_price = rolling_mean_fut(tick_fut['midprice'], today_size, twap_wd_adj_by_freq) 
        twap_today = np.empty(today_size, dtype=twap_price_dtype)
        twap_today['twap_price'] = twap_price
        twap_price_data[date] = twap_today
        

from timeutils import get_eq_spaced_intraday_time_series
from datautils import downsampling
def resample(eva):
    ''' load params '''
    param = eva.eva_params
    target_sample_freq = param['target_sample_freq']
    ''' load dtypes '''
    dtypes = eva.dtypes
    t_dtype = dtypes['t']
    twap_price_dtype = dtypes['twap_price']
    ''' load data or container '''
    t_data = eva.dataset['t']
    twap_price_data = eva.dataset['twap_price']
    ''' load dates '''
    valid_dates = eva.timeline['valid_dates']
    ''' loop '''
    # for date in tqdm(valid_dates, desc='resample'):
    default_v = 0
    for date in valid_dates:
        date_in_dt = datetime.strptime(date, "%Y-%m-%d")
        tgt_timeline = np.array(get_eq_spaced_intraday_time_series(date_in_dt, target_sample_freq, mode='l')).astype('i8') #{'minutes': 15}
        org_timeline = t_data[date]['timestamp'].view('i8')
        org_value_arr = twap_price_data[date]['twap_price']
        data_resampled = downsampling(org_timeline, tgt_timeline, org_value_arr, 1,
                                      default_v)
        len_resampled = len(data_resampled)
        t_today = np.empty(len_resampled, dtype=t_dtype)
        twap_price_today = np.empty(len_resampled, dtype=twap_price_dtype)
        t_today['timestamp'] = tgt_timeline
        twap_price_today['twap_price'] = data_resampled
        t_data[date] = t_today
        twap_price_data[date] = twap_price_today
        
        default_v = org_value_arr[-1]
        

# %% oneday calc
def merge_to_df(eva):
    ''' load info '''
    symbol = eva.symbol
    ''' load data or container '''
    t_data = eva.dataset['t']
    twap_price_data = eva.dataset['twap_price']
    ''' load dates '''
    valid_dates = eva.timeline['valid_dates']
    ''' load dir '''
    output_dir = eva.output_dir
    # factor_path = output_dir / f'{factor_name}.parquet'
    ''' loop '''
    data_list = []
    for date in valid_dates:
        data_today = pd.DataFrame({symbol: twap_price_data[date]['twap_price']}, index=t_data[date]['timestamp'])
        data_list.append(data_today)
    data_one_factor_one_symbol = pd.concat(data_list)
    return data_one_factor_one_symbol
        
    
# %% main
def calc_twap_price_one_symbol(process_name, factor_name,
                               timeline_params, 
                               eva_params, dtypes, data_dir,
                               result_dir, symbol=''):
    # try:
    eva = TwapProcessor(eval_name, factor_name, symbol, 
                        timeline_params=timeline_params, 
                        eva_params=eva_params, dtypes=dtypes,
                        data_dir=data_dir,
                        result_dir=result_dir)
    load_data_and_check(eva)
    calc_twap_and_resample(eva)
    data_one_factor_one_symbol = merge_to_df(eva)
    # except:
    #     traceback.print_exc()
    #     return None
    return data_one_factor_one_symbol


import queue
from functools import partial
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
def process_multi_symbol(process_name='', factor_name='', symbol_list='',
                         timeline_params=None, 
                         eva_params=None, dtypes=None,
                         data_dir=None, result_dir=None, max_workers=None):
    # manager = Manager()
    # lock = manager.Lock()
    process_func = partial(calc_twap_price_one_symbol, process_name, factor_name, timeline_params,
                           eva_params, dtypes, data_dir, result_dir) #, lock
    all_symbol_res = []
    if max_workers is None:
        for symbol in tqdm(symbol_list, desc=process_name):
            merged = process_func(symbol=symbol)
            all_symbol_res.append(merged)
    else:
        res_queue = queue.Queue()
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            all_tasks = [executor.submit(process_func, symbol=symbol)
                         for symbol in symbol_list]
            for task in tqdm(as_completed(all_tasks), total=len(all_tasks), desc=process_name):
                res = task.result()
                if res is not None:
                    res_queue.put(res)
        while not res_queue.empty():
            merged = res_queue.get()
            all_symbol_res.append(merged)
            
    data = pd.concat(all_symbol_res, axis=1, join='outer')

    output_dir = result_dir
    factor_path = output_dir / f'{process_name}.parquet'
    data.to_parquet(factor_path)


# %%
if __name__=='__main__':
    pass
    
    
        