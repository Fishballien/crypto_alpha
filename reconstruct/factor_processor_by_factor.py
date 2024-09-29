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


from timeutils import MIN_SEC, DATA_FREQ, DAY_SEC, timestr_to_seconds, timedelta_to_seconds
from algo import rolling_mean_fut, ma, mstd, get_moving_min, get_moving_max, mskew
from dirs import PARAM_DIR, DATA_DIR, PROCESSED_DATA_DIR
from feval import FEvaluation
from speedutils import timeit


# %% 
eval_name = 'fma1_sp15_rp10'
# factor_name_list = ['ACTmidpos', 'mpc']
factor_name_list = ['ACTmidpos', 'Svwap_POSHL_small']
# factor_name = 'mpc'
symbol = 'btcusdt'


# %%
timeline_params = {
    'start_date': datetime(2023, 10, 16),
    'end_date': datetime(2023, 10, 16),
    'data_start_date': datetime(2023, 10, 15),
    'data_end_date': datetime(2023, 10, 17),
    }


# eva_params = toml.load(PARAM_DIR / f'{eval_name}.toml')


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


# %% load data
def load_data_and_check(eva):
    # eva.a = 1
    is_valid_arr = load_data(eva)
    # eva.a = 2
    get_valid_dates(eva, is_valid_arr)
    # eva.a = 3
            

from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
def load_data(eva):
    ''' load info '''
    symbol = eva.symbol
    factor_name = eva.factor_name
    ''' load params '''
    param = eva.eva_params
    valid_prop_lmt = param['valid_prop_lmt']
    ''' load dtypes '''
    dtypes = eva.dtypes
    t_dtype, factor_dtype = dtypes['t'], dtypes['factor']
    ''' load data or container '''
    t_data = eva.dataset["t"]
    factor_data = eva.dataset["factor"]
    factor = eva.factor
    ''' load dates '''
    dates_for_data = eva.timeline["dates_for_data"]
    ''' init '''
    data_col_list = ['timestamp', factor_name]
    target_col_list = ['timestamp', 'factor']
    dtype_list = [t_dtype, factor_dtype]
    data_list = [t_data, factor_data]
    is_valid_arr = np.zeros(len(dates_for_data), dtype=np.int32)
    ''' loop ''' 
    # for i_d, dt in enumerate(tqdm(dates_for_data, desc='load_data')):
    for i_d, dt in enumerate(dates_for_data):
        data = factor.read_oneday(dt, symbol, columns=data_col_list, method='pandas')
        # print(data)
        # data = factor.read_oneday(dt, symbol, columns=data_col_list, method='pyarrow')
        # if data is not None:
        #     breakpoint()
        # print(data)
        # data = factor.read_oneday(dt, symbol, columns=data_col_list, method='dask')
        # print(data)
        # if data is not None:
        #     raise Exception
        
        # breakpoint()
        is_not_valid = (data is None 
                        or (not check_if_enough_valid_points(data, factor_name, valid_prop_lmt))
                        )
        if is_not_valid:
            t_data[dt] = np.zeros(0, dtype=t_dtype)
            factor_data[dt] = np.zeros(0, dtype=factor_dtype)
            continue
        today_size = len(data)
        for data_col, target_col, dtype, target_data in zip(data_col_list, target_col_list,
                                                            dtype_list, data_list):
            x_data_today = np.empty(today_size, dtype=dtype)
            if data_col == 'timestamp':
                x_data_today[target_col] = data[data_col] / 1e3
            else:
                x_data_today[target_col] = pd.Series(data[data_col]).ffill().bfill()
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

        
# %% feature engineering
def feature_engineering(eva):
    # eva.a = 4
    param = eva.eva_params
    mfunc_name = param['mfunc']
    calc_mfunc = globals()[mfunc_name]
    calc_mfunc(eva)
    # eva.a = 5
    resample(eva)
    # eva.a = 6
    calc_rp(eva)
    # eva.a = 7


# @timeit
def calc_factor_ma(eva):
    ''' load params '''
    param = eva.eva_params
    raw_factor_ma_wd = param['factor_ma_wd']
    factor_ma_wd = int(raw_factor_ma_wd * MIN_SEC / DATA_FREQ)
    ''' load dtypes '''
    dtypes = eva.dtypes
    factor_ma_dtype = dtypes['factor_ma']
    ''' load data or container '''
    factor_data = eva.dataset['factor']
    factor_ma_data = eva.dataset['factor_ma']
    ''' load dates '''
    valid_dates_for_data = eva.timeline['valid_dates_for_data']
    dates_for_data = eva.timeline["dates_for_data"]
    dates_for_data_ordinal = eva.timeline["dates_for_data_ordinal"]
    invalid_dates_for_data = eva.timeline["invalid_dates_for_data"]
    ''' loop '''
    # for date in tqdm(valid_dates_for_data, desc='calc_factor_ma'):
    for date in valid_dates_for_data:
        ordinal = dates_for_data_ordinal[date]
        factor_today = factor_data[date]
        today_size = factor_today.size
        read_data_len = 1
        factor_his = np.hstack([factor_data[dates_for_data[i_d]]
                              for i_d in range(max(ordinal-read_data_len, 0), ordinal+1)])
        factor_ma_today = np.empty(today_size, dtype=factor_ma_dtype)
        factor_ffill = advanced_forward_fill(factor_his['factor'])
        factor_ma_today['factor_ma'] = ma(factor_ffill, factor_ma_wd)[-today_size:]
        factor_ma_data[date] = factor_ma_today
    for date in invalid_dates_for_data:
        factor_ma_data[date] = np.zeros(0, dtype=factor_ma_dtype)
        
        
def calc_factor_mskew(eva):
    ''' load params '''
    param = eva.eva_params
    raw_factor_ma_wd = param['factor_ma_wd']
    factor_ma_wd = raw_factor_ma_wd * MIN_SEC / DATA_FREQ
    ''' load dtypes '''
    dtypes = eva.dtypes
    factor_ma_dtype = dtypes['factor_ma']
    ''' load data or container '''
    factor_data = eva.dataset['factor']
    factor_ma_data = eva.dataset['factor_ma']
    ''' load dates '''
    valid_dates_for_data = eva.timeline['valid_dates_for_data']
    dates_for_data = eva.timeline["dates_for_data"]
    dates_for_data_ordinal = eva.timeline["dates_for_data_ordinal"]
    invalid_dates_for_data = eva.timeline["invalid_dates_for_data"]
    ''' loop '''
    # for date in tqdm(valid_dates_for_data, desc='calc_factor_ma'):
    for date in valid_dates_for_data:
        ordinal = dates_for_data_ordinal[date]
        factor_today = factor_data[date]
        today_size = factor_today.size
        read_data_len = 1
        factor_his = np.hstack([factor_data[dates_for_data[i_d]]
                              for i_d in range(max(ordinal-read_data_len, 0), ordinal+1)])
        factor_ma_today = np.empty(today_size, dtype=factor_ma_dtype)
        factor_ffill = advanced_forward_fill(factor_his['factor'])
        factor_ma_today['factor_ma'] = mskew(factor_ffill, factor_ma_wd)[-today_size:]
        factor_ma_data[date] = factor_ma_today
    for date in invalid_dates_for_data:
        factor_ma_data[date] = np.zeros(0, dtype=factor_ma_dtype)
        
        
def calc_factor_mkurt(eva):
    ''' load params '''
    param = eva.eva_params
    raw_factor_ma_wd = param['factor_ma_wd']
    factor_ma_wd = int(raw_factor_ma_wd * MIN_SEC / DATA_FREQ)
    ''' load dtypes '''
    dtypes = eva.dtypes
    factor_ma_dtype = dtypes['factor_ma']
    ''' load data or container '''
    factor_data = eva.dataset['factor']
    factor_ma_data = eva.dataset['factor_ma']
    ''' load dates '''
    valid_dates_for_data = eva.timeline['valid_dates_for_data']
    dates_for_data = eva.timeline["dates_for_data"]
    dates_for_data_ordinal = eva.timeline["dates_for_data_ordinal"]
    invalid_dates_for_data = eva.timeline["invalid_dates_for_data"]
    ''' loop '''
    # for date in tqdm(valid_dates_for_data, desc='calc_factor_ma'):
    for date in valid_dates_for_data:
        ordinal = dates_for_data_ordinal[date]
        factor_today = factor_data[date]
        today_size = factor_today.size
        read_data_len = 1
        factor_his = np.hstack([factor_data[dates_for_data[i_d]]
                              for i_d in range(max(ordinal-read_data_len, 0), ordinal+1)])
        factor_ma_today = np.empty(today_size, dtype=factor_ma_dtype)
        factor_ffill = advanced_forward_fill(factor_his['factor'])
        factor_ma_today['factor_ma'] = pd.Series(factor_ffill).rolling(factor_ma_wd).kurt()[-today_size:]
        factor_ma_data[date] = factor_ma_today
    for date in invalid_dates_for_data:
        factor_ma_data[date] = np.zeros(0, dtype=factor_ma_dtype)
        
        
def calc_factor_mzscr(eva):
    ''' load params '''
    param = eva.eva_params
    raw_factor_ma_wd = param['factor_ma_wd']
    factor_ma_wd = int(raw_factor_ma_wd * MIN_SEC / DATA_FREQ)
    ''' load dtypes '''
    dtypes = eva.dtypes
    factor_ma_dtype = dtypes['factor_ma']
    ''' load data or container '''
    factor_data = eva.dataset['factor']
    factor_ma_data = eva.dataset['factor_ma']
    ''' load dates '''
    valid_dates_for_data = eva.timeline['valid_dates_for_data']
    dates_for_data = eva.timeline["dates_for_data"]
    dates_for_data_ordinal = eva.timeline["dates_for_data_ordinal"]
    invalid_dates_for_data = eva.timeline["invalid_dates_for_data"]
    ''' loop '''
    # for date in tqdm(valid_dates_for_data, desc='calc_factor_ma'):
    for date in valid_dates_for_data:
        ordinal = dates_for_data_ordinal[date]
        factor_today = factor_data[date]
        today_size = factor_today.size
        read_data_len = 1
        factor_his = np.hstack([factor_data[dates_for_data[i_d]]
                              for i_d in range(max(ordinal-read_data_len, 0), ordinal+1)])
        factor_ma_today = np.empty(today_size, dtype=factor_ma_dtype)
        factor_ffill = advanced_forward_fill(factor_his['factor'])
        factor_ma_today['factor_ma'] = (ma(factor_ffill, factor_ma_wd)[-today_size:] 
                                        / mstd(factor_ffill, factor_ma_wd)[-today_size:])
        factor_ma_data[date] = factor_ma_today
    for date in invalid_dates_for_data:
        factor_ma_data[date] = np.zeros(0, dtype=factor_ma_dtype)
        
        
def calc_factor_diff_std(eva):
    ''' load params '''
    param = eva.eva_params
    raw_factor_ma_wd = param['factor_ma_wd']
    diff_wd = int(param['diff_wd'])
    factor_ma_wd = int(raw_factor_ma_wd * MIN_SEC / DATA_FREQ)
    ''' load dtypes '''
    dtypes = eva.dtypes
    factor_ma_dtype = dtypes['factor_ma']
    ''' load data or container '''
    factor_data = eva.dataset['factor']
    factor_ma_data = eva.dataset['factor_ma']
    ''' load dates '''
    valid_dates_for_data = eva.timeline['valid_dates_for_data']
    dates_for_data = eva.timeline["dates_for_data"]
    dates_for_data_ordinal = eva.timeline["dates_for_data_ordinal"]
    invalid_dates_for_data = eva.timeline["invalid_dates_for_data"]
    ''' loop '''
    # for date in tqdm(valid_dates_for_data, desc='calc_factor_ma'):
    for date in valid_dates_for_data:
        ordinal = dates_for_data_ordinal[date]
        factor_today = factor_data[date]
        today_size = factor_today.size
        read_data_len = 1
        factor_his = np.hstack([factor_data[dates_for_data[i_d]]
                              for i_d in range(max(ordinal-read_data_len, 0), ordinal+1)])
        factor_ma_today = np.zeros(today_size, dtype=factor_ma_dtype)
        factor_ffill = advanced_forward_fill(factor_his['factor'])
        factor_diff = factor_ffill[diff_wd:] - factor_ffill[:-diff_wd]
        std_arr = mstd(factor_diff, factor_ma_wd)[-today_size:]
        len_of_std_arr = len(std_arr)
        factor_ma_today['factor_ma'][-len_of_std_arr:] = std_arr
        factor_ma_data[date] = factor_ma_today
    for date in invalid_dates_for_data:
        factor_ma_data[date] = np.zeros(0, dtype=factor_ma_dtype)
        

from timeutils import get_eq_spaced_intraday_time_series
from datautils import downsampling
# @timeit
def resample(eva):
    ''' load params '''
    param = eva.eva_params
    target_sample_freq = param['target_sample_freq']
    ''' load dtypes '''
    dtypes = eva.dtypes
    t_dtype = dtypes['t']
    factor_ma_dtype = dtypes['factor_ma']
    ''' load data or container '''
    t_data = eva.dataset['t']
    factor_ma_data = eva.dataset['factor_ma']
    ''' load dates '''
    valid_dates = eva.timeline['valid_dates']
    ''' loop '''
    # for date in tqdm(valid_dates, desc='resample'):
    default_v = 0
    for date in valid_dates:
        date_in_dt = datetime.strptime(date, "%Y-%m-%d")
        tgt_timeline = np.array(get_eq_spaced_intraday_time_series(date_in_dt, target_sample_freq)).astype('i8') #{'minutes': 15}
        org_timeline = t_data[date]['timestamp'].view('i8')
        org_value_arr = factor_ma_data[date]['factor_ma']
        data_resampled = downsampling(org_timeline, tgt_timeline, org_value_arr, 
                                      default_v)
        len_resampled = len(data_resampled)
        t_today = np.empty(len_resampled, dtype=t_dtype)
        factor_ma_today = np.empty(len_resampled, dtype=factor_ma_dtype)
        t_today['timestamp'] = tgt_timeline
        factor_ma_today['factor_ma'] = data_resampled
        t_data[date] = t_today
        factor_ma_data[date] = factor_ma_today
        
        default_v = org_value_arr[-1]


# @timeit
def calc_rp(eva):
    ''' load params '''
    param = eva.eva_params
    raw_rp_wd = param.get('rp_wd') # d
    if raw_rp_wd is None:
        return
    target_sample_freq = param['target_sample_freq']
    resampled_data_freq = timedelta_to_seconds(target_sample_freq)
    rp_wd = raw_rp_wd * DAY_SEC / resampled_data_freq
    norm_method_name = param.get('norm_method_name', 'calc_rp_oneday')
    norm_func = globals()[norm_method_name]
    ''' load dtypes '''
    dtypes = eva.dtypes
    factor_rp_dtype = dtypes['factor_rp']
    ''' load data or container '''
    factor_ma_data = eva.dataset['factor_ma']
    factor_rp_data = eva.dataset['factor_rp']
    ''' load dates '''
    valid_dates = eva.timeline['valid_dates']
    dates_for_data = eva.timeline["dates_for_data"]
    dates_for_data_ordinal = eva.timeline["dates_for_data_ordinal"]
    # for date in tqdm(valid_dates, desc='calc_rp'):
    for date in valid_dates:
        ordinal = dates_for_data_ordinal[date]
        factor_ma_today = factor_ma_data[date]
        today_size = factor_ma_today.size
        read_data_len = raw_rp_wd + 1
        factor_ma_his = np.hstack([factor_ma_data[dates_for_data[i_d]]
                                   for i_d in range(max(ordinal-read_data_len, 0), ordinal+1)])
        rp_today = np.empty(today_size, dtype=factor_rp_dtype)
        rp_today['factor_rp'] = norm_func(
            factor_ma_his['factor_ma'], factor_ma_today['factor_ma'], 
            today_size, rp_wd)
        factor_rp_data[date] = rp_today
        
    
def calc_rp_oneday(target_his, target_today, today_size, wd):
    min_his = get_moving_min(target_his, today_size, wd)
    max_his = get_moving_max(target_his, today_size, wd)
    # res = np.where(max_his - min_his != 0, (target_today - min_his) / (max_his - min_his), 0.5)
    res = np.divide(target_today - min_his, max_his - min_his, 
                    out=np.full(len(target_today), 0.5), where=(max_his - min_his) != 0)
    return res


from algo import advanced_forward_fill
def calc_zscore_oneday(target_his, target_today, today_size, wd):
    target_today = advanced_forward_fill(target_today)
    target_his = advanced_forward_fill(target_his)
    ma_ = ma(target_his, wd)[-today_size:]
    mstd_ = mstd(target_his, wd)[-today_size:]
    z_score = np.divide(target_today - ma_, mstd_,
                        out=np.full(len(target_today), 0.0), where=mstd_ != 0)
    return z_score
        

# %% after calc
# @timeit
def merge_to_df(eva):
    ''' load params '''
    param = eva.eva_params
    raw_rp_wd = param.get('rp_wd') # d
    ''' load info '''
    symbol = eva.symbol
    factor_name = eva.factor_name
    ''' load data or container '''
    t_data = eva.dataset['t']
    factor_ma_data = eva.dataset['factor_ma']
    factor_rp_data = eva.dataset['factor_rp']
    target_data = factor_ma_data if raw_rp_wd is None else factor_rp_data
    target_name = 'factor_ma' if raw_rp_wd is None else 'factor_rp'
    ''' load dates '''
    valid_dates = eva.timeline['valid_dates']
    ''' load dir '''
    output_dir = eva.output_dir
    # factor_path = output_dir / f'{factor_name}.parquet'
    ''' loop '''
    data_list = []
    # for date in tqdm(valid_dates, desc='merge_to_df'):
    for date in valid_dates:
        data_today = pd.DataFrame({symbol: target_data[date][target_name]}, index=t_data[date]['timestamp'])
        data_list.append(data_today)
    data_one_factor_one_symbol = pd.concat(data_list)
    return data_one_factor_one_symbol
# =============================================================================
#     existing_data = pd.read_parquet(factor_path) if os.path.exists(factor_path) else pd.DataFrame()
#     data = pd.concat([existing_data, data_one_factor_one_symbol], axis=1, join='outer')
#     data.to_parquet(factor_path)
# =============================================================================
            

# %% main
def all_process(eval_name, factor_name, timeline_params,
                eva_params, dtypes, data_dir, result_dir, symbol):
    try:
        eva = FEvaluation(eval_name, factor_name, symbol, 
                          timeline_params=timeline_params, 
                          eva_params=eva_params, dtypes=dtypes,
                          data_dir=data_dir,
                          result_dir=result_dir)
        load_data_and_check(eva)
        feature_engineering(eva)
        data_one_factor_one_symbol = merge_to_df(eva)
    except:
        traceback.print_exc()
        return None
    return data_one_factor_one_symbol
    

import queue
from functools import partial
def process_one_factor_multi_symbol(process_name='', factor_name='', symbol_list='',
                                    timeline_params=timeline_params, 
                                    eva_params={}, dtypes=dtypes,
                                    data_dir=None, result_dir=None, max_workers=None):
    # manager = Manager()
    # lock = manager.Lock()
    process_func = partial(all_process, process_name, factor_name, timeline_params,
                           eva_params, dtypes, data_dir, result_dir) #, lock
    all_symbol_res = []
    if max_workers is None:
        for symbol in symbol_list:
        # for symbol in tqdm(symbol_list, desc=factor_name):
            merged = process_func(symbol=symbol)
            all_symbol_res.append(merged)
    else:
        res_queue = queue.Queue()
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            all_tasks = [executor.submit(process_func, symbol=symbol)
                         for symbol in symbol_list]
        for task in tqdm(as_completed(all_tasks), total=len(all_tasks), desc=factor_name):
            res = task.result()
            if res is not None:
                res_queue.put(res)
        while not res_queue.empty():
            merged = res_queue.get()
            all_symbol_res.append(merged)
    
    data = pd.concat(all_symbol_res, axis=1, join='outer')[symbol_list]

    output_dir = result_dir / process_name
    output_dir.mkdir(parents=True, exist_ok=True)
    factor_path = output_dir / f'{factor_name}.parquet'
    data.to_parquet(factor_path)


# %%
if __name__=='__main__':
    pass
    
    
        