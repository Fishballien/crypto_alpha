# -*- coding: utf-8 -*-
"""
Created on Thu Dec 26 15:53:38 2024

@author: Xintang Zheng

星星: ★ ☆ ✪ ✩ 🌟 ⭐ ✨ 🌠 💫 ⭐️
勾勾叉叉: ✓ ✔ ✕ ✖ ✅ ❎
报警啦: ⚠ ⓘ ℹ ☣
箭头: ➔ ➜ ➙ ➤ ➥ ↩ ↪
emoji: 🔔 ⏳ ⏰ 🔒 🔓 🛑 🚫 ❗ ❓ ❌ ⭕ 🚀 🔥 💧 💡 🎵 🎶 🧭 📅 🤔 🧮 🔢 📊 📈 📉 🧠 📝

"""
# %% imports
import sys
from pathlib import Path
import pandas as pd
from functools import partial
import numpy as np
from datetime import datetime
import pickle


# %% add sys path
file_path = Path(__file__).resolve()
file_dir = file_path.parents[0]
project_dir = file_path.parents[1]
sys.path.append(str(project_dir))


# %%
from utils.dirutils import load_path_config
from utils.datautils import load_all_factors, get_one_factor
from data_processing.feature_engineering import normalization


# %%
def generate_half_hour_timestamps_pandas(date_str):
    """
    生成指定日期内每半小时一个的时间戳列表。

    参数:
    - date_str (str): 日期字符串，格式为 'YYYYMMDD'。

    返回:
    - List[str]: 格式为 'YYYY-MM-DD HH:MM:SS' 的时间戳列表。
    """
    # 将日期字符串转换为 pandas Timestamp
    start = pd.to_datetime(date_str, format='%Y%m%d')
    # 生成半小时间隔的时间范围，覆盖整天（48个时间点）
    timestamps = pd.date_range(start=start, periods=48, freq='30min')
    # 将 Timestamp 转换为字符串格式
    return timestamps.strftime('%Y-%m-%d %H:%M:%S').tolist()


def norm_factor(factor, to_mask, normalization_func):
    factor_mask = factor.isna() | to_mask
    factor_masked = factor.mask(factor_mask)
    factor = normalization_func(factor_masked)
    factor = factor.mask(factor_mask)
    return factor


def convert_timestamp_format(timestamp_str):
    """
    将时间戳从 '%Y-%m-%d %H:%M:%S' 格式转换为 '%Y%m%d%H%M%S' 格式。

    参数:
    - timestamp_str (str): 原始时间戳字符串，格式为 '%Y-%m-%d %H:%M:%S'。

    返回:
    - str: 转换后的时间戳字符串，格式为 '%Y%m%d%H%M%S'。
    """
    try:
        # 解析原始时间戳字符串
        dt = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S')
        # 格式化为新格式
        new_format = dt.strftime('%Y%m%d%H%M%S')
        return new_format
    except ValueError as e:
        print(f"时间格式错误: {e}")
        return None


# %%
cluster_name = 'agg_250127_cgy_zxt_double3m'
period = '230201_250201'
date = '20250220'
twap_name = 'twd30_sp30'
cluster_path = f'/mnt/Data/xintang/crypto/multi_factor/factor_test_by_alpha/results/cluster/{cluster_name}/cluster_info_{period}.csv'
sp = 30
outlier = 30
start_date = '20250219'
end_date = '20250224'
res_dir = Path('/mnt/Data/xintang/crypto/multi_factor/factor_test_by_alpha/results/verify') / cluster_name
res_dir.mkdir(parents=True, exist_ok=True)


# %%
file_path = Path(__file__).resolve()
project_dir = file_path.parents[1]
path_config = load_path_config(project_dir)
twap_dir = Path(path_config['twap_price'])
data_dir = Path(path_config['processed_data'])


# %%
twap_path = twap_dir / f'{twap_name}.parquet'
twap_price = pd.read_parquet(twap_path)
twap_price = twap_price[(twap_price.index >= start_date) & (twap_price.index < end_date)]
to_mask = twap_price.isna()


# %%
cluster_info = pd.read_csv(cluster_path)
cluster_info = cluster_info[cluster_info['factor'].apply(lambda x: 'P0.4' in x)].reset_index(drop=True)
# cluster_info.loc[cluster_info['process_name'] == 'LOB_2024-12-13_valid0.2_R2', 'process_name'] = 'LOB_2024-12-26_f64_R2'
# cluster_info.loc[cluster_info['process_name'] == 'LOB_2024-12-13_valid0.2_R1', 'process_name'] = 'LOB_2024-12-26_f64_R1'


# %%
normalization_func = partial(normalization, outlier_n=outlier)


# %%
get_one_factor_func = partial(get_one_factor, sp=sp, 
                              date_start=start_date, date_end=end_date,
                              ref_order_col=twap_price.columns, ref_index=twap_price.index)
factor_dict = load_all_factors(cluster_info, get_one_factor_func, data_dir, 100)
factor_norm_dict = {fac_idx: norm_factor(factor, to_mask, normalization_func) for fac_idx, factor in factor_dict.items()}


# %%
group = {}
group_norm = {}

for gid, group_info in cluster_info.groupby('group'):
    len_of_group = len(group_info)
    group_factor = None
    for id_, index in enumerate(group_info.index):
        process_name, factor_name, direction = group_info.loc[index, ['process_name', 'factor', 'direction']]
        factor = factor_dict[index]
        if not process_name.startswith('gp'):
            factor_mask = factor.isna() | to_mask
            factor_masked = factor.mask(factor_mask)
            factor = normalization_func(factor_masked)
            if factor is None:
                len_of_group -= 1
                print(process_name, factor_name)
                continue
            factor = factor.mask(factor_mask) # TODO: 改为用参数设置mask
        factor = factor * direction
        if group_factor is None:
            group_factor = factor
        else:
            group_factor += factor
    group_factor = group_factor / len_of_group
    group[gid] = group_factor
    group_factor = normalization_func(group_factor)
    group_norm[gid] = group_factor
    

# %%
timestamps = generate_half_hour_timestamps_pandas(date)

for ts in timestamps:

    factor_df = pd.DataFrame(np.nan, index=twap_price.columns, columns=np.arange(len(cluster_info)))
    factor_norm_df = pd.DataFrame(np.nan, index=twap_price.columns, columns=np.arange(len(cluster_info)))
    group_df = pd.DataFrame(np.nan, index=twap_price.columns, columns=np.arange(len(cluster_info['group'].unique())))
    group_norm_df = pd.DataFrame(np.nan, index=twap_price.columns, columns=np.arange(len(cluster_info['group'].unique())))
    
    
    for fac_idx, factor in factor_dict.items():
        factor_df.loc[:, fac_idx] = factor.loc[ts, :]
    for fac_idx, factor_norm in factor_norm_dict.items():
        factor_norm_df.loc[:, fac_idx] = factor_norm.loc[ts, :]
    # for gid, group_factor in group.items():
    #     group_df.loc[:, gid] = group_factor.loc[ts, :]
    # for gid, group_norm_factor in group_norm.items():
    #     group_norm_df.loc[:, gid] = group_norm_factor.loc[ts, :]
        
    res = {
           'factor_org': factor_df,
           'factor_norm': factor_norm_df,
           # 'group_org': group_df,
           # 'group_norm': group_norm_df,
           }
    with open(res_dir / f"{convert_timestamp_format(ts)}.pkl", 'wb') as f:
        pickle.dump(res, f)