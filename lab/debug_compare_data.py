# -*- coding: utf-8 -*-
"""
Created on Wed May  8 22:23:08 2024

@author: Xintang Zheng

"""
# %% imports
import pandas as pd
from pathlib import Path


from dirs import DATA_DIR


# %% 对比同sp新旧两种数据
# =============================================================================
# data_dir = Path(r'D:\mnt\Data\Crypto\ProcessedData\15m_cross_sectional')
# old_dir = data_dir / 'fma15_sp15'
# new_dir = data_dir / 'ma15_sp15'
# factor_name = 'bidask_amount_ratio'
# # factor_name = 'SOIR1'
# symbol = 'solusdt'
# 
# old_data = pd.read_parquet(old_dir / f'{factor_name}.parquet')
# new_data = pd.read_parquet(new_dir / f'{factor_name}.parquet')
# 
# old_data[symbol].plot()
# new_data[symbol].plot()
# (new_data[symbol]-old_data[symbol]).plot()
# =============================================================================


# 发现新数据抖动远大于旧数据


# %% 检查新数据reconstruct部分是否符合预期
# =============================================================================
# sample_dir = Path(r'D:\crypto\multi_factor\factor_test_by_alpha\sample_data')
# 
# raw_data = pd.read_parquet(sample_dir / 'apeusdt.parquet')
# re_data = pd.read_parquet(sample_dir / 'resampled.parquet')
# raw_data['t'] = pd.to_datetime(raw_data['timestamp'] / 1000, unit='ms')
# raw_data.set_index('t', inplace=True)
# re_data['t'] = pd.to_datetime(re_data['timestamp'] / 1000, unit='ms')
# re_data.set_index('t', inplace=True)
# =============================================================================

# 符合预期


# %% 查看feature的原始数据
# =============================================================================
# bnb_path = DATA_DIR / '2021-01-15' / 'bnbusdt.parquet'
# bnb_data = pd.read_parquet(bnb_path)
# 
# bnb_data['t'] = pd.to_datetime(bnb_data['timestamp'] / 1000, unit='ms')
# bnb_data.set_index('t', inplace=True)
# 
# bnb_feature = bnb_data[factor_name]
# =============================================================================


# 新版ma计算可以和原始数据对上


# %% 检查原始数据是否有变化
bnb_dir = DATA_DIR / '2023-10-15'
old_data = pd.read_parquet(bnb_dir / 'bnbusdt_old.parquet')
new_data = pd.read_parquet(bnb_dir / 'bnbusdt_0412.parquet')
# factor_name = 'bidask_amount_ratio'
# factor_name = 'SOIR1'
# factor_name = 'bid_amount_sum'
# factor_name = 'ask_amount_sum'
factor_name = 'Svwap_POSOC'

old_data[factor_name].plot()
new_data[factor_name].plot()
(new_data[factor_name]-old_data[factor_name]).plot()

# c = old_data == new_data
# print(c.all().sum())

# unequal_positions = old_data != new_data

# # 输出不一样的列
# unequal_columns = old_data.columns[unequal_positions.any()]
# print("不一样的列：", unequal_columns)


# %% 比较twap
# =============================================================================
# data_dir = Path(r'D:\mnt\Data\Crypto\ProcessedData\twap_cross_sectional')
# old_name = 'twd15'
# new_name = 'twd15_sp240'
# symbol = 'bnbusdt'
# 
# raw_old_data = pd.read_parquet(data_dir / f'{old_name}.parquet')
# old_data = raw_old_data.resample('240T').first()
# new_data = pd.read_parquet(data_dir / f'{new_name}.parquet')
# 
# old_data[symbol].plot()
# new_data[symbol].plot()
# (new_data[symbol]-old_data[symbol]).plot()
# 
# comp = new_data[symbol]-old_data[symbol]
# =============================================================================


# %% 比较模型多空曲线
# =============================================================================
# old_dir = Path(r'D:\crypto\multi_factor\factor_test_by_alpha\results\ridge_v5_ma15_only\predict\240T\data')
# new_dir = Path(r'D:\crypto\multi_factor\factor_test_by_alpha\results\ridge_v5_ma15_only_2\predict\data')
# old_data = pd.read_parquet(old_dir / 'gp_predict.parquet')
# new_data = pd.read_parquet(new_dir / 'gp_predict.parquet')
# 
# target = 'long_short_0'
# 
# old_data[target].cumsum().plot()
# new_data[target].cumsum().plot()
# (old_data[target].cumsum()-new_data[target].cumsum()).plot()
# =============================================================================


# %% 对比poly前后
# =============================================================================
# data_dir = Path(r'D:\mnt\Data\Crypto\ProcessedData\15m_cross_sectional')
# old_dir = data_dir / 'ma15_sp240'
# new_dir = data_dir / 'ma15_sp240_poly_1y_3_sp240'
# factor_name = 'bidask_amount_ratio'
# # factor_name = 'SOIR1'
# symbol = 'bnbusdt'
# 
# old_data = pd.read_parquet(old_dir / f'{factor_name}.parquet')
# new_data = pd.read_parquet(new_dir / f'{factor_name}.parquet')
# 
# old_data[symbol].plot()
# new_data[symbol].plot()
# (new_data[symbol]-old_data[symbol]).plot()
# =============================================================================
