# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 13:16:33 2024

@author: Xintang Zheng

星星: ★ ☆ ✪ ✩ 🌟 ⭐ ✨ 🌠 💫 ⭐️
勾勾叉叉: ✓ ✔ ✕ ✖ ✅ ❎
报警啦: ⚠ ⓘ ℹ ☣
箭头: ➔ ➜ ➙ ➤ ➥ ↩ ↪
emoji: 🔔 ⏳ ⏰ 🔒 🔓 🛑 🚫 ❗ ❓ ❌ ⭕ 🚀 🔥 💧 💡 🎵 🎶 🧭 📅 🤔 🧮 🔢 📊 📈 📉 🧠 📝

"""
import os
import sys
from datetime import datetime
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import psutil
from pathlib import Path


# %% add sys path
file_path = Path(__file__).resolve()
file_dir = file_path.parents[0]
project_dir = file_path.parents[1]
sys.path.append(str(project_dir))


# %%
from utils.datautils import align_index_with_main


# %%
def process_symbol(symbol, date_folders, input_dir, intermediate_dir, freq='15T'):
    symbol_data = []
    
    for date_folder in date_folders:
        date_folder_path = os.path.join(input_dir, date_folder)
        file_path = os.path.join(date_folder_path, f"{symbol}.parquet")
        if os.path.exists(file_path):
            df = pd.read_parquet(file_path)
            df['timestamp'] = pd.to_datetime(df['timestamp'] / 1000, unit='ms')
            symbol_data.append(df)
    
    if symbol_data:
        combined_df = pd.concat(symbol_data).sort_values(by='timestamp')
        
        # 新增一列 't' 并将其设为索引
        combined_df.set_index('timestamp', inplace=True)
        
        # 获取时间范围并生成从00:00开始的时间索引
        start_time = combined_df.index.min().floor('D')
        end_time = combined_df.index.max()
        time_range = pd.date_range(start=start_time, end=end_time, freq='3s')
        
        # 按指定频率重采样并计算聚合值
        combined_df = combined_df.reindex(time_range).iloc[1:]
        resampled_df = combined_df.resample(freq, label='right').mean()
        
        # 保存中间结果
        symbol_intermediate_path = os.path.join(intermediate_dir, f"{symbol}.parquet")
        resampled_df.to_parquet(symbol_intermediate_path)

def read_and_save_intermediate(input_dir, intermediate_dir, freq='15T', max_workers=5):
    if not os.path.exists(intermediate_dir):
        os.makedirs(intermediate_dir)
    
    date_folders = [d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))]
    symbols = set()
    
    for date_folder in date_folders:
        date_folder_path = os.path.join(input_dir, date_folder)
        for file in os.listdir(date_folder_path):
            if file.endswith('.parquet'):
                symbol = file.split('.')[0]
                symbols.add(symbol)
    
    if max_workers == 1 or max_workers is None:
        # 单线程处理
        for symbol in tqdm(symbols, desc="Processing symbols"):
            process_symbol(symbol, date_folders, input_dir, intermediate_dir, freq)
    else:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(process_symbol, symbol, date_folders, input_dir, intermediate_dir, freq) for symbol in symbols]
            
            for future in tqdm(as_completed(futures), total=len(futures), desc="Processing symbols"):
                future.result()

def process_and_save_factor(symbol, intermediate_dir):
    file_path = os.path.join(intermediate_dir, f"{symbol}.parquet")
    df = pd.read_parquet(file_path)
    factor_data = {}

    # 收集因子数据，忽略timestamp相关列
    for factor in df.columns:
        if factor != 'timestamp':
            if factor not in factor_data:
                factor_data[factor] = pd.DataFrame(index=df.index)
            factor_data[factor][symbol] = df[factor]

    return factor_data

def resample_and_save_mean(intermediate_dir, output_dir, max_workers=4):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    symbol_files = [f for f in os.listdir(intermediate_dir) if f.endswith('.parquet')]
    symbols = [f.split('.')[0] for f in symbol_files]
    
    factor_data = {}
    
    if max_workers == 1 or max_workers is None:
        # 单线程处理
        for symbol in symbols:
            print(symbol, 'start', datetime.now())
            symbol_factor_data = process_and_save_factor(symbol, intermediate_dir)
            print(symbol, 'factor get', datetime.now())
            for factor, df in symbol_factor_data.items():
                if factor not in factor_data:
                    factor_data[factor] = [df]
                else:
                    factor_data[factor].append(df)
            print(symbol, 'done', datetime.now())
            
            # 内存监控
            process = psutil.Process(os.getpid())
            print(f"Memory usage after processing {symbol}: {process.memory_info().rss / 1024 ** 2:.2f} MB")
    else:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(process_and_save_factor, symbol, intermediate_dir) for symbol in symbols]
            
            for future in tqdm(as_completed(futures), total=len(futures), desc="Processing symbols"):
                symbol_factor_data = future.result()
                for factor, df in symbol_factor_data.items():
                    if factor not in factor_data:
                        factor_data[factor] = [df]
                    else:
                        factor_data[factor].append(df)
    
    # 合并并保存因子数据
    for factor, df_list in tqdm(factor_data.items(), desc="Saving factors"):
        combined_df = pd.concat(df_list, axis=1).sort_index()  # 按时间戳排序
        file_path = os.path.join(output_dir, f"{factor}.parquet")
        combined_df.to_parquet(file_path)
    
    # 删除中间文件
    for file in tqdm(symbol_files, desc="Deleting intermediate files"):
        file_path = os.path.join(intermediate_dir, file)
        os.remove(file_path)


# 示例调用
freq = '15min'
input_directory = '/mnt/133_feat_transfer'
intermediate_directory = f'/mnt/Data/Crypto/ProcessedData/sk_intermediate/{freq}'
output_directory = f'/mnt/Data/Crypto/ProcessedData/15m_cross_sectional/sk_0723_{freq}'

# 第一步：读取、重采样并保存中间结果
read_and_save_intermediate(input_directory, intermediate_directory, freq=freq,
                            max_workers=20)

# 第二步：读取中间结果并合并
resample_and_save_mean(intermediate_directory, output_directory,
                       max_workers=40)

