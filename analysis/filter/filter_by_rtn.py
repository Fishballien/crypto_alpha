# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 10:49:03 2025

@author: Xintang Zheng

星星: ★ ☆ ✪ ✩ 🌟 ⭐ ✨ 🌠 💫 ⭐️
勾勾叉叉: ✓ ✔ ✕ ✖ ✅ ❎
报警啦: ⚠ ⓘ ℹ ☣
箭头: ➔ ➜ ➙ ➤ ➥ ↩ ↪
emoji: 🔔 ⏳ ⏰ 🔒 🔓 🛑 🚫 ❗ ❓ ❌ ⭕ 🚀 🔥 💧 💡 🎵 🎶 🧭 📅 🤔 🧮 🔢 📊 📈 📉 🧠 📝

"""
# %%
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# %%
price_path = 'D:/mnt/Data/Crypto/ProcessedData/updated_twap/curr_price_sp30.parquet'
predict_path = 'D:/crypto/multi_factor/factor_test_by_alpha/results/model/merge_agg_241227_cgy_zxt_double3m_15d_73/predict/predict_merge_agg_241227_cgy_zxt_double3m_15d_73.parquet'
save_dir = Path('D:/crypto/multi_factor/factor_test_by_alpha/results/model/merge_agg_241227_cgy_zxt_double3m_15d_73/predict/filtered')


# %%
curr_price = pd.read_parquet(price_path)
prediction = pd.read_parquet(predict_path)


# %%
def calculate_mask_state(col_data, period):
    mask = np.zeros_like(col_data, dtype=bool)  # 初始化为 False
    in_mask = False  # 用于标记是否进入mask状态
    consecutive_below_threshold = 0  # 连续小于阈值的计数器

    for i in range(len(col_data)):
        if col_data.iloc[i]:  # 如果大于阈值，进入mask状态
            in_mask = True
            consecutive_below_threshold = 0  # 重置小于阈值的计数器
        else:
            if in_mask:  # 如果已经是mask状态
                consecutive_below_threshold += 1
                if consecutive_below_threshold >= period:
                    in_mask = False  # 如果连续小于阈值的期数达到period，则解除mask
                    consecutive_below_threshold = 0  # 重置计数器

        mask[i] = in_mask  # 更新每个时刻的mask状态

    return mask


# def get_top_k_mask(z_scores, k):
#     # Initialize mask with False values
#     top_k_mask = pd.DataFrame(False, index=z_scores.index, columns=z_scores.columns)
    
#     # For each row (timestamp), get the indices of the top K values
#     for timestamp in z_scores.index:
#         row = z_scores.loc[timestamp]
        
#         # Skip if all values are NaN
#         if row.isna().all():
#             continue
        
#         # Get indices of top K values, ignoring NaNs
#         k_to_use = min(k, (~row.isna()).sum())  # Use at most k, limited by non-NaN values
#         top_indices = row.nlargest(k_to_use).index
        
#         # Set those indices to True for this timestamp
#         top_k_mask.loc[timestamp, top_indices] = True
    
#     return top_k_mask


def get_top_k_mask(z_scores, k):
    # Function to apply to each row
    def get_top_k_in_row(row):
        # Skip if all values are NaN
        if row.isna().all():
            return pd.Series(False, index=row.index)
        
        # Create a mask series initialized with False
        mask = pd.Series(False, index=row.index)
        
        # Get indices of top K values, ignoring NaNs
        k_to_use = min(k, (~row.isna()).sum())  # Use at most k, limited by non-NaN values
        if k_to_use > 0:
            top_indices = row.nlargest(k_to_use).index
            mask[top_indices] = True
            
        return mask
    
    # Apply the function to each row and return the resulting DataFrame
    return z_scores.apply(get_top_k_in_row, axis=1)


# %% pr-zscore: 使用4h价格波动的历史3个月zscore来确定
n = 2
m = 2*24*30*3

# 计算滚动窗口内的最大值和最小值
rolling_max = curr_price.rolling(window=n, min_periods=1).max()
rolling_min = curr_price.rolling(window=n, min_periods=1).min()

# 计算滚动价格变化范围（最大值 - 最小值）
price_range = rolling_max - rolling_min

# 计算滚动窗口内的均值和标准差
rolling_mean = price_range.rolling(window=m, min_periods=1).mean()
rolling_std = price_range.rolling(window=m, min_periods=1).std()

# 计算Z-Score
z_scores = (price_range - rolling_mean) / rolling_std

# for col in z_scores.columns:
#     plt.figure(figsize=(10, 6))  # 设置图像的大小
#     z_scores[col].plot(title=f'Z-Score of {col}', color='blue')  # 绘制每个币种的Z-Score
#     plt.xlabel('Time')  # X轴标签
#     plt.ylabel('Z-Score')  # Y轴标签
#     plt.grid(True)  # 加网格
#     plt.show()  # 显示图像

# 计算每行的 95% 分位数
percentile_95 = z_scores.quantile(0.95, axis=0)
# percentile_95.mean()
# Out[11]: 1.9573887393813636

percentile_975 = z_scores.quantile(0.975, axis=0)
# percentile_975.mean()
# Out[13]: 2.8032975686583685

percentile_99 = z_scores.quantile(0.99, axis=0)
# percentile_99.mean()
# Out[23]: 4.173447496470324

percentile_999 = z_scores.quantile(0.999, axis=0)
# percentile_999.mean()
# Out[23]: 4.173447496470324

# =============================================================================
# przsc_dir = save_dir / 'pr_zscore'
# przsc_dir.mkdir(parents=True, exist_ok=True)
# 
# thres_list = [1.5, 2, 2.25, 2.5, 2.75, 3, 3.25, 3.5, 3.75, 4, 4.25, 4.5, 4.75, 5, 5.25, 5.5, 5.75, 6]
# for thres in thres_list:
#     pr_zscore_mask = z_scores > thres
#     new_fac = prediction.mask(pr_zscore_mask)
#     
#     # # Calculate the count ratio between new_fac and prediction
#     # count_ratio = new_fac.count(axis=1) / prediction.count(axis=1)
#     
#     # # Resample by day and calculate the mean
#     # daily_avg_ratio = count_ratio.resample('D').mean()
#     
#     # # Generate a dynamic title based on the current `thres` and `period`
#     # new_fac_name = f'predict-pr_zscore_{thres}'
#     # title = f"Daily Average of Count Ratio (masked / prediction)\nParameters: {new_fac_name}"
#     
#     # # Plot the daily average ratio with the dynamic title
#     # daily_avg_ratio.plot(figsize=(10, 6), title=title)
#     # plt.xlabel("Date")
#     # plt.ylabel("Count Ratio")
#     # plt.grid(True)
#     # plt.show()
#     
#     new_fac_name = f'predict-pr_zscore_{thres}'
#     new_fac.to_parquet(przsc_dir / f'{new_fac_name}.parquet')
# =============================================================================

przsc_dir = save_dir / 'pr_zscore_smthout'
przsc_dir.mkdir(parents=True, exist_ok=True)

thres_list = [15] # 7.5, 10, 12.5, 
periods = [8] # 8, 48, 144, 480
k_values = [2]  # 2, 3, 5, 100

for thres in thres_list:
    # Calculate z-scores threshold mask
    pr_zscore_above_thres = z_scores > thres
    
    for k in k_values:
        # Get top K mask using your provided function
        top_k_mask = get_top_k_mask(z_scores, k=k)
        
        # Apply both filters BEFORE the rolling window
        # Only keep points that are both above threshold AND in top K
        combined_filter = pr_zscore_above_thres & top_k_mask
        
        for period in periods:
            # Now calculate rolling window on the pre-filtered data
            rolling_below_threshold = combined_filter.rolling(window=period, min_periods=1).mean()
            final_mask = rolling_below_threshold != 0
            
            # Apply the final mask to prediction
            new_fac = prediction.mask(final_mask)
            
            # Calculate the count ratio between new_fac and prediction
            count_ratio = new_fac.count(axis=1) / prediction.count(axis=1)
            
            # Resample by day and calculate the mean
            daily_avg_ratio = count_ratio.resample('D').mean()
            
            # Generate a dynamic title based on the current parameters
            new_fac_name = f'predict-pr_zscore_n{n}_m{m}_{thres}_p{period}_topk{k}'
            title = f"Daily Average of Count Ratio (masked / prediction)\nParameters: {new_fac_name}"
            
            # Plot the daily average ratio with the dynamic title
            daily_avg_ratio.plot(figsize=(10, 6), title=title)
            plt.xlabel("Date")
            plt.ylabel("Count Ratio")
            plt.grid(True)
            plt.show()
            
            # Save to parquet file
            # new_fac.to_parquet(przsc_dir / f'{new_fac_name}.parquet')
