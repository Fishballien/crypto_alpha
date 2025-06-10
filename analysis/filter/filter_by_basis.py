# -*- coding: utf-8 -*-
"""
Created on Wed Mar 12 10:14:48 2025

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
import seaborn as sns


# %%
path = 'D:/mnt/Data/Crypto/ProcessedData/FundingRate/binance_usd_funding_rates_30min.parquet'
save_dir = Path('D:/crypto/multi_factor/factor_test_by_alpha/results/analysis/backtest_filter/filter_by_funding')

# %%
funding = pd.read_parquet(path)


# %% org
# 示例：如果数据已经在变量名为funding的DataFrame中
save_name = 'predict_funding'
df = funding  # 使用您已有的funding变量


# %%
# =============================================================================
# def analyze_funding_thresholds(df, thresholds):
#     """
#     分析在不同阈值下每一行中绝对值超过阈值的元素数量，并按日期计算平均值
#     
#     参数:
#     df: 包含基差数据的DataFrame
#     thresholds: 要分析的阈值列表
#     
#     返回:
#     daily_avg_counts: 每个阈值的每日平均计数的DataFrame
#     """
#     # 确保索引是日期时间类型
#     if not isinstance(df.index, pd.DatetimeIndex):
#         df.index = pd.to_datetime(df.index)
#     
#     # 创建用于存储每个阈值结果的字典
#     results = {}
#     
#     for threshold in thresholds:
#         # 计算每行中绝对值大于阈值的元素数量
#         # 使用np.abs计算绝对值，然后比较是否大于阈值
#         # 使用sum(axis=1)沿行方向求和，即计算每行中满足条件的元素个数
#         counts = (np.abs(df) > threshold).sum(axis=1)
#         
#         # 将结果添加到字典中，使用阈值作为键
#         results[f'threshold_{threshold}'] = counts
#     
#     # 将所有结果合并到一个DataFrame中
#     counts_df = pd.DataFrame(results)
#     
#     # 添加日期列，只保留日期部分
#     counts_df['date'] = counts_df.index.date
#     
#     # 按日期分组并计算每列的平均值
#     daily_avg = counts_df.groupby('date').mean()
#     
#     return daily_avg
# 
# def plot_threshold_analysis(daily_avg, title="Daily Average Count of Elements Exceeding Thresholds"):
#     """
#     Plot the daily average count of elements exceeding thresholds
#     
#     Parameters:
#     daily_avg: DataFrame containing daily average counts for each threshold
#     title: Chart title
#     """
#     plt.figure(figsize=(14, 8))
#     
#     # Set style
#     sns.set_style("whitegrid")
#     
#     # Pastel, soft, cute color palette
#     cute_palette = [
#         '#FF6B6B',  # Bright coral
#         '#4ECDC4',  # Turquoise
#         '#FFD166',  # Marigold yellow
#         '#6A8EAE',  # Steel blue
#         '#FF7EB9',  # Hot pink
#         '#7CEC9F',  # Bright mint
#         '#9381FF',  # Medium purple
#         '#ACDF87',  # Light green
#         '#FF9ECE',  # Brighter pink
#         '#5DC8CD'   # Bright teal
#     ]
#     
#     # Plot lines for each threshold with cute colors
#     threshold_columns = [col for col in daily_avg.columns if col.startswith('threshold_')]
#     
#     # Sort columns in descending order (from largest to smallest threshold)
#     threshold_columns = sorted(threshold_columns, 
#                               key=lambda x: float(x.split('_')[1]) if x.split('_')[1].replace('.', '', 1).isdigit() else 0)
#     
#     for i, column in enumerate(threshold_columns):
#         # Extract threshold value from column name
#         threshold_value = column.split('_')[1]
#         color_idx = i % len(cute_palette)  # Cycle through colors if more thresholds than colors
#         
#         sns.lineplot(
#             data=daily_avg, 
#             x=daily_avg.index, 
#             y=column, 
#             label=f'Threshold = {threshold_value}',
#             color=cute_palette[color_idx],
#             linewidth=3.0,  # Slightly thicker for better visibility with pastel colors
#             alpha=1.0       # Full opacity for pastel colors
#         )
#     
#     # Set logarithmic scale for y-axis
#     plt.yscale('log')
#     
#     # Set chart properties with softer styling
#     plt.title(title, fontsize=18, fontweight='bold')
#     plt.xlabel('Date', fontsize=14)
#     plt.ylabel('Average Count (log scale)', fontsize=14)
#     
#     # Place legend outside the plot on the right side
#     plt.legend(
#         title="Threshold Levels", 
#         loc='center left', 
#         bbox_to_anchor=(1.02, 0.5),
#         fontsize=12,
#         title_fontsize=14,
#         frameon=True,
#         facecolor='#FFFCF7',  # Softer background for legend
#         edgecolor='#E6E6E6'   # Softer border
#     )
#     
#     # Adjust grid for softer look
#     plt.grid(True, alpha=0.2, linestyle='-.')
#     plt.xticks(rotation=45)
#     
#     # Add a soft background color to the plot
#     plt.gca().set_facecolor('#FCFCFC')
#     
#     plt.tight_layout()
#     
#     return plt
# 
# 
# # 定义要分析的阈值列表
# thresholds = [0.02, 0.01, 0.005, 0.002, 0.001, 0.0005]
# 
# # 分析数据
# daily_avg_counts = analyze_funding_thresholds(df, thresholds)
# 
# # 绘制结果
# plt = plot_threshold_analysis(daily_avg_counts)
# plt.savefig(save_dir / f'{save_name}.jpg', dpi=300)
# plt.show()
# 
# =============================================================================

# %%
funding_abs_limit = 0.015
funding_invalid = funding.abs() > funding_abs_limit
