# -*- coding: utf-8 -*-
"""
Created on Wed Aug 21 13:48:21 2024

@author: Xintang Zheng

星星: ★ ☆ ✪ ✩ 🌟 ⭐ ✨ 🌠 💫 ⭐️
勾勾叉叉: ✓ ✔ ✕ ✖ ✅ ❎
报警啦: ⚠ ⓘ ℹ ☣
箭头: ➔ ➜ ➙ ➤ ➥ ↩ ↪
emoji: 🔔 ⏳ ⏰ 🔒 🔓 🛑 🚫 ❗ ❓ ❌ ⭕ 🚀 🔥 💧 💡 🎵 🎶 🧭 📅 🤔 🧮 🔢 📊 📈 📉 🧠 📝

"""
# %% imports 
import numpy as np
import pandas as pd


# %%
def top_5_minus_bottom_95(row):
    """
    计算每行数据的前5%最大值的均值减去后95%最小值的均值，自动忽略NaN值。

    参数:
    row (pd.Series): 一行数据。

    返回:
    float: 每行前5%最大值均值减去后95%最小值均值的结果。
    """
    # Calculate quantiles, which will ignore NaN values
    q95 = row.quantile(0.95)

    # Mean of top 5% (values >= 95th percentile), ignoring NaN
    top_5_mean = row[row >= q95].mean()

    # Mean of bottom 95% (values <= 5th percentile), ignoring NaN
    bottom_95_mean = row[row < q95].mean()

    return top_5_mean / bottom_95_mean