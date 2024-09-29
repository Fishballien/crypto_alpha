# -*- coding: utf-8 -*-
"""
Created on Tue Aug 13 09:42:02 2024

@author: Xintang Zheng

星星: ★ ☆ ✪ ✩ 🌟 ⭐ ✨ 🌠 💫 ⭐️
勾勾叉叉: ✓ ✔ ✕ ✖ ✅ ❎
报警啦: ⚠ ⓘ ℹ ☣
箭头: ➔ ➜ ➙ ➤ ➥ ↩ ↪
emoji: 🔔 ⏳ ⏰ 🔒 🔓 🛑 🚫 ❗ ❓ ❌ ⭕ 🚀 🔥 💧 💡 🎵 🎶 🧭 📅 🤔 🧮 🔢 📊 📈 📉 🧠 📝

"""
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


from utils.datautils import align_index


path_new = 'D:/crypto/multi_factor/factor_test_by_alpha/debug/twd30_sp240_new.parquet'
path_old = 'D:/crypto/multi_factor/factor_test_by_alpha/debug/twd30_sp240.parquet'


twap_new = pd.read_parquet(path_new)
twap_old = pd.read_parquet(path_old)


cols = twap_old.columns.intersection(twap_new.columns)
twap_new_test = twap_new[cols]
twap_old_test = twap_old[cols]
twap_new_test, twap_old_test = align_index(twap_new_test, twap_old_test)

res_dir = Path(r'D:\crypto\multi_factor\factor_test_by_alpha\debug\twap_diff_with_mask_new')
res_dir.mkdir(parents=True, exist_ok=True)

# for col in cols:
#     plt.figure(figsize=(10, 6))
#     plt.plot(twap_new_test[col], label='Org', color='blue')
#     plt.plot(twap_old_test[col], label='Mask New', color='orange')
#     plt.title(col)
#     plt.grid(True)
#     plt.legend()
#     plt.savefig(res_dir / f"{col}.jpg")
#     plt.show()
    
    
def find_nan_segments(series):
    """
    查找Series中所有前后有数据，中间是连续NaN的段落，并返回这些NaN段落的起始和结束索引。
    
    参数:
    series (pd.Series): 需要检查的Pandas序列。
    
    返回:
    List[Tuple]: 所有满足条件的NaN段落的起始和结束索引的列表。
    """
    nan_positions = series.isna()
    gap_periods = []
    inside_gap = False
    start_index = None

    for i in range(1, len(series) - 1):
        if nan_positions.iloc[i]:
            # 如果当前是NaN，且前一个不是NaN，说明可能进入了一个gap
            if not nan_positions.iloc[i-1]:
                inside_gap = True
                start_index = series.index[i]
        else:
            # 如果当前不是NaN，且之前在gap中，说明gap结束
            if inside_gap:
                end_index = series.index[i-1]
                gap_periods.append((start_index, end_index))
                inside_gap = False
                start_index = None

    # 特殊情况：处理序列末尾连续的NaN段落，如果最后一个数据点仍然是NaN，不记录该段落
    if inside_gap and not nan_positions.iloc[-1]:
        end_index = series.index[-2]
        gap_periods.append((start_index, end_index))

    return gap_periods


for col in twap_new_test.columns:
    gap_indices = find_nan_segments(twap_new_test[col].iloc[:-1])
    if gap_indices:
        print(col, gap_indices)