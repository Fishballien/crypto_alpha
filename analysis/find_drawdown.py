# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 10:36:52 2024

@author: Xintang Zheng

星星: ★ ☆ ✪ ✩ 🌟 ⭐ ✨ 🌠 💫 ⭐️
勾勾叉叉: ✓ ✔ ✕ ✖ ✅ ❎
报警啦: ⚠ ⓘ ℹ ☣
箭头: ➔ ➜ ➙ ➤ ➥ ↩ ↪
emoji: 🔔 ⏳ ⏰ 🔒 🔓 🛑 🚫 ❗ ❓ ❌ ⭕ 🚀 🔥 💧 💡 🎵 🎶 🧭 📅 🤔 🧮 🔢 📊 📈 📉 🧠 📝

"""
# %% imports
import sys
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt


# %% add sys path
file_path = Path(__file__).resolve()
file_dir = file_path.parents[0]
project_dir = file_path.parents[1]
sys.path.append(str(project_dir))


# %% import self-defined
from synthesis.filter_methods import *
from utils.datautils import align_to_primary, align_index


# %%
## 2312
# =============================================================================
# feval_name = 'agg_240715'
# long_period = '211201_231201'
# short_period = '230901_231201'
# target_period = '231201_231210'
# date_start = '20231201'
# date_end = '20231210'
# =============================================================================

## 2308
feval_name = 'agg_240902'
long_period = '220416_240416'
short_period = '240116_240416'
target_period = '240501_240730'
date_start = '20240501'
date_end = '20240730'

feval_dir = Path('/mnt/Data/xintang/crypto/multi_factor/factor_test_by_alpha/results/factor_evaluation')
test_dir = Path('/mnt/Data/xintang/crypto/multi_factor/factor_test_by_alpha/results/test')
result_dir = feval_dir / feval_name
feval_path_long = feval_dir / feval_name / f'factor_eval_{long_period}.csv'
feval_path_short = feval_dir / feval_name / f'factor_eval_{short_period}.csv'
feval_path_target = feval_dir / feval_name / f'factor_eval_{target_period}.csv'
filter_func = filter_func_v6
rec_filter_func = filter_func_rec_v0


# %%
feval_long = pd.read_csv(feval_path_long)
feval_short = pd.read_csv(feval_path_short)
feval_target = pd.read_csv(feval_path_target)

selected_idx = filter_func(feval_long)
if rec_filter_func is not None:
    feval_short = align_to_primary(feval_long, feval_short, 'process_name', 'factor')
    selected_idx_rec = rec_filter_func(feval_short)
    selected_idx = selected_idx & selected_idx_rec
feval_selected = feval_long[selected_idx].set_index(['test_name', 'tag_name', 'process_name', 'factor'])

feval_target = feval_target.set_index(['test_name', 'tag_name', 'process_name', 'factor'])

feval_selected, feval_target = align_index(feval_selected, feval_target)
feval_target['return'] = feval_selected['direction'] * feval_target['direction'] * feval_target['return']

# 生成日期列名
date_range = pd.date_range(start=date_start, end=date_end)
date_columns = [date.strftime('%Y%m%d') for date in date_range]

# 计算每日收益率并拼接到feval_target
daily_returns = []
for idx_g in feval_target.index:
    test_name, tag_name, process_name, factor = idx_g
    path = test_dir / test_name / tag_name / process_name / 'data' / f'gp_{factor}.parquet'
    df_gp = pd.read_parquet(path)
    df_gp = df_gp[(df_gp.index >= date_start) & (df_gp.index <= date_end)]
    direction = feval_target.loc[idx_g, 'direction'] * feval_selected.loc[idx_g, 'direction']
    daily_return = (df_gp['long_short_0'] * direction).values
    daily_returns.append(daily_return)

# 将每日收益率列表转换为DataFrame
daily_returns_df = pd.DataFrame(daily_returns, index=feval_target.index, columns=date_columns)

# 将每日收益率拼接到feval_target
feval_target_with_returns = pd.concat([feval_target, daily_returns_df], axis=1)

# 保存结果（如果需要，可以取消注释下面的行）
feval_target_with_returns.to_csv(result_dir / f'feval_target_with_daily_returns_{target_period}.csv')

# 设置字体大小
FONTSIZE_L1 = 20
FONTSIZE_L2 = 18
FONTSIZE_L3 = 15

# 设置标题
title = f'{feval_name} Return {target_period}'

# 创建一个图像，包含两个子图（上下布局）
fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(54, 54), dpi=100, gridspec_kw={'height_ratios': [2, 3]})

# 第一张子图
ax0.set_title(f'{title}', fontsize=FONTSIZE_L1, pad=25)
for i, (process_name, group_data) in enumerate(feval_target_with_returns.groupby('process_name')):
    ax0.hist(group_data['return'], label=process_name, alpha=.5, bins=100, color=plt.cm.tab20(i))

ax0.grid(linestyle=":")
ax0.legend(loc="upper left", borderaxespad=0.5, fontsize=FONTSIZE_L3)
ax0.tick_params(labelsize=FONTSIZE_L2, pad=15)

# 第二张子图
ax1.set_title(f'{title}', fontsize=FONTSIZE_L1, pad=25)
for i, (process_name, group_data) in enumerate(feval_target_with_returns.groupby('process_name')):
    for idx_g in group_data.index:
        daily_returns = group_data.loc[idx_g, date_columns].values.flatten()
        ax1.plot(date_range, daily_returns.cumsum(), alpha=.5, color=plt.cm.tab20(i))

ax1.grid(linestyle=":")
ax1.legend(loc="upper left", borderaxespad=0.5, fontsize=FONTSIZE_L3)
ax1.tick_params(labelsize=FONTSIZE_L2, pad=15)

# 保存图像
plt.savefig(result_dir / f"factor_rtn_{target_period}.jpg", dpi=100, bbox_inches="tight")
plt.close()