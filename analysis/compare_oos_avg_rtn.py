# -*- coding: utf-8 -*-
"""
Created on Mon May 13 13:15:55 2024

@author: Xintang Zheng

"""
# %% imports
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt


from dirs import CLUSTER_RES_DIR, RESULT_DIR
from timeutils import RollingPeriods, period_shortcut
from datautils import align_index


# %%
# cluster_1_name = 'v4_fix_rtn_fill_0'
cluster_1_name = 'v10'
# cluster_1_name = 'gp_v0_1'
cluster_2_name = 'gp_v5_1'
rolling_params = {
    'fstart': datetime(2021, 1, 1),
    'pstart': datetime(2022, 1, 1),
    'puntil': datetime(2024, 3, 29),
    'rrule_kwargs': {'freq': 'M', 'interval': 1, 'bymonthday': 1},
    'window_kwargs': {'months': 24},
    'end_by': 'time'
    }


# %%
r_1_dir = CLUSTER_RES_DIR / cluster_1_name / 'robustness'
r_2_dir = CLUSTER_RES_DIR / cluster_2_name / 'robustness'
result_dir = CLUSTER_RES_DIR / cluster_2_name / f'compare_with_{cluster_1_name}'
result_dir.mkdir(parents=True, exist_ok=True)

oos_avg_rtn_1 = pd.read_parquet(r_1_dir / 'oos_group_avg_rtn.parquet')
oos_avg_rtn_2 = pd.read_parquet(r_2_dir / 'oos_group_avg_rtn.parquet')
oos_avg_rtn_1, oos_avg_rtn_2 = align_index(oos_avg_rtn_1, oos_avg_rtn_2)

oos_avg_hsr_1 = pd.read_parquet(r_1_dir / 'oos_group_avg_hsr.parquet')
oos_avg_hsr_2 = pd.read_parquet(r_2_dir / 'oos_group_avg_hsr.parquet')
oos_avg_hsr_1, oos_avg_hsr_2 = align_index(oos_avg_hsr_1, oos_avg_hsr_2)
hsr_merged = pd.merge(oos_avg_hsr_1, oos_avg_hsr_2, left_index=True, right_index=True,
                      suffixes=[f'_{cluster_1_name}', f'_{cluster_2_name}'])

factor_num_info_1 = pd.read_parquet(r_1_dir / 'factor_group_num.parquet').set_index('period_start')
factor_num_info_2 = pd.read_parquet(r_2_dir / 'factor_group_num.parquet').set_index('period_start')
factor_num_info_1, factor_num_info_2 = align_index(factor_num_info_1, factor_num_info_2)
factor_num_merged = pd.merge(factor_num_info_1, factor_num_info_2, on='period_start', 
                             suffixes=[f'_{cluster_1_name}', f'_{cluster_2_name}'])
# factor_num_merged.set_index('period_start', inplace=True)
    
FONTSIZE_L1 = 20
FONTSIZE_L2 = 18
FONTSIZE_L3 = 15

title = f"{cluster_1_name}_vs_{cluster_2_name}_group"

fig = plt.figure(figsize=(20, 20), dpi=100, layout="constrained")
spec = fig.add_gridspec(ncols=1, nrows=4)

ax0 = fig.add_subplot(spec[:2, :])
ax0.set_title(f'{title}', fontsize=FONTSIZE_L1, pad=25)
ax0.plot(oos_avg_rtn_1['long_short_0'].cumsum(), label=cluster_1_name, linewidth=3, color=plt.cm.Dark2(0))
ax0.plot(oos_avg_rtn_2['long_short_0'].cumsum(), label=cluster_2_name, linewidth=3, color=plt.cm.Dark2(1))
ax0.plot((oos_avg_rtn_2['long_short_0'].cumsum() - oos_avg_rtn_1['long_short_0'].cumsum()), 
         label='diff', linewidth=3, color=plt.cm.Dark2(2))

ax1 = fig.add_subplot(spec[2, :])
ax12 = ax1.twinx()
hsr_merged = hsr_merged.set_index((hsr_merged.index.year * 100 + hsr_merged.index.month) % 10000)
# factor_num_merged.plot(kind='bar', ax=ax1)
# ax1.bar(factor_num_info_1['period_start'], factor_num_info_1['factor_num'], label=cluster_1_name, width=2)
# ax1.bar(factor_num_info_2['period_start'], factor_num_info_2['factor_num'], label=cluster_2_name, width=2)

# 绘制第一个数据框的条形图
hsr_merged[f'turnover_{cluster_1_name}'].plot(kind='bar', ax=ax1, width=0.4, position=1, color=plt.cm.Dark2(0), 
                                         align='center')
ax1.set_ylabel(f'Turnover {cluster_1_name}', fontsize=FONTSIZE_L2, color=plt.cm.Dark2(0))
ax1.tick_params(axis='y', labelcolor=plt.cm.Dark2(0))

# 绘制第二个数据框的条形图
hsr_merged[f'turnover_{cluster_2_name}'].plot(kind='bar', ax=ax12, width=0.4, position=0, color=plt.cm.Dark2(1), 
                                         align='center')
ax12.set_ylabel(f'Turnover {cluster_2_name}', fontsize=FONTSIZE_L2, color=plt.cm.Dark2(1))
ax12.tick_params(axis='y', labelcolor=plt.cm.Dark2(1))

ax2 = fig.add_subplot(spec[-1, :])
ax22 = ax2.twinx()
factor_num_merged = factor_num_merged.set_index((factor_num_merged.index.year * 100 + factor_num_merged.index.month) % 10000)
# factor_num_merged.plot(kind='bar', ax=ax1)
# ax1.bar(factor_num_info_1['period_start'], factor_num_info_1['factor_num'], label=cluster_1_name, width=2)
# ax1.bar(factor_num_info_2['period_start'], factor_num_info_2['factor_num'], label=cluster_2_name, width=2)

# 绘制第一个数据框的条形图
factor_num_merged[f'factor_num_{cluster_1_name}'].plot(kind='bar', ax=ax2, width=0.4, position=1, color=plt.cm.Dark2(0), 
                                                       align='center')
ax2.set_ylabel(f'Factor Num {cluster_1_name}', fontsize=FONTSIZE_L2, color=plt.cm.Dark2(0))
ax2.tick_params(axis='y', labelcolor=plt.cm.Dark2(0))

# 绘制第二个数据框的条形图
factor_num_merged[f'factor_num_{cluster_2_name}'].plot(kind='bar', ax=ax22, width=0.4, position=0, color=plt.cm.Dark2(1), 
                                                       align='center')
ax22.set_ylabel(f'Factor Num {cluster_2_name}', fontsize=FONTSIZE_L2, color=plt.cm.Dark2(1))
ax22.tick_params(axis='y', labelcolor=plt.cm.Dark2(1))

for ax in (ax0, ax1, ax2,):
    ax.grid(linestyle=":")
    ax.legend(loc="upper left", borderaxespad=0.5, fontsize=FONTSIZE_L3)
    ax.tick_params(labelsize=FONTSIZE_L2, pad=15)
    
for ax in (ax12, ax22):
    ax.grid(linestyle=":")
    ax.legend(loc="upper right", borderaxespad=0.5, fontsize=FONTSIZE_L3)
    ax.tick_params(labelsize=FONTSIZE_L2, pad=15)

plt.savefig(result_dir / f"{title}.jpg", dpi=100, bbox_inches="tight")
plt.close()