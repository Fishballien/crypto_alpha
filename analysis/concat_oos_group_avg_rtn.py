# -*- coding: utf-8 -*-
"""
Created on Mon May 13 13:15:55 2024

@author: Xintang Zheng

"""
# %% imports
import sys
import yaml
import toml
from pathlib import Path
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt


# %% add sys path
file_path = Path(__file__).resolve()
file_dir = file_path.parents[0]
project_dir = file_path.parents[1]
sys.path.append(str(project_dir))


# %%
#from dirs import CLUSTER_RES_DIR, RESULT_DIR
from utils.timeutils import RollingPeriods, period_shortcut


# %%
# cluster_name = 'v4_fix_rtn_fill_0'
cluster_name = 'agg_240715_2'

rolling_params = {
    'fstart': datetime(2021, 1, 1),
    'pstart': datetime(2023, 7, 1),
    'puntil': datetime(2024, 4, 30),
    'rrule_kwargs': {'freq': 'M', 'interval': 1, 'bymonthday': 1},
    'window_kwargs': {'months': 24},
    'end_by': 'time'
    }


# %% path
path_config_path = project_dir / '.path_config.yaml'
with path_config_path.open('r') as file:
    path_config = yaml.safe_load(file)
    
result_dir = Path(path_config['result'])
param_dir = Path(path_config['param'])


# %%
cluster_dir = result_dir / 'cluster' / cluster_name
robustness_dir = cluster_dir / 'robustness'
robustness_dir.mkdir(parents=True, exist_ok=True)
test_dir = result_dir / 'test'

cl_params = toml.load(param_dir / 'cluster' / f'{cluster_name}.toml')
feval_name = cl_params['feval_name']
feval_params = toml.load(param_dir / 'feval' / f'{feval_name}.toml')
test_name = feval_params['test_name']

rolling = RollingPeriods(**rolling_params)
filter_periods = rolling.fit_periods
predict_periods = rolling.predict_periods
oos_avg_rtn_list = []
oos_avg_hsr_list = []
factor_num_list = []
for fltp, pp in list(zip(filter_periods, predict_periods)):
    is_period_name = period_shortcut(*fltp)
    oos_period_name = period_shortcut(*pp)
    cluster_info_path = cluster_dir / f'cluster_info_{is_period_name}.csv'
    cluster_info = pd.read_csv(cluster_info_path)
    cluster_group = cluster_info.groupby('group')
    group_num = len(cluster_group)
    oos_avg = None
    oos_avg_hsr = None
    for i_g, (group_name, group_data) in enumerate(cluster_group):
        # hsr_compare = []
        len_group = len(group_data)
        group_oos_avg = None
        group_oos_hsr = None
        for i_d, idx in enumerate(group_data.index):
            tag_name, process_name, factor_name, direction = group_data.loc[idx, ['tag_name', 'process_name', 'factor', 'direction']]
            process_dir = (test_dir / test_name / tag_name if tag_name is not np.nan or None
                        else test_dir / test_name)
            data_dir = process_dir / process_name / 'data' 
            factor_gp_path = data_dir / f'gp_{factor_name}.parquet'
            factor_hsr_path = data_dir / f'hsr_{factor_name}.parquet'
            factor_gp = pd.read_parquet(factor_gp_path)
            factor_hsr = pd.read_parquet(factor_hsr_path)
            oos_gp = factor_gp[(factor_gp.index >= pp[0]) & (factor_gp.index <= pp[1])]
            oos_hsr = factor_hsr[(factor_hsr.index >= pp[0]) & (factor_hsr.index <= pp[1])]
            # hsr_compare.append(list(oos_hsr['turnover'])[0])
            if np.isnan(oos_hsr).any().any():
                len_group -= 1
            else:
                if group_oos_avg is None:
                    group_oos_avg = oos_gp * direction
                else:
                    group_oos_avg += oos_gp * direction
                if group_oos_hsr is None:
                    group_oos_hsr = oos_hsr
                else:
                    group_oos_hsr += oos_hsr
        if len_group == 0:
            group_num -= 1
            continue
        group_oos_avg = group_oos_avg / len_group
        group_oos_hsr = group_oos_hsr / len_group
        if oos_avg is None:
            oos_avg = group_oos_avg
        else:
            oos_avg += group_oos_avg
        if oos_avg_hsr is None:
            oos_avg_hsr = group_oos_hsr
        else:
            oos_avg_hsr += group_oos_hsr
    oos_avg = oos_avg / group_num
    oos_avg_hsr = oos_avg_hsr / group_num
    oos_avg_rtn_list.append(oos_avg)
    oos_avg_hsr_list.append(oos_avg_hsr)
    factor_num_list.append((pp[0], len(cluster_info['group'].unique())))
oos_avg_rtn = pd.concat(oos_avg_rtn_list)
oos_avg_rtn.to_parquet(robustness_dir / 'oos_group_avg_rtn.parquet')
oos_avg_hsr = pd.concat(oos_avg_hsr_list)
oos_avg_hsr.to_parquet(robustness_dir / 'oos_group_avg_hsr.parquet')
factor_num_info = pd.DataFrame(factor_num_list, columns=['period_start', 'factor_num'])
factor_num_info.to_parquet(robustness_dir / 'factor_group_num.parquet')
    
FONTSIZE_L1 = 20
FONTSIZE_L2 = 18
FONTSIZE_L3 = 15

title = f"{cluster_name}_oos_group_avg_rtn"

fig = plt.figure(figsize=(20, 20), dpi=100, layout="constrained")
spec = fig.add_gridspec(ncols=1, nrows=4)

ax0 = fig.add_subplot(spec[:2, :])
ax0.set_title(f'{title}', fontsize=FONTSIZE_L1, pad=25)
ax0.plot(oos_avg_rtn['long_short_0'].cumsum(), color='k', label='oos_avg_rtn', linewidth=3)

ax1 = fig.add_subplot(spec[2, :])
ax1.set_title(f'{title}', fontsize=FONTSIZE_L1, pad=25)
oos_avg_hsr = oos_avg_hsr.set_index((oos_avg_hsr.index.year * 100 + oos_avg_hsr.index.month) % 10000)
oos_avg_hsr.plot.bar(ax=ax1)

ax2 = fig.add_subplot(spec[-1, :])
ax2.bar(factor_num_info['period_start'], factor_num_info['factor_num'], label='factor_group_num', width=10)

for ax in (ax0, ax1, ax2,):
    ax.grid(linestyle=":")
    ax.legend(loc="upper left", borderaxespad=0.5, fontsize=FONTSIZE_L3)
    ax.tick_params(labelsize=FONTSIZE_L2, pad=15)

plt.savefig(robustness_dir / f"{title}.jpg", dpi=100, bbox_inches="tight")
plt.close()
