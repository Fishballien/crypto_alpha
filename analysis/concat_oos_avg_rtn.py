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


# %%
cluster_name = 'v6'
rolling_params = {
    'fstart': datetime(2021, 1, 1),
    'pstart': datetime(2022, 1, 1),
    'puntil': datetime(2024, 3, 29),
    'rrule_kwargs': {'freq': 'M', 'interval': 1, 'bymonthday': 1},
    'window_kwargs': {'months': 24},
    'end_by': 'time'
    }


# %%
cluster_dir = CLUSTER_RES_DIR / cluster_name
robustness_dir = cluster_dir / 'robustness'
robustness_dir.mkdir(parents=True, exist_ok=True)

rolling = RollingPeriods(**rolling_params)
filter_periods = rolling.fit_periods
predict_periods = rolling.predict_periods
oos_avg_rtn_list = []
for fltp, pp in list(zip(filter_periods, predict_periods)):
    is_period_name = period_shortcut(*fltp)
    oos_period_name = period_shortcut(*pp)
    cluster_info_path = cluster_dir / f'cluster_info_{is_period_name}.csv'
    cluster_info = pd.read_csv(cluster_info_path)
    for idx in cluster_info.index:
        process_name, factor_name, direction = cluster_info.loc[idx, ['process_name', 'factor', 'direction']]
        if process_name.startswith('f'):
            process_dir = RESULT_DIR / process_name / '240T' / 'data' 
        else:
            process_dir = RESULT_DIR / process_name / 'data'
        factor_path = process_dir / f'gp_{factor_name}.parquet'
        factor_gp = pd.read_parquet(factor_path)
        oos_gp = factor_gp[(factor_gp.index >= pp[0]) & (factor_gp.index <= pp[1])]
        if idx == 0:
            oos_avg = oos_gp * direction
        else:
            oos_avg += oos_gp * direction
    oos_avg = oos_avg / len(cluster_info)
    oos_avg_rtn_list.append(oos_avg)
oos_avg_rtn = pd.concat(oos_avg_rtn_list)
oos_avg_rtn.to_parquet(robustness_dir / 'oos_avg_rtn.parquet')
    
FONTSIZE_L1 = 20
FONTSIZE_L2 = 18
FONTSIZE_L3 = 15

title = f"{cluster_name}_oos_avg_rtn"

fig = plt.figure(figsize=(20, 20), dpi=100, layout="constrained")
spec = fig.add_gridspec(ncols=1, nrows=1)

ax0 = fig.add_subplot(spec[0, :])
ax0.set_title(f'{title}', fontsize=FONTSIZE_L1, pad=25)
ax0.plot(oos_avg_rtn['long_short_0'].cumsum(), color='k', label='oos_avg_rtn', linewidth=3)

for ax in (ax0,):
    ax.grid(linestyle=":")
    ax.legend(loc="upper left", borderaxespad=0.5, fontsize=FONTSIZE_L3)
    ax.tick_params(labelsize=FONTSIZE_L2, pad=15)

plt.savefig(robustness_dir / f"{title}.jpg", dpi=100, bbox_inches="tight")
plt.close()