# -*- coding: utf-8 -*-
"""
Created on Mon May 13 13:57:22 2024

@author: Xintang Zheng

"""
# %% imports
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt


from dirs import CLUSTER_RES_DIR, RESULT_DIR
from timeutils import RollingPeriods, period_shortcut
from scores import get_general_return_metrics


# %%
cluster_1_name = 'v4_ma15_only'
cluster_2_name = 'v4_ma15_only_1'
target_metric = 'sharpe_ratio'
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

rolling = RollingPeriods(**rolling_params)
filter_periods = rolling.fit_periods
predict_periods = rolling.predict_periods
for fltp, pp in list(zip(filter_periods, predict_periods)):
    is_period_name = period_shortcut(*fltp)
    oos_period_name = period_shortcut(*pp)
    is_1 = pd.read_csv(r_1_dir / f'is_metrics_{is_period_name}.csv')
    oos_1 = pd.read_csv(r_1_dir / f'oos_metrics_{is_period_name}.csv')
    is_2 = pd.read_csv(r_2_dir / f'is_metrics_{is_period_name}.csv')
    oos_2 = pd.read_csv(r_2_dir / f'oos_metrics_{is_period_name}.csv')
    
    FONTSIZE_L1 = 20
    FONTSIZE_L2 = 18
    FONTSIZE_L3 = 15
    
    title = f"is_{is_period_name}_os_{oos_period_name}"
    
    fig = plt.figure(figsize=(20, 20), dpi=100, layout="constrained")
    spec = fig.add_gridspec(ncols=1, nrows=1)
    
    ax0 = fig.add_subplot(spec[0, :])
    ax0.set_title(f'{title}', fontsize=FONTSIZE_L1, pad=25)
    ax0.scatter(is_1[target_metric], oos_1[target_metric], 
                label=f'{cluster_1_name}: {len(is_1)}')
    ax0.scatter(is_2[target_metric], oos_2[target_metric], 
                label=f'{cluster_2_name}: {len(is_2)}')
    ax0.axhline(0, color='k', linestyle='--')
    ax0.axvline(0, color='k', linestyle='--')
    ax0.set_xlabel('IS', fontsize=FONTSIZE_L2)
    ax0.set_ylabel('OOS', fontsize=FONTSIZE_L2)
    
    for ax in (ax0,):
        ax.grid(linestyle=":")
        ax.legend(loc="upper left", borderaxespad=0.5, fontsize=FONTSIZE_L3)
        ax.tick_params(labelsize=FONTSIZE_L2, pad=15)

    plt.savefig(result_dir / f"{title}.jpg", dpi=100, bbox_inches="tight")
    plt.close()