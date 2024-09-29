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
from scores import get_general_return_metrics


# %%
cluster_name = 'v6_poly_only'
is_target_metric = 'drawdown_recovery_ratio'
oos_target_metric = 'sharpe_ratio'
cl_rolling_params = {
    'fstart': datetime(2021, 1, 1),
    'pstart': datetime(2022, 1, 1),
    'puntil': datetime(2024, 3, 29),
    'rrule_kwargs': {'freq': 'M', 'interval': 1, 'bymonthday': 1},
    'window_kwargs': {'months': 24},
    'end_by': 'time'
    }
rolling_params = {
    'fstart': datetime(2021, 1, 1),
    'pstart': datetime(2022, 1, 1),
    'puntil': datetime(2024, 3, 29),
    'rrule_kwargs': {'freq': 'M', 'interval': 1, 'bymonthday': 1},
    'window_kwargs': {'months': 6},
    'end_by': 'time'
    }


# %%
cluster_dir = CLUSTER_RES_DIR / cluster_name
robustness_dir = cluster_dir / 'robustness' / f"{is_target_metric}2{oos_target_metric}"
robustness_dir.mkdir(parents=True, exist_ok=True)

cl_rolling = RollingPeriods(**cl_rolling_params)
filter_periods = cl_rolling.fit_periods
predict_periods = cl_rolling.predict_periods
rolling = RollingPeriods(**rolling_params)
fit_periods = rolling.fit_periods

for fltp, fp, pp in list(zip(filter_periods, fit_periods, predict_periods)):
    cl_period_name = period_shortcut(*fltp)
    is_period_name = period_shortcut(*fp)
    oos_period_name = period_shortcut(*pp)
    cluster_info_path = cluster_dir / f'cluster_info_{cl_period_name}.csv'
    cluster_info = pd.read_csv(cluster_info_path)
    is_metric_list = []
    oos_metric_list = []
    for idx in cluster_info.index:
        process_name, factor_name, direction = cluster_info.loc[idx, ['process_name', 'factor', 'direction']]
        if process_name.startswith('f'):
            process_dir = RESULT_DIR / process_name / '240T' / 'data' 
        else:
            process_dir = RESULT_DIR / process_name / 'data'
        factor_path = process_dir / f'gp_{factor_name}.parquet'
        factor_gp = pd.read_parquet(factor_path)
        is_gp = factor_gp[(factor_gp.index >= fp[0]) & (factor_gp.index <= fp[1])]['long_short_0']
        oos_gp = factor_gp[(factor_gp.index >= pp[0]) & (factor_gp.index <= pp[1])]['long_short_0']
        is_metric_list.append(get_general_return_metrics(is_gp * direction))
        oos_metric_list.append(get_general_return_metrics(oos_gp * direction))
    is_metrics = pd.DataFrame(is_metric_list)
    oos_metrics = pd.DataFrame(oos_metric_list)
    is_metrics.to_csv(robustness_dir / f'is_{is_target_metric}_{is_period_name}.csv', index=None)
    oos_metrics.to_csv(robustness_dir / f'oos_{oos_target_metric}_{oos_period_name}.csv', index=None)
    
    FONTSIZE_L1 = 20
    FONTSIZE_L2 = 18
    FONTSIZE_L3 = 15
    
    title = f"{is_target_metric}2{oos_target_metric}_is_{is_period_name}_oos_{oos_period_name}"
    
    fig = plt.figure(figsize=(20, 20), dpi=100, layout="constrained")
    spec = fig.add_gridspec(ncols=1, nrows=1)
    
    ax0 = fig.add_subplot(spec[0, :])
    ax0.set_title(f'{title}', fontsize=FONTSIZE_L1, pad=25)
    ax0.scatter(is_metrics[is_target_metric], oos_metrics[oos_target_metric])
    ax0.axhline(0, color='k', linestyle='--')
    ax0.axvline(0, color='k', linestyle='--')
    ax0.set_xlabel('IS', fontsize=FONTSIZE_L2)
    ax0.set_ylabel('OOS', fontsize=FONTSIZE_L2)
    
    for ax in (ax0,):
        ax.grid(linestyle=":")
        # ax.legend(loc="upper left", borderaxespad=0.5, fontsize=FONTSIZE_L3)
        ax.tick_params(labelsize=FONTSIZE_L2, pad=15)

    plt.savefig(robustness_dir / f"{title}.jpg", dpi=100, bbox_inches="tight")
    plt.close()