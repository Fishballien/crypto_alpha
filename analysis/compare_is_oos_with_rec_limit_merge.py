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
cluster_name = 'v4_ma15_only_1'
# cluster_name = 'v6_poly_only'
# is_target_metric = 'return_annualized'
is_target_metric_list = ['sharpe_ratio', "calmar_ratio", "sortino_ratio", 
                         "burke_ratio", "ulcer_index", "drawdown_recovery_ratio"]
oos_target_metric = 'sharpe_ratio'
mode_list = ['is', 'rec', 'relative']
long_p = 24
short_p = 3
cl_rolling_params = {
    'fstart': datetime(2021, 1, 1),
    'pstart': datetime(2022, 1, 1),
    'puntil': datetime(2024, 3, 29),
    'rrule_kwargs': {'freq': 'M', 'interval': 1, 'bymonthday': 1},
    'window_kwargs': {'months': long_p},
    'end_by': 'time'
    }
rec_rolling_params = {
    'fstart': datetime(2021, 1, 1),
    'pstart': datetime(2022, 1, 1),
    'puntil': datetime(2024, 3, 29),
    'rrule_kwargs': {'freq': 'M', 'interval': 1, 'bymonthday': 1},
    'window_kwargs': {'months': short_p},
    'end_by': 'time'
    }


# %%
for mode in mode_list:
    cluster_dir = CLUSTER_RES_DIR / cluster_name
    dir_suffix = f'{short_p}in{long_p}' if mode == 'relative' else (f'{short_p}' if mode == 'rec' else f'{long_p}')
    robustness_dir = cluster_dir / 'robustness' / f"metrics2{oos_target_metric}_{mode}_{dir_suffix}"
    robustness_dir.mkdir(parents=True, exist_ok=True)
    
    cl_rolling = RollingPeriods(**cl_rolling_params)
    filter_periods = cl_rolling.fit_periods
    predict_periods = cl_rolling.predict_periods
    rolling = RollingPeriods(**rec_rolling_params)
    rec_periods = rolling.fit_periods
    
    FONTSIZE_L1 = 20
    FONTSIZE_L2 = 18
    FONTSIZE_L3 = 15

    fig = plt.figure(figsize=(36, 27), dpi=100, layout="constrained")
    spec = fig.add_gridspec(ncols=3, nrows=2)

    ax0 = fig.add_subplot(spec[0, 0])
    ax1 = fig.add_subplot(spec[0, 1])
    ax2 = fig.add_subplot(spec[0, 2])
    ax3 = fig.add_subplot(spec[1, 0])
    ax4 = fig.add_subplot(spec[1, 1])
    ax5 = fig.add_subplot(spec[1, 2])
    
    for fltp, rp, pp in list(zip(filter_periods, rec_periods, predict_periods)):
        is_period_name = period_shortcut(*fltp)
        rec_period_name = period_shortcut(*rp)
        oos_period_name = period_shortcut(*pp)
        cluster_info_path = cluster_dir / f'cluster_info_{is_period_name}.csv'
        cluster_info = pd.read_csv(cluster_info_path)
        is_metric_list = []
        rec_metric_list = []
        oos_metric_list = []
        for idx in cluster_info.index:
            process_name, factor_name, direction = cluster_info.loc[idx, ['process_name', 'factor', 'direction']]
            if process_name.startswith('f'):
                process_dir = RESULT_DIR / process_name / '240T' / 'data' 
            else:
                process_dir = RESULT_DIR / process_name / 'data'
            factor_path = process_dir / f'gp_{factor_name}.parquet'
            factor_gp = pd.read_parquet(factor_path)
            is_gp = factor_gp[(factor_gp.index >= fltp[0]) & (factor_gp.index <= fltp[1])]['long_short_0']
            rec_gp = factor_gp[(factor_gp.index >= rp[0]) & (factor_gp.index <= rp[1])]['long_short_0']
            oos_gp = factor_gp[(factor_gp.index >= pp[0]) & (factor_gp.index <= pp[1])]['long_short_0']
            is_metric_list.append(get_general_return_metrics(is_gp * direction))
            rec_metric_list.append(get_general_return_metrics(rec_gp * direction))
            oos_metric_list.append(get_general_return_metrics(oos_gp * direction))
        is_metrics = pd.DataFrame(is_metric_list)
        rec_metrics = pd.DataFrame(rec_metric_list)
        oos_metrics = pd.DataFrame(oos_metric_list)
        is_metrics.to_csv(robustness_dir / f'is_{is_period_name}.csv', index=None)
        rec_metrics.to_csv(robustness_dir / f'rec_{rec_period_name}.csv', index=None)
        oos_metrics.to_csv(robustness_dir / f'oos_{oos_period_name}.csv', index=None)
        
        for is_target_metric, ax in zip(is_target_metric_list, [ax0, ax1, ax2, ax3, ax4, ax5]):
            title = f'{is_target_metric}2{oos_target_metric}'
            
            if mode == 'is':
                is_m = is_metrics[is_target_metric]
            elif mode == 'rec':
                is_m = rec_metrics[is_target_metric]
            elif mode == 'relative':
                is_m = (rec_metrics[is_target_metric] - is_metrics[is_target_metric]) / abs(is_metrics[is_target_metric])
            
            ax.set_title(f'{title}', fontsize=FONTSIZE_L1, pad=25)
            ax.scatter(is_m, oos_metrics[oos_target_metric])
            ax.set_xlabel(f'IS {mode} {is_target_metric} {dir_suffix}', fontsize=FONTSIZE_L2)
            ax.set_ylabel(f'OOS {oos_target_metric}', fontsize=FONTSIZE_L2)
        
    for ax in (ax0, ax1, ax2, ax3, ax4, ax5):
        ax.grid(linestyle=":")
        ax.axhline(0, color='k', linestyle='--')
        ax.axvline(0, color='k', linestyle='--')
        # ax.legend(loc="upper left", borderaxespad=0.5, fontsize=FONTSIZE_L3)
        ax.tick_params(labelsize=FONTSIZE_L2, pad=15)
    
    plt.savefig(robustness_dir / f"{cluster_name}_is_oos_merged_comparison_{dir_suffix}.jpg", dpi=100, bbox_inches="tight")
    plt.close()