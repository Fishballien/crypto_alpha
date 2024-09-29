# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 10:06:38 2024

@author: Xintang Zheng

"""
# %% imports
import sys
import yaml
from pathlib import Path
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt


# %% add sys path
file_path = Path(__file__).resolve()
file_dir = file_path.parents[0]
project_dir = file_path.parents[1]
sys.path.append(str(project_dir))


# %%
analysis_name = 'f68_no_range_limit_compare'
factor_path_list = [
    ('/mnt/Data/xintang/crypto/multi_factor/factor_test_by_alpha/results/test/regular_twd30_sp30/zxt/2408/orderbook_imbalance/f40_bidask_amount_ratio/futures/lob/1min_2_30min/data',
     'total'),
    ('/mnt/Data/xintang/crypto/multi_factor/factor_test_by_alpha/results/test/regular_twd30_sp30/zxt/2408/orderbook_imbalance/f67_imb_no_range_limit/futures/lob/1min_2_30min/data',
     'norl_in_10'),
    ('/mnt/Data/xintang/crypto/multi_factor/factor_test_by_alpha/results/test/regular_twd30_sp30/zxt/2408/orderbook_imbalance/f67_imb_no_range_limit/futures/lob/1min_2_30min/data',
     'norl_in_03'),
    ('/mnt/Data/xintang/crypto/multi_factor/factor_test_by_alpha/results/test/regular_twd30_sp30/zxt/2408/orderbook_imbalance/f67_imb_no_range_limit/futures/lob/1min_2_30min/data',
     'norl_in_02'),
    ('/mnt/Data/xintang/crypto/multi_factor/factor_test_by_alpha/results/test/regular_twd30_sp30/zxt/2408/orderbook_imbalance/f67_imb_no_range_limit/futures/lob/1min_2_30min/data',
     'norl_in_01'),
    ('/mnt/Data/xintang/crypto/multi_factor/factor_test_by_alpha/results/test/regular_twd30_sp30/zxt/2408/orderbook_imbalance/f67_imb_no_range_limit/futures/lob/1min_2_30min/data',
     'norl_in_005'),
    ('/mnt/Data/xintang/crypto/multi_factor/factor_test_by_alpha/results/test/regular_twd30_sp30/zxt/2408/orderbook_imbalance/f68_imb_no_range_limit_merge/futures/lob/1min_2_30min/data',
     'norl_in_merge_03_10'),
    ('/mnt/Data/xintang/crypto/multi_factor/factor_test_by_alpha/results/test/regular_twd30_sp30/zxt/2408/orderbook_imbalance/f68_imb_no_range_limit_merge/futures/lob/1min_2_30min/data',
     'norl_in_merge_02_03_10'),
    ('/mnt/Data/xintang/crypto/multi_factor/factor_test_by_alpha/results/test/regular_twd30_sp30/zxt/2408/orderbook_imbalance/f68_imb_no_range_limit_merge/futures/lob/1min_2_30min/data',
     'norl_in_merge_015_02_03_10'),
    ('/mnt/Data/xintang/crypto/multi_factor/factor_test_by_alpha/results/test/regular_twd30_sp30/zxt/2408/orderbook_imbalance/f68_imb_no_range_limit_merge/futures/lob/1min_2_30min/data',
     'norl_in_merge_01_015_02_03_10'),
    # ('/mnt/Data/xintang/crypto/multi_factor/factor_test_by_alpha/results/test/regular_twd30_sp30/zxt/2408/orderbook_imbalance/f40_bidask_amount_ratio/futures/lob/1min_2_30min/data',
    #  'total_mmt15min_ma'),
    # ('/mnt/Data/xintang/crypto/multi_factor/factor_test_by_alpha/results/test/regular_twd30_sp30/zxt/2408/orderbook_imbalance/f51_ba_amt_ratio_filter_by_dist_out/futures/lob/1min_2_30min/data',
    #   'n75_out_01_mmt15min_ma'),
    # ('/mnt/Data/xintang/crypto/multi_factor/factor_test_by_alpha/results/test/regular_twd30_sp30/zxt/2408/orderbook_imbalance/f56_ba_amt_ratio_fsmall_by_dist_in/futures/lob/1min_2_30min/data',
    #   'n75_in_001_mmt15min_ma'),
    # ('/mnt/Data/xintang/crypto/multi_factor/factor_test_by_alpha/results/test/regular_twd30_sp30/zxt/2408/orderbook_imbalance/f60_ba_amt_ratio_rm_hs_ll_test/futures/lob/1min_2_30min/data',
    #  'hsig75_hin001_lsig75_lout01'),
    # ('/mnt/Data/xintang/crypto/multi_factor/factor_test_by_alpha/results/test/regular_twd30/zxt/2406/orderbook_imbalance/f56_ba_amt_ratio_fsmall_by_dist_in/futures/lob/1min_2_240min/data',
    #   'n75_in_0005'),
    # ('/mnt/Data/xintang/crypto/multi_factor/factor_test_by_alpha/results/test/regular_twd30/zxt/2406/orderbook_imbalance/f56_ba_amt_ratio_fsmall_by_dist_in/futures/lob/1min_2_240min/data',
    #   'n75_in_001'),
    # ('/mnt/Data/xintang/crypto/multi_factor/factor_test_by_alpha/results/test/regular_twd30/zxt/2406/orderbook_imbalance/f56_ba_amt_ratio_fsmall_by_dist_in/futures/lob/1min_2_240min/data',
    #   'n75_in_0025'),
    # ('/mnt/Data/xintang/crypto/multi_factor/factor_test_by_alpha/results/test/regular_twd30/zxt/2406/orderbook_imbalance/f56_ba_amt_ratio_fsmall_by_dist_in/futures/lob/1min_2_240min/data',
    #   'n75_in_005'),
    # ('/mnt/Data/xintang/crypto/multi_factor/factor_test_by_alpha/results/test/regular_twd30/zxt/2406/orderbook_imbalance/f56_ba_amt_ratio_fsmall_by_dist_in/futures/lob/1min_2_240min/data',
    #   'n75_in_01'),
    ]
sp = '240T'
date_start = datetime(2023, 7, 1)
date_end = datetime(2024, 7, 31)


# %%
path_config_path = project_dir/ '.path_config.yaml'
with path_config_path.open('r') as file:
    path_config = yaml.safe_load(file)
    
result_dir = Path(path_config['result'])


# %%
save_dir = result_dir / 'analysis' / analysis_name
save_dir.mkdir(exist_ok=True, parents=True)


# %%
# fe = FactorEvaluation(eval_name, process_name_list, factor_name_list, sp, result_dir)
# fe.eval_one_period(date_start, date_end)

gp_dict = {}
for factor_dir, factor_name in factor_path_list:
    df_gp = pd.read_parquet(Path(factor_dir) / f'gp_{factor_name}.parquet')
    df_gp = df_gp[(df_gp.index >= date_start) & (df_gp.index <= date_end)]
    if factor_name in gp_dict.keys():
        factor_name = f'{factor_name}_1'
    gp_dict[factor_name] = df_gp['long_short_0']
    
# Read HSR files and store in a dictionary
hsr_dict = {}
for factor_dir, factor_name in factor_path_list:
    df_hsr = pd.read_parquet(Path(factor_dir) / f'hsr_{factor_name}.parquet')
    df_hsr = df_hsr[(df_hsr.index >= date_start) & (df_hsr.index <= date_end)] # & (df_hsr.index <= date_end)
    df_hsr = df_hsr.set_index((df_hsr.index.year * 100 + df_hsr.index.month) % 10000)
    if factor_name in hsr_dict.keys():
        factor_name = f'{factor_name}_1'
    hsr_dict[factor_name] = df_hsr['turnover']
    
    
FONTSIZE_L1 = 20
FONTSIZE_L2 = 18
FONTSIZE_L3 = 15

title = f"Comparison regards to {analysis_name}"

fig, (ax0, ax1) = plt.subplots(nrows=2, ncols=1, figsize=(20, 20), dpi=100, layout="constrained")

# Plot cumulative returns
ax0.set_title(f'{title}', fontsize=FONTSIZE_L1, pad=25)
for process_name in gp_dict:
    gp = gp_dict[process_name]
    ax0.plot(gp.cumsum(), label=process_name, linewidth=3)
# ax0.plot((gp_dict['hsig75_hin001_lsig75_lout01'] - gp_dict['total_mmt15min_ma']).cumsum(), label='diff', linewidth=3)

ax0.grid(linestyle=":")
ax0.legend(loc="upper left", borderaxespad=0.5, fontsize=FONTSIZE_L3)
ax0.tick_params(labelsize=FONTSIZE_L2, pad=15)

# Plot HSR data
hsr_colors = {process_name: ax0.get_lines()[i].get_color() for i, process_name in enumerate(gp_dict.keys())}
width = 0.1  # Adjust the width of the bars to be thinner
for i, process_name in enumerate(hsr_dict.keys()):
    df_hsr = hsr_dict[process_name]
    dates = df_hsr.index.unique()
    date_indices = range(len(dates))
    ax1.bar([x + i * width for x in date_indices], df_hsr, width=width, label=process_name, color=hsr_colors[process_name])

# Set xtick labels to actual dates
ax1.set_xticks([x + width * (len(hsr_dict) - 1) / 2 for x in date_indices])
ax1.set_xticklabels([f'{date // 100:02d}-{date % 100:02d}' for date in dates], rotation=45)


ax1.grid(linestyle=":")
ax1.legend(loc="upper left", borderaxespad=0.5, fontsize=FONTSIZE_L3)
ax1.tick_params(labelsize=FONTSIZE_L2, pad=15)

plt.savefig(save_dir / "comparison_plot.jpg", dpi=100, bbox_inches="tight")
plt.close()