# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 13:02:39 2024

@author: Xintang Zheng

星星: ★ ☆ ✪ ✩ 🌟 ⭐ ✨ 🌠 💫 ⭐️
勾勾叉叉: ✓ ✔ ✕ ✖ ✅ ❎
报警啦: ⚠ ⓘ ℹ ☣
箭头: ➔ ➜ ➙ ➤ ➥ ↩ ↪
emoji: 🔔 ⏳ ⏰ 🔒 🔓 🛑 🚫 ❗ ❓ ❌ ⭕ 🚀 🔥 💧 💡 🎵 🎶 🧭 📅 🤔 🧮 🔢 📊 📈 📉 🧠 📝

"""
# %% imports
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# %% add sys path
file_path = Path(__file__).resolve()
file_dir = file_path.parents[0]
project_dir = file_path.parents[1]
sys.path.append(str(project_dir))


# %% import self-defined
from utils.dirutils import load_path_config
from test_and_eval.scores import get_general_return_metrics
from backtest.analysisutils import top_5_minus_bottom_95


# %%
path_config = load_path_config(project_dir)
model_dir = Path(path_config['result']) / 'model'
analysis_dir = Path(path_config['result']) / 'analysis'
twap_data_dir = Path(path_config['twap_price'])


# %%
compare_name = 'backtest__p02_vs_p04_count_funding'
compare_dict = {
    # 'tf+zxt': {
    #     'model_name': 'merge_agg_241029_double3m_15d_73',
    #     'backtest_name': 'to_00125_maxmulti_2_mm_03_pf_001',
    #     },
    # 'cgy+tf+zxt': {
    #     'model_name': 'merge_agg_241109_double3m_15d_73',
    #     'backtest_name': 'to_00125_maxmulti_2_mm_03_pf_001',
    #     },
    # 'm1': {
    #     'model_name': 'merge_agg_241113_double3m_15d_73',
    #     'backtest_name': 'to_00125_maxmulti_2_mm_03_pf_001',
    #     },
    # 'cgy+tf': {
    #     'model_name': 'merge_agg_241114_tf_cgy_double3m_15d_73',
    #     'backtest_name': 'to_00125_maxmulti_2_mm_03_pf_001',
    #     },
    # 'm2': {
    #     'model_name': 'merge_agg_241114_zxt_cgy_double3m_15d_73',
    #     'backtest_name': 'to_00125_maxmulti_2_mm_03_pf_001',
    #     },
    # 'cgy': {
    #     'model_name': 'merge_agg_241214_cgy_double3m_15d_73',
    #     'backtest_name': 'to_00125_maxmulti_2_mm_03_pf_001',
    #     },
    # 'cgy+zxt': {
    #     'model_name': 'merge_agg_241214_cgy_zxt_double3m_15d_73',
    #     'backtest_name': 'to_00125_maxmulti_2_mm_03_pf_001',
    #     },
    'p0.2': {
        'model_name': 'merge_agg_241227_cgy_zxt_double3m_15d_73',
        'backtest_name': 'to_00125_maxmulti_2_mm_03_pf_001_count_funding',
        },
    'p0.4': {
        'model_name': 'merge_agg_250127_cgy_zxt_double3m_15d_73',
        'backtest_name': 'to_00125_maxmulti_2_mm_03_pf_001_count_funding',
        },
    }

start_date = '20230701'
end_date = '20250307'

sp = 30
twap_list = ['twd30_sp30']


# %% dir
save_dir = analysis_dir / compare_name
save_dir.mkdir(parents=True, exist_ok=True)


# %%
profd_dict = {}

for name, info in compare_dict.items():
    model_name = info['model_name']
    backtest_name = info['backtest_name']
    path = model_dir / model_name / 'backtest' / backtest_name / f'profit_{model_name}__{backtest_name}.parquet'
    profit = pd.read_parquet(path)
    profit.index = pd.to_datetime(profit.index)
    profit = profit.loc[start_date:end_date]
    profd = profit.resample('1d').sum()
    profd['return'] = profd[f'raw_rtn_twd30_sp{sp}'] + profd['fee'] + profd['funding']
    profd_dict[name] = profd
    
    
metrics = {name: get_general_return_metrics(profd.loc[:, 'return'].values)
                for name, profd in profd_dict.items()}
    

# %% price
# load twap & calc rtn
curr_px_path = twap_data_dir / f'curr_price_sp{sp}.parquet'
curr_price = pd.read_parquet(curr_px_path)
main_columns = curr_price.columns
to_mask = curr_price.isna()


rtn = curr_price.pct_change(int(240/sp), fill_method=None).replace([np.inf, -np.inf], np.nan)
cross_sectional_volatility = rtn.std(axis=1).resample('1d').mean()
cross_sectional_kurt = rtn.kurtosis(axis=1).resample('1d').mean()
cross_sectional_top_bottom_diff = rtn.abs().apply(top_5_minus_bottom_95, axis=1).resample('1d').mean()

    
# %%
title_text = compare_name

FONTSIZE_L1 = 25
FONTSIZE_L2 = 18
FONTSIZE_L3 = 15

fig, (ax0, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(36, 36), 
                                         gridspec_kw={'height_ratios': [4, 1, 1, 1]})

# Set the title with the specified font size
fig.suptitle(title_text, fontsize=FONTSIZE_L1, y=0.9)

# Loop through profd_dict and plot cumulative returns for each profd
combined_returns = pd.DataFrame()
for idx, (name, profd) in enumerate(profd_dict.items()):
    return_text = f"Return: {metrics[name]['return']:.2%}, Max DD: {metrics[name]['max_dd']:.2%}, Sharpe: {metrics[name]['sharpe_ratio']:.2f}"
    for twap_name in twap_list:
        combined_returns[f"{name}_{twap_name}"] = profd[f'raw_rtn_{twap_name}'] + profd['fee'] + profd['funding']
        
        ax0.plot((profd[f'raw_rtn_{twap_name}'] + profd['fee'] + profd['funding']).cumsum(), 
                 label=f"{name}: {return_text}", linewidth=3, color=plt.cm.Set1(idx))
        ax0.plot((profd['fee']).abs().cumsum(), label=f"{name}: fee", linewidth=3, color=plt.cm.Set1(idx), linestyle='--')
        ax0.plot((profd['funding']).abs().cumsum(), label=f"{name}: funding", linewidth=3, color=plt.cm.Set1(idx), linestyle=':')
# 计算均线
avg_return = combined_returns.mean(axis=1)
avg_metric = get_general_return_metrics(avg_return.values)
return_text = f"Return: {avg_metric['return']:.2%}, Max DD: {avg_metric['max_dd']:.2%}, Sharpe: {avg_metric['sharpe_ratio']:.2f}"
ax0.plot(avg_return.cumsum(), label=f"Average Cumulative Return: {return_text}", linewidth=3, color='black', linestyle='-')

ax0.grid(linestyle=":")
ax0.legend(loc="upper left", borderaxespad=0.5, fontsize=FONTSIZE_L2)
ax0.tick_params(labelsize=FONTSIZE_L2, pad=15)

# Plot cross-sectional volatility (occupying 1/6 of the height)
ax2.plot(cross_sectional_volatility.loc[profd.index], label='Cross-Sectional Volatility', color=plt.cm.Dark2(0))
ax2.grid(linestyle=":")
ax2.legend(loc="upper left", borderaxespad=0.5, fontsize=FONTSIZE_L2)
ax2.tick_params(labelsize=FONTSIZE_L2, pad=15)

# Plot cross-sectional kurtosis (occupying 1/6 of the height)
ax3.plot(cross_sectional_kurt.loc[profd.index], label='Cross-Sectional Kurtosis', color=plt.cm.Dark2(1))
ax3.grid(linestyle=":")
ax3.legend(loc="upper left", borderaxespad=0.5, fontsize=FONTSIZE_L2)
ax3.tick_params(labelsize=FONTSIZE_L2, pad=15)

# Plot cross-sectional top-bottom difference (occupying 1/6 of the height)
ax4.plot(cross_sectional_top_bottom_diff.loc[profd.index], label='Cross-Sectional Top-Bottom', color=plt.cm.Dark2(1))
ax4.grid(linestyle=":")
ax4.legend(loc="upper left", borderaxespad=0.5, fontsize=FONTSIZE_L2)
ax4.tick_params(labelsize=FONTSIZE_L2, pad=15)

plt.subplots_adjust(hspace=0.2)
plt.savefig(save_dir / f"{compare_name}_with_vol.jpg", dpi=100, bbox_inches="tight")
plt.show()