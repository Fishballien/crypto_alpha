# -*- coding: utf-8 -*-
"""
Created on Sat Feb 22 21:56:29 2025

@author: Xintang Zheng

星星: ★ ☆ ✪ ✩ 🌟 ⭐ ✨ 🌠 💫 ⭐️
勾勾叉叉: ✓ ✔ ✕ ✖ ✅ ❎
报警啦: ⚠ ⓘ ℹ ☣
箭头: ➔ ➜ ➙ ➤ ➥ ↩ ↪
emoji: 🔔 ⏳ ⏰ 🔒 🔓 🛑 🚫 ❗ ❓ ❌ ⭕ 🚀 🔥 💧 💡 🎵 🎶 🧭 📅 🤔 🧮 🔢 📊 📈 📉 🧠 📝

"""
# %% imports
import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from functools import partial


# %% add sys path
file_path = Path(__file__).resolve()
file_dir = file_path.parents[0]
project_dir = file_path.parents[1]
sys.path.append(str(project_dir))


# %% import self-defined
from test_and_eval.scores import get_general_return_metrics


# %%
def align_columns(main_col, sub_df):
    sub_aligned = sub_df.reindex(columns=main_col)
    return sub_aligned


def align_index_with_main(main_index, sub_df):
    # 使用reindex直接对齐索引
    sub_aligned = sub_df.reindex(index=main_index)
    return sub_aligned


def calc_profit_before_next_t(t, w0, w1, rtn_c2c, rtn_cw0, rtn_cw1):
    n_assets = w0.shape[0]
    
    # Initialize profit arrays
    hold_pft1 = np.zeros(n_assets)
    hold_pft2 = np.zeros(n_assets)
    hold_pft3 = np.zeros(n_assets)
    hold_pft4 = np.zeros(n_assets)

    # Case 1: w0 >= 0 & w1 >= 0
    w01 = np.where((w0 >= 0) & (w1 >= 0), w0, 0)
    w11 = np.where((w0 >= 0) & (w1 >= 0), w1, 0)
    cw = w01 - w11
    cw0 = np.where(cw > 0, cw, 0)
    cw1 = np.where(cw < 0, -cw, 0)
    hold_pft1 = rtn_c2c.loc[t].values * (w01 - cw0) + rtn_cw0.loc[t].values * cw0 + rtn_cw1.loc[t].values * cw1

    # Case 2: w0 < 0 & w1 < 0
    w02 = np.where((w0 < 0) & (w1 < 0), w0, 0)
    w12 = np.where((w0 < 0) & (w1 < 0), w1, 0)
    cw = w02 - w12
    cw0 = np.where(cw > 0, -cw, 0)
    cw1 = np.where(cw < 0, cw, 0)
    hold_pft2 = rtn_c2c.loc[t].values * (w02 - cw1) + rtn_cw1.loc[t].values * cw0 + rtn_cw0.loc[t].values * cw1

    # Case 3: w0 < 0 & w1 >= 0
    w03 = np.where((w0 < 0) & (w1 >= 0), w0, 0)
    w13 = np.where((w0 < 0) & (w1 >= 0), w1, 0)
    hold_pft3 = rtn_cw0.loc[t].values * w03 + rtn_cw1.loc[t].values * w13

    # Case 4: w0 >= 0 & w1 < 0
    w04 = np.where((w0 >= 0) & (w1 < 0), w0, 0)
    w14 = np.where((w0 >= 0) & (w1 < 0), w1, 0)
    hold_pft4 = rtn_cw0.loc[t].values * w04 + rtn_cw1.loc[t].values * w14

    # Sum up all holding profits
    hold_pft = hold_pft1 + hold_pft2 + hold_pft3 + hold_pft4
    
    return hold_pft


def merge_and_save(new_data, save_path):
    if os.path.exists(save_path):
        pre_data = pd.read_parquet(save_path)
        
        new_data = new_data[~new_data.index.isin(pre_data.index)]
        
        # 如果有新数据才进行合并和保存
        if not new_data.empty:
            # 合并并对索引进行排序
            data = pd.concat([pre_data, new_data]).sort_index()
            
        else:
            data = pre_data
            
        data = data[~data.index.duplicated(keep='first')] # 去重
    else:
        data = new_data
    data.to_parquet(save_path)


# %% 参数
twap_list = ['twd30_sp30']    
sp = 30
FEE = 0.00075


# %% 需要自定义的部分
twap_data_dir = Path() # 自定义twap路径
w = pd.DataFrame() # 即仓位文件，读出即可


# %% 数据准备
# load twap & calc rtn
curr_px_path = twap_data_dir / 'curr_price_sp30.parquet' # !!!
curr_price = pd.read_parquet(curr_px_path) #.loc['20230701':]
curr_price = curr_price.resample(f'{sp}min').first()
close_price = curr_price.shift(-1)
main_columns = close_price.columns
main_index = close_price.index
# del curr_price


twap_dict = {}                                               
for twap_name in twap_list:
    twap_path = twap_data_dir / f'{twap_name}.parquet'
    twap_price = pd.read_parquet(twap_path) #.loc['20230701':]
    twap_price = align_columns(main_columns, twap_price)
    twap_price = align_index_with_main(main_index, twap_price)
    twap_dict[twap_name] = twap_price
    end_t = twap_price.index[-1]


calc_pft_func_dict = {}
to_mask = None
for twap_name in twap_dict:
    twap_price = twap_dict[twap_name]
    rtn_c2c = (close_price / close_price.shift(1) - 1).replace([np.inf, -np.inf], np.nan) #.fillna(0.0)
    rtn_cw0 = (twap_price / close_price.shift(1) - 1).replace([np.inf, -np.inf], np.nan) #.fillna(0.0)
    rtn_cw1 = (close_price / twap_price - 1).replace([np.inf, -np.inf], np.nan) #.fillna(0.0)
    # breakpoint()
    if to_mask is None:
        to_mask = rtn_c2c.isna() | rtn_cw0.isna() | rtn_cw1.isna()
    else:
        to_mask = to_mask | rtn_c2c.isna() | rtn_cw0.isna() | rtn_cw1.isna()
    calc_pft_func_dict[twap_name] = partial(calc_profit_before_next_t,
                                            rtn_c2c=rtn_c2c.fillna(0.0), 
                                            rtn_cw0=rtn_cw0.fillna(0.0), 
                                            rtn_cw1=rtn_cw1.fillna(0.0))

w = align_columns(main_columns, w)    


# %% 计算twap收益
# initialize
t_list = []
profit_list = []
w0 = np.zeros(len(w.columns))
twap_profit_dict = {twap_name: pd.DataFrame(index=w.index, columns=w.columns) for twap_name in twap_list}

for idx, t in enumerate(tqdm(w.index, desc='simulating trades')):  # 遍历已有的仓位 w
    try:
        # 直接使用已有的仓位 w
        w1 = w.loc[t].values  # 取时间 t 的仓位数据

        holding_pft_list = [calc_pft_func_dict[twap_name](t, w0, w1) for twap_name in twap_list]
        trade_cost = - FEE * np.sum(np.abs(w1 - w0))

        total_profit_list = []
        for twap_name, profit in zip(twap_list, holding_pft_list):
            twap_profit_dict[twap_name].loc[t] = profit
            total_profit = np.sum(profit)
            total_profit_list.append(total_profit)

        profit_list.append([t, trade_cost] + total_profit_list)

        w0 = w1.copy()

        t_list.append(t)
    except KeyError:
        break


# %% Save
'''
merge_and_save主要应对已有历史文件情况，改写成直接保存即可
'''
# Create DataFrame for profits and save
profit = pd.DataFrame(profit_list, columns=['t', 'fee'] + [f'raw_rtn_{twap_name}' for twap_name in twap_list])
profit.set_index('t', inplace=True)
merge_and_save(profit, backtest_dir / f"profit_{name_to_save}.parquet")

# Save positions DataFrame
w = pd.DataFrame(w_list, index=t_list, columns=factor.columns)
merge_and_save(w, backtest_dir / f"pos_{name_to_save}.parquet")

# Save profit data for each TWAP
for twap_name, df in twap_profit_dict.items():
    merge_and_save(df, backtest_dir / f"profit_{twap_name}_{name_to_save}.parquet")
    cs_vol = pd.DataFrame(df.std(axis=1), columns=['cs_vol'])
    merge_and_save(cs_vol, backtest_dir / f"cs_vol_{twap_name}_{name_to_save}.parquet")

# Calculate long and short positions counts
long_num = w.gt(1e-3).sum(axis=1)
short_num = w.lt(-1e-3).sum(axis=1)
long_short_num = pd.DataFrame({
    'long_num': long_num,
    'short_num': short_num,
})
merge_and_save(long_short_num, backtest_dir / f"long_short_num_{name_to_save}.parquet")


# %% Plotting
profit = pd.read_parquet(backtest_dir / f"profit_{name_to_save}.parquet")
long_short_num = pd.read_parquet(backtest_dir / f"long_short_num_{name_to_save}.parquet")

profd = profit.resample('1d').sum()
profd['return'] = profd[f'raw_rtn_twd30_sp30'] + profd['fee']
metrics = get_general_return_metrics(profd['return'].values)

title_text = (f"{name_to_save}\n\n"
              f"Return: {metrics['return']:.2%}, Annualized Return: {metrics['return_annualized']:.2%}, "
              f"Max Drawdown: {metrics['max_dd']:.2%}, Sharpe Ratio: {metrics['sharpe_ratio']:.2f}\n"
              f"Calmar Ratio: {metrics['calmar_ratio']:.2f}, Sortino Ratio: {metrics['sortino_ratio']:.2f}, "
              f"Sterling Ratio: {metrics['sterling_ratio']:.2f}, Burke Ratio: {metrics['burke_ratio']:.2f}\n"
              f"Ulcer Index: {metrics['ulcer_index']:.2f}, Drawdown Recovery Ratio: {metrics['drawdown_recovery_ratio']:.2f}")

FONTSIZE_L1 = 25
FONTSIZE_L2 = 18
FONTSIZE_L3 = 15

fig = plt.figure(figsize=(36, 27), dpi=100, layout="constrained")
spec = fig.add_gridspec(ncols=1, nrows=3)

ax0 = fig.add_subplot(spec[:2, :])
ax0.set_title(title_text, fontsize=FONTSIZE_L1, pad=25)
for twap_name in twap_list:
    ax0.plot((profd['return']).cumsum(), label=f'rtn_{twap_name}', linewidth=3)
ax0.plot((profd['fee']).abs().cumsum(), label='fee', color='r', linewidth=3)

ax1 = fig.add_subplot(spec[2, :])
ax1.plot(long_short_num['long_num'], label='long_num', color='r')
ax1.plot(long_short_num['short_num'], label='short_num', color='g')

for ax in [ax0, ax1]:
    ax.grid(linestyle=":")
    ax.legend(loc="upper left", borderaxespad=0.5, fontsize=FONTSIZE_L3)
    ax.tick_params(labelsize=FONTSIZE_L2, pad=15)

plt.savefig(backtest_dir / f"{name_to_save}.jpg", dpi=100, bbox_inches="tight")
plt.close()
