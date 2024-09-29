# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 15:23:57 2024

@author: Xintang Zheng

"""
# %% imports
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from functools import partial
from datetime import timedelta


from dirs import TWAP_PRICE_DIR, RESULT_DIR, PROCESSED_DATA_DIR
from datautils import align_columns, align_index_with_main
from portfolio_management import future_optimal_weight_with_to_rate_opt
from timeutils import DAY_SEC, DATA_FREQ


# %%
twap_data_dir = TWAP_PRICE_DIR
result_dir = RESULT_DIR
processed_data_dir = PROCESSED_DATA_DIR
twap_list = ['twd15_sp240'] #, 'twd30', 'twd60', 'twd120', 'twd240'
# test_name = 'ridge_v5_ma15_only_1'
# test_name = 'ridge_v5_lb_1y_alpha_1000'
test_name = 'ridge_v9_1y'
# test_name = 'ridge_v10_15T'
backtest_name = 'to_rate_opt_mwgt_025_1_diff_00015'  # mwgt_010 #_lqdt_500_mm_06


# %%
sp = '240T'
to_rate_params = (0.02, 0.5, 0.01)
max_multi = None
max_wgt = 0.025
tradv_thresh = 0 #5000000
momentum_limits = {} #{11: 0.06, 22: 0.06, 33: 0.06, 44: 0.06}
big_move_days = 10000
_lambda = 0.01
FEE = 0.0005


# %%
backtest_dir = result_dir / test_name / 'backtest' / backtest_name
backtest_dir.mkdir(parents=True, exist_ok=True)


# %%
def calc_profit_before_next_t(w0, w1, rtn_c2c, rtn_cw0, rtn_cw1):
    w01 = np.zeros(w0.shape)
    w11 = np.zeros(w1.shape)
    w01[(w0 >= 0) & (w1 >= 0)] = w0[(w0 >= 0) & (w1 >= 0)]
    w11[(w0 >= 0) & (w1 >= 0)] = w1[(w0 >= 0) & (w1 >= 0)]
    cw = w01 - w11
    cw0 = np.zeros(cw.shape)
    cw0[cw > 0] = cw[cw > 0]
    cw1 = np.zeros(cw.shape)
    cw1[cw < 0] = - cw[cw < 0]
    hold_pft1 = rtn_c2c.loc[t].values.dot(w01 - cw0) + rtn_cw0.loc[t].values.dot(cw0) + rtn_cw1.loc[t].values.dot(cw1)

    w02 = np.zeros(w0.shape)
    w12 = np.zeros(w1.shape)
    w02[(w0 < 0) & (w1 < 0)] = w0[(w0 < 0) & (w1 < 0)]
    w12[(w0 < 0) & (w1 < 0)] = w1[(w0 < 0) & (w1 < 0)]
    cw = w02 - w12
    cw0 = np.zeros(cw.shape)
    cw0[cw > 0] = - cw[cw > 0]
    cw1 = np.zeros(cw.shape)
    cw1[cw < 0] = cw[cw < 0]
    hold_pft2 = rtn_c2c.loc[t].values.dot(w02 - cw1) + rtn_cw1.loc[t].values.dot(cw0) + rtn_cw0.loc[t].values.dot(cw1)

    w03 = np.zeros(w0.shape)
    w13 = np.zeros(w1.shape)
    w03[(w0 < 0) & (w1 >= 0)] = w0[(w0 < 0) & (w1 >= 0)]
    w13[(w0 < 0) & (w1 >= 0)] = w1[(w0 < 0) & (w1 >= 0)]
    hold_pft3 = rtn_cw0.loc[t].values.dot(w03) + rtn_cw1.loc[t].values.dot(w13)

    w04 = np.zeros(w0.shape)
    w14 = np.zeros(w1.shape)
    w04[(w0 >= 0) & (w1 < 0)] = w0[(w0 >= 0) & (w1 < 0)]
    w14[(w0 >= 0) & (w1 < 0)] = w1[(w0 >= 0) & (w1 < 0)]
    hold_pft4 = rtn_cw0.loc[t].values.dot(w04) + rtn_cw1.loc[t].values.dot(w14)

    hold_pft = hold_pft1 + hold_pft2 + hold_pft3 + hold_pft4
    return hold_pft


# %%
name_to_save = f'{test_name}__{backtest_name}'
# load twap & calc rtn
curr_px_path = twap_data_dir / 'curr_price_sp240.parquet'
curr_price = pd.read_parquet(curr_px_path)
close_price = curr_price.shift(-1)
main_columns = close_price.columns
# del curr_price

twap_dict = {}
for twap_name in twap_list:
    twap_path = twap_data_dir / f'{twap_name}.parquet'
    twap_price = pd.read_parquet(twap_path)
    twap_price = align_columns(main_columns, twap_price)
    twap_dict[twap_name] = twap_price

main_index = close_price.index

calc_pft_func_dict = {}
to_mask = None
for twap_name in twap_dict:
    twap_price = twap_dict[twap_name]
    rtn_c2c = (close_price / close_price.shift(1) - 1).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    rtn_cw0 = (twap_price / close_price.shift(1) - 1).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    rtn_cw1 = (close_price / twap_price - 1).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    if to_mask is None:
        to_mask = rtn_c2c.isna() | rtn_cw0.isna() | rtn_cw1.isna()
    else:
        to_mask = to_mask | rtn_c2c.isna() | rtn_cw0.isna() | rtn_cw1.isna()
    calc_pft_func_dict[twap_name] = partial(calc_profit_before_next_t,
                                            rtn_c2c=rtn_c2c, 
                                            rtn_cw0=rtn_cw0, 
                                            rtn_cw1=rtn_cw1)
del twap_dict

# tradv
tradv_path = processed_data_dir / 'ma1440_sp240' / 'tradv.parquet'
tradv = pd.read_parquet(tradv_path)
tradv = align_columns(main_columns, tradv)


# momentum
mm_dict = {}
for mm_wd in momentum_limits:
    mm = (curr_price / curr_price.shift(mm_wd) - 1).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    mm_dict[mm_wd] = mm

# load predict
predict_dir = result_dir / test_name / 'predict'
predict_res_path = predict_dir / 'predict.parquet'
predict_res = pd.read_parquet(predict_res_path)

predict_res = align_columns(main_columns, predict_res)
to_mask = to_mask | predict_res.isna()
predict_res = predict_res.mask(to_mask)
factor_rank = predict_res.rank(axis=1, pct=True
                               ).sub(0.5 / predict_res.count(axis=1), axis=0
                                     ).replace([np.inf, -np.inf], np.nan
                                               )
fct_n_pct = 2 * (factor_rank - 0.5)
factor = fct_n_pct.div(fct_n_pct.abs().sum(axis=1), axis=0).fillna(0)


# initialize
w_list = []
profit_list = []
max_wgt_list = []
    
# w0 = np.zeros(factor.shape[1])
w0 = factor.iloc[0] / np.sum(np.abs(factor.iloc[0]))
opt_func = partial(future_optimal_weight_with_to_rate_opt, max_multi=max_multi, max_wgt=max_wgt, to_rate_params=to_rate_params,
                   tradv_thresh=tradv_thresh, momentum_limits=momentum_limits,
                   _lambda=_lambda, backtest_dir=backtest_dir)

last_big_move_t = factor.index[0]

for t in tqdm(factor.index, desc='simulating trades'):
    alpha = factor.loc[t].values
    if_to_trade = ~to_mask.loc[t].values
    if not any(if_to_trade):
        w1 = w0.copy()
        w_list.append(w1.copy())
        max_wgt_list.append(max_wgt)
        holding_pft_list = [calc_pft_func_dict[twap_name](w0, w1) for twap_name in twap_list]
        trade_cost = - FEE * np.sum(np.abs(w1 - w0))
        profit_list.append([t, trade_cost] + holding_pft_list)
        continue
    tradv_t = tradv.loc[t].values * DAY_SEC / DATA_FREQ
    mm_t = {mm_wd: mm.loc[t].values for mm_wd, mm in mm_dict.items()}
    
    if t - last_big_move_t > timedelta(days=big_move_days):
        w1 = alpha / np.sum(np.abs(alpha))
        last_big_move_t = t
    else:
        w1, status, max_wgt = opt_func(t, alpha, w0, if_to_trade, tradv_t, mm_t)

    w_list.append(w1.copy())
    max_wgt_list.append(max_wgt)
    
    holding_pft_list = [calc_pft_func_dict[twap_name](w0, w1) for twap_name in twap_list]
    trade_cost = - FEE * np.sum(np.abs(w1 - w0))

    profit_list.append([t, trade_cost] + holding_pft_list)
    
    w0 = w1.copy()
    
profit = pd.DataFrame(profit_list, columns=['t', 'fee'] + [f'raw_rtn_{twap_name}' for twap_name in twap_list])
profit.set_index('t', inplace=True)
profit.to_csv(backtest_dir / f"profit_{name_to_save}.csv")

w = pd.DataFrame(w_list, index=factor.index, columns=factor.columns)
long_num = w.gt(1e-3).sum(axis=1)
short_num = w.lt(-1e-3).sum(axis=1)
w.to_csv(backtest_dir / f"pos_{name_to_save}.csv")

max_wgt_df = pd.DataFrame(max_wgt_list, index=factor.index, columns=['max_wgt'])
max_wgt_df.to_csv(backtest_dir / f"max_wgt_{name_to_save}.csv")

FONTSIZE_L1 = 20
FONTSIZE_L2 = 18
FONTSIZE_L3 = 15

fig = plt.figure(figsize=(36, 27), dpi=100, layout="constrained")
spec = fig.add_gridspec(ncols=1, nrows=3)

ax0 = fig.add_subplot(spec[:2, :])
ax0.set_title(f'{name_to_save}', fontsize=FONTSIZE_L1, pad=25)
for twap_name in twap_list:
    ax0.plot((profit[f'raw_rtn_{twap_name}'] + profit['fee']).cumsum(), label=f'rtn_{twap_name}', linewidth=3)
ax0.plot((profit['fee']).abs().cumsum(), label='fee', color='r', linewidth=3)

ax1 = fig.add_subplot(spec[2, :])
ax1.plot(long_num, label='long_num', color='r')
ax1.plot(short_num, label='short_num', color='g')

for ax in [ax0, ax1]:
    ax.grid(linestyle=":")
    ax.legend(loc="upper left", borderaxespad=0.5, fontsize=FONTSIZE_L3)
    ax.tick_params(labelsize=FONTSIZE_L2, pad=15)

plt.savefig(backtest_dir / f"{name_to_save}.jpg", dpi=100, bbox_inches="tight")
plt.close()