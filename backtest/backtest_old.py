# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 15:23:57 2024

@author: Xintang Zheng

"""
# %% imports
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from functools import partial
from datetime import timedelta


# %% add sys path
file_path = Path(__file__).resolve()
file_dir = file_path.parents[0]
project_dir = file_path.parents[1]
sys.path.append(str(project_dir))


# %% import self-defined
from utils.dirutils import load_path_config
from utils.datautils import align_columns, align_index_with_main
from portfolio_management import future_optimal_weight_lp_cvxpy
from utils.timeutils import DAY_SEC, DATA_FREQ
from test_and_eval.scores import get_general_return_metrics


# %%
twap_list = ['twd30_sp240', 'twd60_sp240', 'twd120_sp240', 'twd240_sp240'] #, 'twd30', 'twd60', 'twd120', 'twd240'
# test_name = 'ridge_v5_ma15_only_1'
# test_name = 'ridge_v5_lb_1y_alpha_1000'
# test_name = 'ridge_v9_1y' #_crct_240T
# test_name = 'ridge_v10_15T'
# test_name = 'ridge_v13_agg_240805_2_1y'
# test_name = 'esmb_objv1_agg_240805_2_withbe'
# test_name = "base_for_esmb_ridge_prcall"
test_name = 'merge_agg_240806_gp1_0'
# test_name = 'ridge_v13_agg_240805_lowhsr_1y'
backtest_name = 'to_003_maxmulti_2'  # mwgt_010 #_lqdt_500_mm_06
# backtest_name = 'to_040_mwgt_040'


# %%
sp = '240T'
to_rate_thresh_L0 = 0.03
to_rate_thresh_L1 = 1
to_rate_thresh_L2 = 1
to_rate_thresh_L3 = 1
to_rate_thresh_all = 1
max_multi = 2
max_wgt = None
tradv_thresh = 0 #5000000
momentum_limits = {} #{11: 0.06, 22: 0.06, 33: 0.06, 44: 0.06}
big_move_days = 10000
_lambda = 0.01
steepness = None
min_wgt = None

FEE = 0.0008


# %%
path_config = load_path_config(project_dir)
processed_data_dir = Path(path_config['processed_data'])
result_dir = Path(path_config['result']) / 'model'
twap_data_dir = Path(path_config['twap_price'])



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
    rtn_c2c = (close_price / close_price.shift(1) - 1).replace([np.inf, -np.inf], np.nan) #.fillna(0.0)
    rtn_cw0 = (twap_price / close_price.shift(1) - 1).replace([np.inf, -np.inf], np.nan) #.fillna(0.0)
    rtn_cw1 = (close_price / twap_price - 1).replace([np.inf, -np.inf], np.nan) #.fillna(0.0)
    if to_mask is None:
        to_mask = rtn_c2c.isna() | rtn_cw0.isna() | rtn_cw1.isna()
    else:
        to_mask = to_mask | rtn_c2c.isna() | rtn_cw0.isna() | rtn_cw1.isna()
    calc_pft_func_dict[twap_name] = partial(calc_profit_before_next_t,
                                            rtn_c2c=rtn_c2c.fillna(0.0), 
                                            rtn_cw0=rtn_cw0.fillna(0.0), 
                                            rtn_cw1=rtn_cw1.fillna(0.0))
del twap_dict

# tradv
tradv_path = processed_data_dir / 'ma1440_sp240' / 'tradv.parquet'
tradv = pd.read_parquet(tradv_path)
tradv = align_columns(main_columns, tradv)
tradv = align_index_with_main(main_index, tradv)


# momentum
mm_dict = {}
for mm_wd in momentum_limits:
    mm = (curr_price / curr_price.shift(mm_wd) - 1).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    mm_dict[mm_wd] = mm

# load predict
predict_dir = result_dir / test_name / 'predict'
try:
    predict_res_path = predict_dir / f'predict_{test_name}.parquet'
    predict_res = pd.read_parquet(predict_res_path)
except:
    predict_res_path = predict_dir / 'predict.parquet'
    predict_res = pd.read_parquet(predict_res_path)
# predict_res = predict_res.dropna(how='all')

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
opt_func = partial(future_optimal_weight_lp_cvxpy, max_multi=max_multi, max_wgt=max_wgt, 
                   tradv_thresh=tradv_thresh, momentum_limits=momentum_limits,
                   _lambda=_lambda, steepness=steepness, min_wgt=min_wgt)

last_big_move_t = factor.index[0]

for idx, t in enumerate(tqdm(factor.index, desc='simulating trades')):
    try:
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
            # if idx == 0:
            #     w1, status, max_wgt = opt_func(alpha, w0, if_to_trade, tradv_t, mm_t, to_rate_thresh_L1)
            # else:
            #     w1, status, max_wgt = opt_func(alpha, w0, if_to_trade, tradv_t, mm_t, to_rate_thresh_L0)
            w1, status, max_wgt = opt_func(alpha, w0, if_to_trade, tradv_t, mm_t, to_rate_thresh_L0)
        
            if status != "optimal":
                print(f"Optimal result not found at {t}!")
                w1, status, max_wgt = opt_func(alpha, w0, if_to_trade, tradv_t, mm_t, to_rate_thresh_L1)
                if status != "optimal":
                    w1, status, max_wgt = opt_func(alpha, w0, if_to_trade, tradv_t, mm_t, to_rate_thresh_L2)
                    if status != "optimal":
                        w1, status, max_wgt = opt_func(alpha, w0, if_to_trade, tradv_t, mm_t, to_rate_thresh_L3)
                        if status != "optimal":
                            print(f"Optimal result not found at {t}!")
                            w1 = w0.copy()           
        w_list.append(w1.copy())
        max_wgt_list.append(max_wgt)
        
        holding_pft_list = [calc_pft_func_dict[twap_name](w0, w1) for twap_name in twap_list]
        # print(np.sum(np.abs(w1 - w0)))
        # if np.sum(np.abs(w1 - w0)) > 0.35:
        #     breakpoint()
        trade_cost = - FEE * np.sum(np.abs(w1 - w0))
    
        profit_list.append([t, trade_cost] + holding_pft_list)
        
        w0 = w1.copy()
    except KeyError:
        break

# breakpoint()
profit = pd.DataFrame(profit_list, columns=['t', 'fee'] + [f'raw_rtn_{twap_name}' for twap_name in twap_list])
profit.set_index('t', inplace=True)
profit.to_csv(backtest_dir / f"profit_{name_to_save}.csv")

w = pd.DataFrame(w_list, index=factor.index, columns=factor.columns)
w[np.abs(w) < 1e-3] = 0
w.to_csv(backtest_dir / f"pos_{name_to_save}.csv")

long_num = w.gt(1e-3).sum(axis=1)
short_num = w.lt(-1e-3).sum(axis=1)
long_short_num = pd.DataFrame({
    'long_num': long_num,
    'short_num': short_num,
    })
long_short_num.to_csv(backtest_dir / f"long_short_num_{name_to_save}.csv")

try:
    max_wgt_df = pd.DataFrame(max_wgt_list, index=factor.index, columns=['max_wgt'])
    max_wgt_df.to_csv(backtest_dir / f"max_wgt_{name_to_save}.csv")
except:
    pass

profd = profit.resample('1d').sum()
profd['return'] = profd['raw_rtn_twd30_sp240'] + profd['fee']
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