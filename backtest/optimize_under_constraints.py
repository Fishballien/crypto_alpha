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


from dirs import TWAP_PRICE_DIR, RESULT_DIR, PROCESSED_DATA_DIR
from datautils import align_columns, align_index_with_main
from portfolio_management import optimize_with_constraints_momentum
from timeutils import DAY_SEC, DATA_FREQ
from factor_tester_fix_sp import FactorTest


# %%
twap_data_dir = TWAP_PRICE_DIR
result_dir = RESULT_DIR
processed_data_dir = PROCESSED_DATA_DIR
twap_name = 'twd15_sp240'
test_name = 'ridge_v5_lb_1y_alpha_1000'
constraint_name = 'momentum'


# %% opt params
max_multi = 0.1
max_wgt = None
epsilon = 0
momentum_limits = {11: 0.06, 22: 0.06, 33: 0.06, 44: 0.06} #{11: 0.06, 22: 0.06, 33: 0.06, 44: 0.06}


# %% test params
sp = '240T'
outlier_n = 30
pp_list = [15, 30, 60, 120, 240, 360, 720]
lag_list = list(range(4))
bin_step = 0.1


params = {
    'sp': sp,
    'outlier_n': outlier_n,
    'pp_list': pp_list,
    'lag_list': lag_list,
    'bin_step': bin_step,
}


# %%
predict_dir = result_dir / test_name / 'predict'
res_dir = result_dir / test_name / f'with_constraints_{constraint_name}'
res_dir.mkdir(parents=True, exist_ok=True)


# %%
name_to_save = f'predict_with_constraints_{constraint_name}'
# load twap & calc rtn
curr_px_path = twap_data_dir / 'curr_price_sp240.parquet'
curr_price = pd.read_parquet(curr_px_path)
close_price = curr_price.shift(-1)
main_columns = close_price.columns
# del curr_price

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
to_mask = predict_res.isna()
predict_res = predict_res.mask(to_mask)
factor_rank = predict_res.rank(axis=1, pct=True
                               ).sub(0.5 / predict_res.count(axis=1), axis=0
                                     ).replace([np.inf, -np.inf], np.nan
                                               ).fillna(0)
fct_n_pct = 2 * (factor_rank - 0.5)
factor = fct_n_pct.div(fct_n_pct.abs().sum(axis=1), axis=0).fillna(0)


# initialize
new_factor = pd.DataFrame(columns=factor.columns, index=factor.index)
    
opt_func = partial(optimize_with_constraints_momentum, momentum_limits=momentum_limits, 
                   max_multi=max_multi, max_wgt=max_wgt, epsilon=epsilon)


for t in tqdm(factor.index, desc='optimizing'):
    alpha = factor.loc[t].values
    if_to_trade = ~to_mask.loc[t].values
    mm_t = {mm_wd: mm.loc[t].values for mm_wd, mm in mm_dict.items()}
    
    try:
        new_alpha, status = opt_func(alpha, if_to_trade, mm_t)
        new_factor.loc[t, :] = new_alpha
        if status != 'optimal':
            print(f"Optimal result not found at {t}!")
    except:
        pass

new_factor = new_factor.mask(to_mask)
new_factor.to_parquet(res_dir / f'{name_to_save}.parquet')

process_name = None
factor_data_dir = res_dir
result_dir = res_dir

ft = FactorTest(process_name, twap_name, factor_data_dir, twap_data_dir, result_dir, params)
ft.test_one_factor(name_to_save)