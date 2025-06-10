# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 15:23:57 2024

@author: Xintang Zheng

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
from datetime import timedelta, datetime
import toml


# %% add sys path
file_path = Path(__file__).resolve()
file_dir = file_path.parents[0]
project_dir = file_path.parents[1]
sys.path.append(str(project_dir))


# %% import self-defined
from utils.dirutils import load_path_config
from utils.datautils import align_columns, align_index_with_main
from backtest.portfolio_management import future_optimal_weight_lp_cvxpy
from utils.timeutils import DAY_SEC, DATA_FREQ
from test_and_eval.scores import get_general_return_metrics
from data_processing.feature_engineering import normalization, quantile_transform_with_nan


# %%
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


def cumulative_volatility(returns):
    squared_returns = returns ** 2
    sum_squared = np.sum(squared_returns)
    return np.sqrt(sum_squared)


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
    
    
def pr_zscore(curr_price, n, m):
    # 计算滚动窗口内的最大值和最小值
    rolling_max = curr_price.rolling(window=n, min_periods=1).max()
    rolling_min = curr_price.rolling(window=n, min_periods=1).min()

    # 计算滚动价格变化范围（最大值 - 最小值）
    price_range = rolling_max - rolling_min

    # 计算滚动窗口内的均值和标准差
    rolling_mean = price_range.rolling(window=m, min_periods=1).mean()
    rolling_std = price_range.rolling(window=m, min_periods=1).std()

    # 计算Z-Score
    z_scores = (price_range - rolling_mean) / rolling_std
    return z_scores


def get_top_k_mask(z_scores, k):
    # Initialize mask with False values
    top_k_mask = pd.DataFrame(False, index=z_scores.index, columns=z_scores.columns)
    
    # For each row (timestamp), get the indices of the top K values
    for timestamp in z_scores.index:
        row = z_scores.loc[timestamp]
        
        # Skip if all values are NaN
        if row.isna().all():
            continue
        
        # Get indices of top K values, ignoring NaNs
        k_to_use = min(k, (~row.isna()).sum())  # Use at most k, limited by non-NaN values
        top_indices = row.nlargest(k_to_use).index
        
        # Set those indices to True for this timestamp
        top_k_mask.loc[timestamp, top_indices] = True
    
    return top_k_mask


# %%
def staggered_backtest(test_name, backtest_name):
    """
    Implements a staggered rebalancing strategy where:
    1. The data is divided into 8 channels starting at different 30-min intervals
    2. Each channel rebalances every 4 hours
    3. The positions from all 8 channels are combined to generate a final position
    4. Returns are calculated based on the combined position
    """
    path_config = load_path_config(project_dir)
    processed_data_dir = Path(path_config['factor_data'])
    param_dir = Path(path_config['param']) / 'backtest'
    result_dir = Path(path_config['result']) / 'model'
    twap_data_dir = Path(path_config['twap_price'])
    tradv_dir = Path(path_config['tradv'])
    funding_dir = Path(path_config['funding'])
    
    backtest_param = toml.load(param_dir / f'{backtest_name}.toml')
    backtest_param = {k: None if v == '' else v for k, v in backtest_param.items()}
    for dict_name in ('momentum_limits', 'pf_limits'):
        backtest_param[dict_name] = {int(k): v for k, v in backtest_param[dict_name].items()}
        
    sp = backtest_param['sp']
    twap_list = backtest_param['twap_list']
    to_rate_thresh_L0 = backtest_param['to_rate_thresh_L0']
    to_rate_thresh_L1 = backtest_param['to_rate_thresh_L1']
    to_rate_thresh_L2 = backtest_param['to_rate_thresh_L2']
    to_rate_thresh_L3 = backtest_param['to_rate_thresh_L3']
    to_rate_thresh_all = backtest_param['to_rate_thresh_all']
    max_multi = backtest_param['max_multi']
    max_wgt = backtest_param['max_wgt']
    tradv_rolling_d = backtest_param['tradv_rolling_d']
    tradv_thresh = backtest_param['tradv_thresh']
    tradv_dsct = backtest_param['tradv_dsct']
    momentum_limits = backtest_param['momentum_limits']
    pf_limits = backtest_param['pf_limits']
    big_move_days = backtest_param['big_move_days']
    _lambda = backtest_param['_lambda']
    steepness = backtest_param['steepness']
    min_wgt = backtest_param['min_wgt']
    outlier_n = backtest_param['outlier_n']
    quantile_transform = backtest_param['quantile_transform']
    rtn_vol_wd = backtest_param['rtn_vol_wd']
    rtn_vol_penalty_abs = backtest_param['rtn_vol_penalty_abs']
    rtn_vol_penalty_pct = backtest_param['rtn_vol_penalty_pct']
    rtn_vol_penalty_mul = backtest_param['rtn_vol_penalty_mul']
    FEE = backtest_param['FEE']
    symbols = backtest_param['symbols']
    mask_pr_zscore_params = backtest_param.get('mask_pr_zscore')
    mask_pr_zscore_smth_params = backtest_param.get('mask_pr_zscore_smth')
    funding_abs_limit = backtest_param.get('funding_abs_limit')
        
    backtest_dir = result_dir / test_name / 'backtest' / f"{backtest_name}_staggered_4h"
    backtest_dir.mkdir(parents=True, exist_ok=True)
    
    name_to_save = f'{test_name}__{backtest_name}_staggered_4h'
    
    # load twap & calc rtn
    curr_px_path = twap_data_dir / f'curr_price_sp30.parquet'
    curr_price = pd.read_parquet(curr_px_path)
    if symbols:
        curr_price = curr_price[symbols]
    curr_price = curr_price.resample(f'{sp}min').first()
    close_price = curr_price.shift(-1)
    main_columns = close_price.columns
    main_index = close_price.index
    
    twap_dict = {}                                               
    for twap_name in twap_list:
        twap_path = twap_data_dir / f'{twap_name}.parquet'
        twap_price = pd.read_parquet(twap_path)
        twap_price = align_columns(main_columns, twap_price)
        twap_price = align_index_with_main(main_index, twap_price)
        twap_dict[twap_name] = twap_price
        end_t = twap_price.index[-1]
    
    calc_pft_func_dict = {}
    to_mask = None
    for twap_name in twap_dict:
        twap_price = twap_dict[twap_name]
        rtn_c2c = (close_price / close_price.shift(1) - 1).replace([np.inf, -np.inf], np.nan)
        rtn_cw0 = (twap_price / close_price.shift(1) - 1).replace([np.inf, -np.inf], np.nan)
        rtn_cw1 = (close_price / twap_price - 1).replace([np.inf, -np.inf], np.nan)
        if to_mask is None:
            to_mask = rtn_c2c.isna() | rtn_cw0.isna() | rtn_cw1.isna()
        else:
            to_mask = to_mask | rtn_c2c.isna() | rtn_cw0.isna() | rtn_cw1.isna()
        calc_pft_func_dict[twap_name] = partial(calc_profit_before_next_t,
                                                rtn_c2c=rtn_c2c.fillna(0.0), 
                                                rtn_cw0=rtn_cw0.fillna(0.0), 
                                                rtn_cw1=rtn_cw1.fillna(0.0))
    del twap_dict
    
    # funding rate
    funding_path = funding_dir / 'funding_rates_wide.parquet'
    funding = pd.read_parquet(funding_path)
    funding.columns = funding.columns.str.lower()
    funding = align_columns(main_columns, funding)
    
    # tradv
    tradv_path = tradv_dir / 'trade_A_amount.parquet'
    tradv = pd.read_parquet(tradv_path)
    tradv = align_columns(main_columns, tradv)
    tradv = align_index_with_main(main_index, tradv)
    tradv_avg = tradv.rolling(int(tradv_rolling_d*24*60/sp)).mean()
    
    # momentum
    mm_dict = {}
    for mm_wd in momentum_limits:
        mm = (curr_price / curr_price.shift(int(mm_wd*240/sp)) - 1).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        mm_dict[mm_wd] = mm
        
    # vol
    rtn_1p = (curr_price / curr_price.shift(1) - 1).replace([np.inf, -np.inf], np.nan)
    rtn_vol = rtn_1p.rolling(window=rtn_vol_wd).apply(cumulative_volatility, raw=True)
    vol_rank = rtn_vol.rank(axis=1, pct=True)
    rtn_vol_penalty_abs = rtn_vol.stack().quantile(0.9)
    
    # mask_pr_zscore
    if mask_pr_zscore_params is not None:
        mask_pr_zscore_thres = mask_pr_zscore_params['mask_pr_zscore_thres']
        pr_wd = mask_pr_zscore_params['pr_wd']
        zscore_wd = mask_pr_zscore_params['zscore_wd']
        
        z_scores = pr_zscore(curr_price, pr_wd, zscore_wd)
        pr_zscore_mask = z_scores > mask_pr_zscore_thres
        to_mask = to_mask | pr_zscore_mask
        
    # mask_pr_zscore_smth
    if mask_pr_zscore_smth_params is not None:
        mask_pr_zscore_thres = mask_pr_zscore_smth_params['mask_pr_zscore_thres']
        cool_period = mask_pr_zscore_smth_params['cool_period']
        pr_wd = mask_pr_zscore_smth_params['pr_wd']
        zscore_wd = mask_pr_zscore_smth_params['zscore_wd']
        pr_zsc_top_k = mask_pr_zscore_smth_params.get('pr_zsc_top_k', 100)
        
        # Calculate z-scores
        z_scores = pr_zscore(curr_price, pr_wd, zscore_wd)
        
        # Identify points above threshold
        pr_zscore_above_thres = z_scores > mask_pr_zscore_thres
        
        # Get mask for top K coins with highest z-scores at each timestamp
        top_k_mask = get_top_k_mask(z_scores, k=pr_zsc_top_k)
        
        # Apply both filters BEFORE calculating the rolling window
        combined_filter = pr_zscore_above_thres & top_k_mask
        
        # Calculate rolling window on the pre-filtered data
        rolling_below_threshold = combined_filter.rolling(window=cool_period, min_periods=1).mean()
        final_mask = rolling_below_threshold != 0
        
        # Update your existing mask
        to_mask = to_mask | final_mask
        
    # funding
    if funding_abs_limit is not None:
        funding_cooldown = backtest_param.get('funding_cooldown')
        predict_funding_path = funding_dir / 'binance_usd_funding_rates_30min.parquet'
        predict_funding_rate = pd.read_parquet(predict_funding_path)
        
        funding_invalid = predict_funding_rate.abs() > funding_abs_limit
        
        if funding_cooldown is not None:
            rolling_below_threshold = funding_invalid.rolling(window=funding_cooldown, min_periods=1).mean()
            final_mask = rolling_below_threshold != 0
        else:
            final_mask = funding_invalid
        
        to_mask = to_mask | final_mask
    
    # load predict
    predict_dir = result_dir / test_name / 'predict'
    try:
        predict_res_path = predict_dir / f'predict_{test_name}.parquet'
        predict_res = pd.read_parquet(predict_res_path)
    except:
        predict_res_path = predict_dir / 'predict.parquet'
        predict_res = pd.read_parquet(predict_res_path)
    predict_res = predict_res[(predict_res != 0).any(axis=1)]
    
    predict_res = align_columns(main_columns, predict_res)
    predict_res = predict_res.resample(f'{sp}min').first()
    to_mask = to_mask | predict_res.isna()
    predict_res = predict_res.mask(to_mask)
    
    # rank
    factor_rank = predict_res.rank(axis=1, pct=True
                                  ).sub(0.5 / predict_res.count(axis=1), axis=0
                                       ).replace([np.inf, -np.inf], np.nan)
    fct_n_pct = 2 * (factor_rank - 0.5)
    
    factor = fct_n_pct.div(fct_n_pct.abs().sum(axis=1), axis=0).fillna(0)
    merge_and_save(factor, backtest_dir / f"factor_{name_to_save}.csv")
    
    # Define the 8 channels with 30-minute offsets between them
    # Each channel will rebalance every 4 hours (8 * 30 minutes)
    channels = {}
    first_timestamp = factor.index[0]
    
    # Create 8 channels with different starting offsets
    for channel_idx in range(8):
        channel_offset = pd.Timedelta(minutes=30 * channel_idx)
        channel_start = first_timestamp + channel_offset
        channel_timestamps = pd.date_range(
            start=channel_start,
            end=factor.index[-1],
            freq=pd.Timedelta(hours=4)
        )
        # Only keep timestamps that are in the factor index
        channels[channel_idx] = [ts for ts in channel_timestamps if ts in factor.index]
    
    # Dictionary to store each channel's positions over time
    channel_positions = {}
    
    # Run backtest for each channel
    for channel_idx, channel_timestamps in tqdm(channels.items(), desc='Processing channels'):
        # Initialize parameters for this channel
        opt_func = partial(future_optimal_weight_lp_cvxpy, max_multi=max_multi, max_wgt=max_wgt,
                          tradv_thresh=tradv_thresh, tradv_dsct=tradv_dsct,
                          rtn_vol_penalty_abs=rtn_vol_penalty_abs, rtn_vol_penalty_pct=rtn_vol_penalty_pct,
                          rtn_vol_penalty_mul=rtn_vol_penalty_mul,
                          momentum_limits=momentum_limits, pf_limits=pf_limits,
                          _lambda=_lambda, steepness=steepness, min_wgt=min_wgt)
        
        w0 = factor.iloc[0] / np.sum(np.abs(factor.iloc[0]))
        w_list = []
        t_list = []
        last_big_move_t = factor.index[0]
        
        # Create an empty dataframe to store positions
        channel_pos = pd.DataFrame(index=factor.index, columns=factor.columns)
        
        # Dictionary to store profits for each TWAP
        twap_profit_dict = {twap_name: {} for twap_name in twap_list}
        
        # Process each timestamp in this channel
        for t in channel_timestamps:
            if t >= end_t:
                break
                
            try:
                alpha = factor.loc[t].values
                if quantile_transform:
                    alpha = quantile_transform_with_nan(alpha)
                if_to_trade = ~to_mask.loc[t].values
                
                if not any(if_to_trade):
                    w1 = w0.copy()
                else:
                    tradv_t = tradv_avg.loc[t].values
                    mm_t = {mm_wd: mm.loc[t].values for mm_wd, mm in mm_dict.items()}
                    rtn_vol_t = rtn_vol.loc[t].values
                    vol_rank_t = vol_rank.loc[t].values
                    
                    # Calculate historical profit (simplified for this channel-based approach)
                    # In a real implementation, you'd need to properly calculate historical profits
                    # based on this channel's previous positions
                    his_pft_t = {pf_wd: np.zeros_like(w0) for pf_wd in pf_limits}
                    
                    if t - last_big_move_t > timedelta(days=big_move_days):
                        w1 = alpha / np.sum(np.abs(alpha))
                        last_big_move_t = t
                    else:
                        w1, status, _ = opt_func(alpha, w0, if_to_trade, tradv_t, mm_t, his_pft_t, 
                                               rtn_vol_t, vol_rank_t, to_rate_thresh_L0)
                    
                        if status != "optimal":
                            w1, status, _ = opt_func(alpha, w0, if_to_trade, tradv_t, mm_t, his_pft_t, 
                                                   rtn_vol_t, vol_rank_t, to_rate_thresh_L1)
                            if status != "optimal":
                                w1, status, _ = opt_func(alpha, w0, if_to_trade, tradv_t, mm_t, his_pft_t, 
                                                       rtn_vol_t, vol_rank_t, to_rate_thresh_L2)
                                if status != "optimal":
                                    w1, status, _ = opt_func(alpha, w0, if_to_trade, tradv_t, mm_t, his_pft_t, 
                                                           rtn_vol_t, vol_rank_t, to_rate_thresh_L3)
                                    if status != "optimal":
                                        print(f"Channel {channel_idx}: Optimal result not found at {t}!")
                                        w1 = w0.copy()
                
                # Store the position
                channel_pos.loc[t] = w1
                w0 = w1.copy()
                
            except KeyError:
                print(f"KeyError at time {t} for channel {channel_idx}")
                continue
                
        # Forward fill positions for all timestamps
        channel_pos = channel_pos.fillna(method='ffill')
        channel_positions[channel_idx] = channel_pos
    
    # Combine positions from all channels
    # First, normalize the weight of each channel (dividing each position by 8)
    for channel_idx in channel_positions:
        channel_positions[channel_idx] = channel_positions[channel_idx] / 8
    
    # Next, combine all channels' positions
    combined_positions = pd.DataFrame(0, index=factor.index, columns=factor.columns)
    for channel_idx in channel_positions:
        # Add each channel's position to the combined positions
        valid_indices = combined_positions.index.intersection(channel_positions[channel_idx].index)
        combined_positions.loc[valid_indices] += channel_positions[channel_idx].loc[valid_indices]
    
    # Normalize the combined positions
    position_scale = combined_positions.abs().sum(axis=1)
    normalized_positions = combined_positions.div(position_scale, axis=0).fillna(0)
    
    # Save the combined positions
    merge_and_save(normalized_positions, backtest_dir / f"pos_{name_to_save}.parquet")
    
    # Calculate profits based on the combined positions
    w_list = normalized_positions.values
    t_list = normalized_positions.index
    
    profit_list = []
    
    # Initialize dictionaries to store individual asset profits for each TWAP
    twap_profit_dict = {twap_name: pd.DataFrame(index=t_list, columns=factor.columns) for twap_name in twap_list}
    
    # Calculate profits for each consecutive pair of timestamps
    for idx in tqdm(range(1, len(t_list)), desc='Calculating profits'):
        t_prev = t_list[idx-1]
        t_curr = t_list[idx]
        w0 = normalized_positions.loc[t_prev].values
        w1 = normalized_positions.loc[t_curr].values
        
        # Calculate holding profits for each TWAP
        holding_pft_list = [calc_pft_func_dict[twap_name](t_curr, w0, w1) for twap_name in twap_list]
        
        # Calculate trade costs
        trade_cost = -FEE * np.sum(np.abs(w1 - w0))
        
        # Calculate funding costs
        try:
            funding_t = funding.loc[t_prev].fillna(0).values
            funding_cost = -np.sum(funding_t * w0)
        except:
            funding_cost = 0
        
        # Store profits for each TWAP
        total_profit_list = []
        for twap_name, profit in zip(twap_list, holding_pft_list):
            twap_profit_dict[twap_name].loc[t_prev] = profit
            total_profit = np.sum(profit)
            total_profit_list.append(total_profit)
        
        profit_list.append([t_prev, trade_cost, funding_cost] + total_profit_list)
    
    # Create profit DataFrame
    profit = pd.DataFrame(profit_list, columns=['t', 'fee', 'funding'] + [f'raw_rtn_{twap_name}' for twap_name in twap_list])
    profit.set_index('t', inplace=True)
    merge_and_save(profit, backtest_dir / f"profit_{name_to_save}.parquet")
    
    # Save individual asset profits for each TWAP
    for twap_name, df in twap_profit_dict.items():
        merge_and_save(df, backtest_dir / f"profit_{twap_name}_{name_to_save}.parquet")
        cs_vol = pd.DataFrame(df.std(axis=1), columns=['cs_vol'])
        merge_and_save(cs_vol, backtest_dir / f"cs_vol_{twap_name}_{name_to_save}.parquet")
    
    # Calculate long/short counts
    long_num = normalized_positions.gt(1e-3).sum(axis=1)
    short_num = normalized_positions.lt(-1e-3).sum(axis=1)
    long_short_num = pd.DataFrame({
        'long_num': long_num,
        'short_num': short_num,
    })
    merge_and_save(long_short_num, backtest_dir / f"long_short_num_{name_to_save}.parquet")
    
    # Plot results
    profit = pd.read_parquet(backtest_dir / f"profit_{name_to_save}.parquet")
    long_short_num = pd.read_parquet(backtest_dir / f"long_short_num_{name_to_save}.parquet")
    
    profd = profit.resample('1d').sum()
    profd['return'] = profd[f'raw_rtn_{twap_list[0]}'] + profd['fee'] + profd['funding']
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
    ax0.plot((profd['return']).cumsum(), label='combined_return', linewidth=3)
    ax0.plot((profd['fee']).abs().cumsum(), label='fee', color='r', linewidth=3)
    ax0.plot((- profd['funding']).cumsum(), label='funding', color='purple', linewidth=3)
    
    ax1 = fig.add_subplot(spec[2, :])
    ax1.plot(long_short_num['long_num'], label='long_num', color='r')
    ax1.plot(long_short_num['short_num'], label='short_num', color='g')
    
    for ax in [ax0, ax1]:
        ax.grid(linestyle=":")
        ax.legend(loc="upper left", borderaxespad=0.5, fontsize=FONTSIZE_L3)
        ax.tick_params(labelsize=FONTSIZE_L2, pad=15)
    
    plt.savefig(backtest_dir / f"{name_to_save}.jpg", dpi=100, bbox_inches="tight")
    plt.close()
    
    # Create comparison plot with original strategy if available
    try:
        original_backtest_dir = result_dir / test_name / 'backtest' / backtest_name
        original_profit_path = original_backtest_dir / f"profit_{test_name}__{backtest_name}.parquet"
        original_profit = pd.read_parquet(original_profit_path)
        
        original_profd = original_profit.resample('1d').sum()
        original_profd['return'] = original_profd[f'raw_rtn_{twap_list[0]}'] + original_profd['fee'] + original_profd['funding']
        original_metrics = get_general_return_metrics(original_profd['return'].values)
        
        # Plot comparison
        fig = plt.figure(figsize=(36, 18), dpi=100, layout="constrained")
        ax = fig.add_subplot(111)
        
        # Plot both strategies
        ax.plot((profd['return']).cumsum(), label='Staggered 4h Strategy', linewidth=3)
        ax.plot((original_profd['return']).cumsum(), label='Original Strategy', linewidth=3, linestyle='--')
        
        ax.set_title("Performance Comparison: Original vs Staggered 4h Strategy", fontsize=FONTSIZE_L1, pad=25)
        ax.grid(linestyle=":")
        ax.legend(loc="upper left", borderaxespad=0.5, fontsize=FONTSIZE_L3)
        ax.tick_params(labelsize=FONTSIZE_L2, pad=15)
        
        plt.savefig(backtest_dir / f"{name_to_save}_comparison.jpg", dpi=100, bbox_inches="tight")
        plt.close()
    except Exception as e:
        print(f"Could not create comparison plot: {e}")
        
    print(f"Staggered backtest completed. Results saved to {backtest_dir}")
    return metrics
    
    
# %%
if __name__=='__main__':
    test_name = 'merge_agg_250318_double3m_15d_73'
    # backtest_name = 'to_00125_maxmulti_2_mm_03_pf_001_test'  # _mm_03_pf_001_tradv_5_0_volpen_abs_09
    backtest_name = 'staggered_to_01_maxmulti_2'
    staggered_backtest(test_name, backtest_name)