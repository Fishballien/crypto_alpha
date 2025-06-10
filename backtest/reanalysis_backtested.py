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
import warnings
warnings.filterwarnings("ignore")


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
from analysisutils import top_5_minus_bottom_95


# %%
sp = 30
# twap_list = ['twd30_sp240', 'twd60_sp240', 'twd120_sp240', 'twd240_sp240'] #, 'twd30', 'twd60', 'twd120', 'twd240'
twap_list = ['twd30_sp30']
# test_name = 'ridge_v5_ma15_only_1'
# test_name = 'ridge_v5_lb_1y_alpha_1000'
# test_name = 'ridge_v9_1y' #_crct_240T
# test_name = 'ridge_v10_15T'
# test_name = 'ridge_v13_agg_240805_2_1y'
# test_name = 'esmb_objv1_agg_240805_2'
# test_name = "base_for_esmb_ridge_prcall"
test_name = 'merge_agg_241113_double3m_15d_73'
backtest_name = 'to_00125_maxmulti_2_mm_03_pf_001'  # mwgt_010 #_lqdt_500_mm_06 #_mm_03_pf_001
# backtest_name = 'to_040_mwgt_040'

plot_position_dist = True
plot_alpha_dist = False
plot_dd_period = False


# %%
path_config = load_path_config(project_dir)
processed_data_dir = Path(path_config['processed_data'])
result_dir = Path(path_config['result']) / 'model'
twap_data_dir = Path(path_config['twap_price'])


# %%
backtest_dir = result_dir / test_name / 'backtest' / backtest_name
backtest_dir.mkdir(parents=True, exist_ok=True)


# %%
name_to_save = f'{test_name}__{backtest_name}'


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


# %% predict
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
factor = fct_n_pct.div(fct_n_pct.abs().sum(axis=1), axis=0) #.fillna(0)
                                               

# %%
# 读取保存的文件
profit = pd.read_csv(backtest_dir / f"profit_{name_to_save}.csv", index_col='t')
profit.index = pd.to_datetime(profit.index)
w = pd.read_csv(backtest_dir / f"pos_{name_to_save}.csv", index_col=0)
w.index = pd.to_datetime(w.index)
long_short_num = pd.read_csv(backtest_dir / f"long_short_num_{name_to_save}.csv", index_col=0)
long_short_num.index = pd.to_datetime(long_short_num.index)

profit_files = [backtest_dir / f"profit_{twap_name}_{name_to_save}.csv" for twap_name in twap_list]
cum_returns_dict = {twap_name: pd.DataFrame() for twap_name in twap_list}
for twap_name, file in zip(twap_list, profit_files):
    profit_df = pd.read_csv(file, index_col=0, parse_dates=True)
    cum_returns = (1 + profit_df).cumprod() - 1
    cum_returns_dict[twap_name] = cum_returns
# 获取币种列表（即profit_df的列名）
coins = profit_df.columns

try:
    max_wgt_df = pd.read_csv(backtest_dir / f"max_wgt_{name_to_save}.csv", index_col=0)
except FileNotFoundError:
    max_wgt_df = None
    
    
profd = profit.resample('1d').sum()
profd['return'] = profd[f'raw_rtn_twd30_sp{sp}'] + profd['fee']
metrics = get_general_return_metrics(profd.loc[:, 'return'].values)


# %%
title_text = (f"{name_to_save}\n\n"
              f"Return: {metrics['return']:.2%}, Annualized Return: {metrics['return_annualized']:.2%}, "
              f"Max Drawdown: {metrics['max_dd']:.2%}, Sharpe Ratio: {metrics['sharpe_ratio']:.2f}\n"
              f"Calmar Ratio: {metrics['calmar_ratio']:.2f}, Sortino Ratio: {metrics['sortino_ratio']:.2f}, "
              f"Sterling Ratio: {metrics['sterling_ratio']:.2f}, Burke Ratio: {metrics['burke_ratio']:.2f}\n"
              f"Ulcer Index: {metrics['ulcer_index']:.2f}, Drawdown Recovery Ratio: {metrics['drawdown_recovery_ratio']:.2f}")

FONTSIZE_L1 = 25
FONTSIZE_L2 = 18
FONTSIZE_L3 = 15

fig, (ax0, ax1, ax2, ax3, ax4) = plt.subplots(5, 1, figsize=(36, 36), 
                                              gridspec_kw={'height_ratios': [4, 1, 1, 1, 1]})

# Set the title with the specified font size
fig.suptitle(title_text, fontsize=FONTSIZE_L1, y=0.96)

# Plot cumulative returns (occupying 2/3 of the height)
for twap_name in twap_list:
    ax0.plot((profd[f'raw_rtn_{twap_name}'] + profd['fee']).cumsum(), label=f'rtn_{twap_name}', linewidth=3)
ax0.plot((profd['fee']).abs().cumsum(), label='fee', color='r', linewidth=3)
ax0.grid(linestyle=":")
ax0.legend(loc="upper left", borderaxespad=0.5, fontsize=FONTSIZE_L3)
ax0.tick_params(labelsize=FONTSIZE_L2, pad=15)

# Plot long/short positions (occupying 1/6 of the height)
long_num = w.gt(1e-3).sum(axis=1)
short_num = w.lt(-1e-3).sum(axis=1)

ax1.plot(long_num, label='long_num', color='r')
ax1.plot(short_num, label='short_num', color='g')
ax1.grid(linestyle=":")
ax1.legend(loc="upper left", borderaxespad=0.5, fontsize=FONTSIZE_L3)
ax1.tick_params(labelsize=FONTSIZE_L2, pad=15)

# Plot cross-sectional volatility (occupying 1/6 of the height)
ax2.plot(cross_sectional_volatility.loc[profd.index], label='Cross-Sectional Volatility', color=plt.cm.Dark2(0))
ax2.grid(linestyle=":")
ax2.legend(loc="upper left", borderaxespad=0.5, fontsize=FONTSIZE_L3)
ax2.tick_params(labelsize=FONTSIZE_L2, pad=15)

# Plot cross-sectional kurtosis (occupying 1/6 of the height)
ax3.plot(cross_sectional_kurt.loc[profd.index], label='Cross-Sectional Kurtosis', color=plt.cm.Dark2(1))
ax3.grid(linestyle=":")
ax3.legend(loc="upper left", borderaxespad=0.5, fontsize=FONTSIZE_L3)
ax3.tick_params(labelsize=FONTSIZE_L2, pad=15)

# Plot cross-sectional tbdiff (occupying 1/6 of the height)
ax4.plot(cross_sectional_top_bottom_diff.loc[profd.index], label='Cross-Sectional Top-Bottom', color=plt.cm.Dark2(1))
ax4.grid(linestyle=":")
ax4.legend(loc="upper left", borderaxespad=0.5, fontsize=FONTSIZE_L3)
ax4.tick_params(labelsize=FONTSIZE_L2, pad=15)

plt.subplots_adjust(hspace=0.2)
plt.savefig(backtest_dir / f"{name_to_save}_with_vol.jpg", dpi=100, bbox_inches="tight")
plt.show()


# %%
# 计算符号（多空情况）
w_sign = np.sign(w)

# 按年份分组统计多头和空头的频次
w_yearly_long = (w_sign > 0).groupby(w.index.year).sum()  # 多头
w_yearly_short = (w_sign < 0).groupby(w.index.year).sum()  # 空头

# 特定币种的设置
highlight_symbols = ['btcusdt', 'ethusdt', 'bnbusdt', 'solusdt']

# 绘制每年多空频次的横向条形图
for year in w_yearly_long.index:
    long_data = w_yearly_long.loc[year]
    short_data = w_yearly_short.loc[year]
    
    # 按多空绝对值加总最多的从上到下排序
    combined_data = long_data + short_data
    sorted_index = combined_data.abs().sort_values(ascending=True).index
    long_data_sorted = long_data.reindex(sorted_index)
    short_data_sorted = short_data.reindex(sorted_index)
    
    # 创建较长的横向条形图
    plt.figure(figsize=(14, 50))
    
    # 多头：红色，条形向左
    plt.barh(long_data_sorted.index, long_data_sorted, color='red', label='Long', align='center')
    
    # 空头：绿色，条形向右
    plt.barh(short_data_sorted.index, -short_data_sorted, color='green', label='Short', align='center')
    
    # 设置特定币种的tick加粗或加红
    for tick in plt.gca().get_yticklabels():
        if tick.get_text().lower() in highlight_symbols:
            tick.set_fontweight('bold')
            tick.set_color('red')
    
    plt.title(f'Long/Short Frequency in {year}')
    plt.xlabel('Frequency')
    plt.ylabel('Symbol')
    plt.axvline(x=0, color='black', linewidth=0.8)
    plt.legend(loc='upper right')
    plt.grid(True, linestyle='--', linewidth=0.5)
    
    plt.savefig(backtest_dir / f"{name_to_save}_symbols_{year}.jpg", dpi=100, bbox_inches="tight")
    plt.show()
    

# %% plot pos dist
if plot_position_dist:
    plot_dir = backtest_dir / 'position_distribution'
    plot_dir.mkdir(parents=True, exist_ok=True)
    
    # 设置alpha的区间
    bins = np.arange(0, 1.05, 0.05)
    
    # 对每个时间戳绘制持仓的直方图并保存
    for timestamp in w.index:
        alpha_values = factor_rank.loc[timestamp]
        position_values = w.loc[timestamp]
    
        # 画图
        plt.figure(figsize=(10, 6))
        plt.hist(alpha_values, bins=bins, weights=position_values, edgecolor='black', alpha=0.7)
        plt.title(f'Histogram of Position Distribution on Alpha - {timestamp}')
        plt.xlabel('Alpha')
        plt.ylabel('Total Position')
        plt.grid(True)
        
        # 格式化时间戳为字符串，去掉横杠和冒号
        timestamp_str = str(timestamp).replace('-', '').replace(':', '').replace(' ', '_')
        
        # 保存图片
        plt.savefig(plot_dir / f'position_distribution_{timestamp_str}.png', dpi=100, bbox_inches="tight")
        
        # 关闭当前图像，防止内存占用过多
        plt.close()
        
        
# %% plot alpha dist
from sklearn.preprocessing import QuantileTransformer
if plot_alpha_dist:
    plot_dir = backtest_dir / 'alpha_distribution'
    plot_dir.mkdir(parents=True, exist_ok=True)
    
    rng = np.random.RandomState(304)
    qt = QuantileTransformer(output_distribution='normal', random_state=rng)
    
    # 对每个时间戳（每一行）绘制持仓的直方图并保存
    for timestamp, row in tqdm(list(predict_res.iterrows()), desc='plot_alpha_dist'):
        alpha_values = row  # 获取当前行的alpha值
        # 将一维数组转换为二维数组
        alpha_trans = qt.fit_transform(alpha_values.values.reshape(-1, 1)).flatten()
        
        # 处理 NaN 值，去除 NaN
        alpha_values = alpha_values[~np.isnan(alpha_values)]
        
        # 计算 bin 边界，使用原始数据的 min 和 max 来确定 bin 的范围
        bin_edges = np.histogram_bin_edges(alpha_values, bins=50)

        # 绘制持仓在不同alpha区间的分布
        plt.figure(figsize=(10, 6))
        plt.hist(alpha_values, bins=bin_edges, edgecolor='black', alpha=0.5)
        plt.hist(alpha_trans, bins=bin_edges, edgecolor='black', alpha=0.5)
        plt.title(f'Histogram of Alpha Values - {timestamp}')
        plt.xlabel('Alpha')
        plt.ylabel('Frequency')
        plt.grid(True)
        
        # 格式化时间戳为字符串，去掉横杠和冒号
        timestamp_str = str(timestamp).replace('-', '').replace(':', '').replace(' ', '_')
        
        # 保存图片
        plt.savefig(plot_dir / f'alpha_distribution_{timestamp_str}.png', dpi=100, bbox_inches="tight")
        
        # 关闭当前图像，防止内存占用过多
        plt.close()
    
    
# %%
# =============================================================================
# # 创建一个新文件夹来保存图片
# output_dir = backtest_dir / 'cumulative_returns'
# output_dir.mkdir(parents=True, exist_ok=True)
# 
# # 绘制每个币种的累积收益曲线并保存到文件
# for coin in coins:
#     plt.figure(figsize=(10, 6))
#     
#     for twap_name in twap_list:
#         plt.plot(cum_returns_dict[twap_name].index, cum_returns_dict[twap_name][coin], label=f'{twap_name}')
#     
#     plt.title(f'Cumulative Return of {coin}')
#     plt.xlabel('Date')
#     plt.ylabel('Cumulative Return')
#     plt.legend()
#     plt.grid(True)
#     
#     # 保存图表
#     output_path = output_dir / f'{coin}_cumulative_return.png'
#     plt.savefig(output_path)
#     plt.close()  # 关闭当前图形，防止内存占用过多
# =============================================================================
    
    
# %% 2308
if plot_dd_period:
    # 创建一个新文件夹来保存图片
    output_dir = backtest_dir / '2308'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 定义时间区间
    start_date = pd.to_datetime('2023-08-11')
    end_date = pd.to_datetime('2023-09-01')
    
    # 初始化表格来记录每个币种在指定时间段的收益
    returns_in_period = pd.DataFrame(index=coins, columns=twap_list)
    
    # 绘制每个币种的累积收益曲线并保存到文件
    for coin in coins:
        plt.figure(figsize=(10, 6))
        
        # 创建多行标题
        title_lines = [f'{coin}']
        
        for i_t, twap_name in enumerate(twap_list):
            # 获取该TWAP的累积收益数据
            cum_return = cum_returns_dict[twap_name][coin]
            
            # 计算指定时间段的收益
            return_in_period = np.sum(cum_return.loc[end_date] - cum_return.loc[start_date])
            returns_in_period.loc[coin, twap_name] = return_in_period
            
            # 绘制累积收益曲线
            plt.plot(cum_return.index, cum_return, label=f'{twap_name}')
            
            # 添加收益信息到标题
            if i_t == 0:
                title_lines.append(f'230811-230901 {twap_name}: {return_in_period:.2%}')
        
        # 在图中添加竖线，表示8月1日和9月1日
        plt.axvline(pd.to_datetime(start_date), color='red', linestyle='--', label='230811')
        plt.axvline(pd.to_datetime(end_date), color='blue', linestyle='--', label='230901')
        
        plt.title('\n'.join(title_lines))
        plt.xlabel('Date')
        plt.ylabel('Cumulative Return')
        plt.legend()
        plt.grid(True)
        
        # 保存图表
        output_path = output_dir / f'{coin}.png'
        plt.savefig(output_path)
        plt.close()  # 关闭当前图形，防止内存占用过多
    
    # 将每个币种在指定时间段的收益表格保存为CSV文件
    returns_in_period.to_csv(output_dir / 'returns_in_period_2308.csv')
