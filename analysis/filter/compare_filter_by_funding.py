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
import seaborn as sns


# %% add sys path
file_path = Path(__file__).resolve()
file_dir = file_path.parents[0]
project_dir = file_path.parents[2]
sys.path.append(str(project_dir))


# %% import self-defined
from utils.dirutils import load_path_config
from test_and_eval.scores import get_general_return_metrics


# %% 读取原始基准回测数据
path_config = load_path_config(project_dir)
model_dir = Path(path_config['result']) / 'model'
analysis_dir = Path(path_config['result']) / 'analysis'
twap_data_dir = Path(path_config['twap_price'])

# %% 定义参数
compare_name = 'backtest__compare_filter_by_funding'
model_name = 'merge_agg_241227_cgy_zxt_double3m_15d_73'
org_backtest_name = 'to_00125_maxmulti_2_mm_03_pf_001'
org_backtest_name_to_compare = 'to_00125_maxmulti_2_mm_03_pf_001_count_funding'
start_date = '20230701'
end_date = '20250307'
sp = 30
twap_list = ['twd30_sp30']

# 目标日期范围 - 可以根据需要调整
target_start_date = '20250101'
target_end_date = '20250307'

# %% 创建保存目录
save_dir = analysis_dir / compare_name
save_dir.mkdir(parents=True, exist_ok=True)

# %% 读取基准回测数据并计算profd和metrics
def load_profit_data(model_name, backtest_name, start_date, end_date, sp):
    path = model_dir / model_name / 'backtest' / backtest_name / f'profit_{model_name}__{backtest_name}.parquet'
    profit = pd.read_parquet(path)
    profit.index = pd.to_datetime(profit.index)
    profit = profit.loc[start_date:end_date]
    profd = profit.resample('1d').sum()
    profd['return'] = profd[f'raw_rtn_twd30_sp{sp}'] + profd['fee']  + profd['funding']
    return profd

# 读取基准回测数据
org_profd = load_profit_data(model_name, org_backtest_name_to_compare, start_date, end_date, sp)

# 计算基准的metrics
org_metrics = get_general_return_metrics(org_profd.loc[:, 'return'].values)

print(f"基准回测 {org_backtest_name_to_compare} 的夏普比率: {org_metrics['sharpe_ratio']:.4f}")

# %% 读取所有参数组合的回测数据
funding_abs_limit_list = [0.02, 0.015, 0.01, 0.005, 0.002]
cooldown_list = [16, 48, 96]

# 构建参数字典
param_dict = {}
for funding_abs_limit in funding_abs_limit_list:
    for cooldown in cooldown_list:
        param_dict[(funding_abs_limit, cooldown)] = f'{org_backtest_name}_cnt_fd-fdlmt{funding_abs_limit}_cd{cooldown}'

# 读取所有参数组合的回测数据
profd_dict = {}
metrics_dict = {}

for param, backtest_name in param_dict.items():
    try:
        profd = load_profit_data(model_name, backtest_name, start_date, end_date, sp)
        profd_dict[param] = profd
        metrics_dict[param] = get_general_return_metrics(profd.loc[:, 'return'].values)
    except Exception as e:
        print(f"无法读取 {backtest_name}: {e}")
        continue

print(f"成功读取了 {len(profd_dict)} 个参数组合的回测数据")

# %% 构建热力图数据结构
# 横轴: funding_abs_limit
# 纵轴: cooldown
funding_abs_limit_labels = [str(limit) for limit in funding_abs_limit_list]
cooldown_labels = [str(cd) for cd in cooldown_list]

# 初始化热力图矩阵 - 存储与基准回测的差异
sharpe_diff_matrix = np.zeros((len(cooldown_list), len(funding_abs_limit_list)))
returns_diff_matrix = np.zeros((len(cooldown_list), len(funding_abs_limit_list)))
total_funding_matrix = np.zeros((len(cooldown_list), len(funding_abs_limit_list)))

# 计算基准回测在目标日期范围内的总回报率
org_target_return = org_profd.loc[target_start_date:target_end_date, 'return'].sum()
org_total_funding = org_profd.loc[target_start_date:target_end_date, 'funding'].sum()
print(f"基准回测在目标日期范围内的总回报率: {org_target_return:.4f}")
print(f"基准回测在目标日期范围内的总资金费用: {org_total_funding:.4f}")

# 填充矩阵 - 计算与基准的差异
for y_idx, cooldown in enumerate(cooldown_list):
    for x_idx, funding_abs_limit in enumerate(funding_abs_limit_list):
        param = (funding_abs_limit, cooldown)
        
        if param not in metrics_dict:
            # 如果没有读取到该参数组合的数据，设置为NaN
            sharpe_diff_matrix[y_idx, x_idx] = np.nan
            returns_diff_matrix[y_idx, x_idx] = np.nan
            total_funding_matrix[y_idx, x_idx] = np.nan
            continue
        
        # 计算夏普比率与基准的差异
        sharpe_diff_matrix[y_idx, x_idx] = metrics_dict[param]['sharpe_ratio'] - org_metrics['sharpe_ratio']
        
        # 计算目标日期范围内的回报率与基准的差异
        profd = profd_dict[param]
        target_return = profd.loc[target_start_date:target_end_date, 'return'].sum()
        returns_diff_matrix[y_idx, x_idx] = target_return - org_target_return
        
        # 计算目标日期范围内的总资金费用
        total_funding = profd.loc[target_start_date:target_end_date, 'funding'].sum()
        total_funding_matrix[y_idx, x_idx] = total_funding

# %% 绘制所有参数组合的累计收益曲线
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np
from matplotlib.lines import Line2D

# 创建一个新的图表
plt.figure(figsize=(16, 10))

# 准备颜色渐变
# 根据参数组合数量创建渐变色
n_variants = len(profd_dict)
cmap = plt.cm.viridis
colors = [cmap(i/n_variants) for i in range(n_variants)]

# 添加原始基准的累计收益曲线
cum_returns_org = org_profd['return'].cumsum()
plt.plot(cum_returns_org.index, cum_returns_org.values, 
         color='red', linewidth=3, label=f'Base: {org_backtest_name}')

# 添加所有参数组合的累计收益曲线
legend_handles = []
legend_handles.append(Line2D([0], [0], color='red', linewidth=3, label=f'Base: {org_backtest_name}'))

# 按策略效果（夏普比率）对参数组合进行排序
sorted_params = sorted(metrics_dict.keys(), 
                     key=lambda x: metrics_dict[x]['sharpe_ratio'], 
                     reverse=True)

# 绘制所有参数组合的累计收益曲线
for i, param in enumerate(sorted_params):
    color = colors[min(i, len(colors)-1)]
    
    profd = profd_dict[param]
    cum_returns = profd['return'].cumsum()
    
    funding_abs_limit, cooldown = param
    param_str = f'fdlmt{funding_abs_limit}_cd{cooldown}'
    sharpe = metrics_dict[param]['sharpe_ratio']
    
    # 标记是否是表现最好的（排名前3）
    if i < 3:
        label = f'Top {i+1}: {param_str} (Sharpe: {sharpe:.4f})'
        linestyle = '-'
        linewidth = 2
        alpha = 1.0
    elif i < 10:
        label = f'Top {i+1}: {param_str} (Sharpe: {sharpe:.4f})'
        linestyle = '--'
        linewidth = 1.5
        alpha = 0.8
    else:
        # 对于其余参数组合，不加入图例，使用半透明线条
        label = None
        linestyle = '-'
        linewidth = 0.8
        alpha = 0.4
    
    plt.plot(cum_returns.index, cum_returns.values, 
             color=color, linestyle=linestyle, linewidth=linewidth, alpha=alpha, 
             label=label)
    
    if label:
        legend_handles.append(Line2D([0], [0], color=color, linestyle=linestyle, 
                                    linewidth=linewidth, alpha=alpha, label=label))

# 如果有指定目标日期范围，添加垂直线标记
if target_start_date and target_end_date:
    plt.axvline(x=pd.to_datetime(target_start_date), color='gray', linestyle='--', alpha=0.5)
    plt.axvline(x=pd.to_datetime(target_end_date), color='gray', linestyle='--', alpha=0.5)
    plt.axvspan(pd.to_datetime(target_start_date), pd.to_datetime(target_end_date), 
               alpha=0.1, color='gray', label='Target Period')

# 设置图表标题和标签
plt.title(f'Cumulative Returns Comparison ({start_date} to {end_date})', fontsize=16)
plt.xlabel('Date', fontsize=14)
plt.ylabel('Cumulative Return', fontsize=14)

# 格式化x轴日期
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=3))
plt.gcf().autofmt_xdate()

# 添加网格
plt.grid(True, alpha=0.3)

# 添加图例
plt.legend(handles=legend_handles, loc='upper left', fontsize=9)

# 显示图表
plt.tight_layout()

# 保存图表
save_path = save_dir / f'cumulative_returns_comparison_{start_date}_{end_date}.png'
plt.savefig(save_path, dpi=300, bbox_inches='tight')
print(f"累计收益曲线图表已保存至: {save_path}")

# %% 绘制累计资金费用曲线
plt.figure(figsize=(16, 10))

# 添加原始基准的累计资金费用曲线
cum_funding_org = org_profd['funding'].cumsum()
plt.plot(cum_funding_org.index, cum_funding_org.values, 
         color='red', linewidth=3, label=f'Base: {org_backtest_name}')

# 准备图例
funding_legend_handles = []
funding_legend_handles.append(Line2D([0], [0], color='red', linewidth=3, label=f'Base: {org_backtest_name}'))

# 绘制所有参数组合的累计资金费用曲线
for i, param in enumerate(sorted_params):
    color = colors[min(i, len(colors)-1)]
    
    profd = profd_dict[param]
    cum_funding = profd['funding'].cumsum()
    
    funding_abs_limit, cooldown = param
    param_str = f'fdlmt{funding_abs_limit}_cd{cooldown}'
    
    # 标记是否是表现最好的（排名前3）
    if i < 3:
        label = f'Top {i+1}: {param_str}'
        linestyle = '-'
        linewidth = 2
        alpha = 1.0
    elif i < 10:
        label = f'Top {i+1}: {param_str}'
        linestyle = '--'
        linewidth = 1.5
        alpha = 0.8
    else:
        # 对于其余参数组合，不加入图例，使用半透明线条
        label = None
        linestyle = '-'
        linewidth = 0.8
        alpha = 0.4
    
    plt.plot(cum_funding.index, cum_funding.values, 
             color=color, linestyle=linestyle, linewidth=linewidth, alpha=alpha, 
             label=label)
    
    if label:
        funding_legend_handles.append(Line2D([0], [0], color=color, linestyle=linestyle, 
                                           linewidth=linewidth, alpha=alpha, label=label))

# 如果有指定目标日期范围，添加垂直线标记
if target_start_date and target_end_date:
    plt.axvline(x=pd.to_datetime(target_start_date), color='gray', linestyle='--', alpha=0.5)
    plt.axvline(x=pd.to_datetime(target_end_date), color='gray', linestyle='--', alpha=0.5)
    plt.axvspan(pd.to_datetime(target_start_date), pd.to_datetime(target_end_date), 
               alpha=0.1, color='gray', label='Target Period')

# 设置图表标题和标签
plt.title(f'Cumulative Funding Comparison ({start_date} to {end_date})', fontsize=16)
plt.xlabel('Date', fontsize=14)
plt.ylabel('Cumulative Funding', fontsize=14)

# 格式化x轴日期
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=3))
plt.gcf().autofmt_xdate()

# 添加网格
plt.grid(True, alpha=0.3)

# 添加图例
plt.legend(handles=funding_legend_handles, loc='upper left', fontsize=9)

# 显示图表
plt.tight_layout()

# 保存图表
save_path = save_dir / f'cumulative_funding_comparison_{start_date}_{end_date}.png'
plt.savefig(save_path, dpi=300, bbox_inches='tight')
print(f"累计资金费用曲线图表已保存至: {save_path}")

# %% 创建热力图函数
def plot_heatmap(matrix, x_labels, y_labels, title, cmap, save_path, annot=True, fmt='.3f', 
                center=0, cbar_kws=None, mask=None):
    plt.figure(figsize=(12, 10))
    
    if cbar_kws is None:
        cbar_kws = {'label': 'Diff With Org'}
    
    # 创建热力图
    ax = sns.heatmap(matrix, 
                    xticklabels=x_labels, 
                    yticklabels=y_labels, 
                    cmap=cmap,
                    annot=annot, 
                    fmt=fmt,
                    linewidths=0.5,
                    center=center,
                    cbar_kws=cbar_kws,
                    mask=mask)
    
    # 设置标题和标签
    plt.title(title, fontsize=16)
    plt.xlabel('Funding Abs Limit', fontsize=14)
    plt.ylabel('Cooldown Period', fontsize=14)
    
    # 调整刻度标签
    plt.xticks(rotation=0, fontsize=12)
    plt.yticks(rotation=0, fontsize=12)
    
    # 添加颜色条
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=10)
    
    # 保存图片
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

# 创建缺失数据的掩码
mask = np.isnan(sharpe_diff_matrix)

# %% 绘制夏普比率差异热力图
plot_heatmap(
    sharpe_diff_matrix, 
    funding_abs_limit_labels, 
    cooldown_labels,
    f'Sharpe Ratio Difference vs Baseline ({start_date} to {end_date})',
    'RdBu_r',  # 红蓝色彩方案，负值为红色(差于基准)，正值为蓝色(优于基准)
    save_dir / f'sharpe_diff_heatmap_{start_date}_{end_date}.png',
    center=0,  # 以0为中心点
    cbar_kws={'label': '夏普比率差异 (策略-基准)'},
    mask=mask
)

# %% 绘制目标日期范围回报率差异热力图
plot_heatmap(
    returns_diff_matrix, 
    funding_abs_limit_labels, 
    cooldown_labels,
    f'Total Returns Difference vs Baseline ({target_start_date} to {target_end_date})',
    'RdBu_r',  # 红蓝色彩方案
    save_dir / f'returns_diff_heatmap_{target_start_date}_{target_end_date}.png',
    center=0,  # 以0为中心点
    cbar_kws={'label': '回报率差异 (策略-基准)'},
    mask=mask
)

# %% 绘制总资金费用热力图
plot_heatmap(
    total_funding_matrix, 
    funding_abs_limit_labels, 
    cooldown_labels,
    f'Total Funding ({target_start_date} to {target_end_date})',
    'YlOrRd',  # 黄橙红色彩方案，适合显示资金费用
    save_dir / f'total_funding_heatmap_{target_start_date}_{target_end_date}.png',
    center=None,  
    cbar_kws={'label': '总资金费用'},
    mask=mask
)

# %% 找出表现最好的参数组合
valid_mask = ~np.isnan(sharpe_diff_matrix)
if np.any(valid_mask):
    best_sharpe_idx = np.unravel_index(np.nanargmax(sharpe_diff_matrix), sharpe_diff_matrix.shape)
    best_funding_limit = funding_abs_limit_list[best_sharpe_idx[1]]
    best_cooldown = cooldown_list[best_sharpe_idx[0]]
    print(f"最佳夏普比率参数组合: 资金费用阈值={best_funding_limit}, 冷却期={best_cooldown}, "
          f"夏普比率提升: {sharpe_diff_matrix[best_sharpe_idx]:.4f}")

valid_mask = ~np.isnan(returns_diff_matrix)
if np.any(valid_mask):
    best_returns_idx = np.unravel_index(np.nanargmax(returns_diff_matrix), returns_diff_matrix.shape)
    best_funding_limit = funding_abs_limit_list[best_returns_idx[1]]
    best_cooldown = cooldown_list[best_returns_idx[0]]
    print(f"最佳回报率参数组合: 资金费用阈值={best_funding_limit}, 冷却期={best_cooldown}, "
          f"回报率提升: {returns_diff_matrix[best_returns_idx]:.4f}")

# %% 保存结果数据框以便进一步分析
results_df = pd.DataFrame(index=pd.MultiIndex.from_product([cooldown_list], names=['cooldown']))

# 添加夏普比率差异和回报率差异
for y_idx, cooldown in enumerate(cooldown_list):
    for x_idx, funding_abs_limit in enumerate(funding_abs_limit_list):
        param = (funding_abs_limit, cooldown)
        col_name = f'fdlmt{funding_abs_limit}'
        
        if col_name not in results_df.columns:
            results_df[col_name] = np.nan
        
        if param in metrics_dict:
            results_df.loc[cooldown, col_name] = metrics_dict[param]['sharpe_ratio'] - org_metrics['sharpe_ratio']

# 保存结果表格
results_df.to_csv(save_dir / f'param_comparison_sharpe_diff_{start_date}_{end_date}.csv')

# %% 绘制箱线图，分析参数对性能的影响
plt.figure(figsize=(15, 10))

# 资金费用阈值的影响
fdlmt_groups = []
fdlmt_labels = []
for fdlmt in funding_abs_limit_list:
    fdlmt_values = []
    for y_idx in range(len(cooldown_list)):
        for x_idx, limit in enumerate(funding_abs_limit_list):
            if limit == fdlmt and not np.isnan(sharpe_diff_matrix[y_idx, x_idx]):
                fdlmt_values.append(sharpe_diff_matrix[y_idx, x_idx])
    if fdlmt_values:  # 只有在有数据时才添加
        fdlmt_groups.append(fdlmt_values)
        fdlmt_labels.append(f'fdlmt={fdlmt}')

if fdlmt_groups:  # 只有在有数据时才绘图
    plt.subplot(1, 2, 1)
    plt.boxplot(fdlmt_groups, labels=fdlmt_labels)
    plt.title('Impact of Funding Limit on Sharpe Ratio Difference')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)  # 添加基准线

# 冷却期的影响
cooldown_groups = []
cooldown_labels = []
for cd in cooldown_list:
    cooldown_values = []
    for y_idx, cooldown in enumerate(cooldown_list):
        if cooldown == cd:
            for x_idx in range(len(funding_abs_limit_list)):
                if not np.isnan(sharpe_diff_matrix[y_idx, x_idx]):
                    cooldown_values.append(sharpe_diff_matrix[y_idx, x_idx])
    if cooldown_values:
        cooldown_groups.append(cooldown_values)
        cooldown_labels.append(f'cooldown={cd}')

if cooldown_groups:
    plt.subplot(1, 2, 2)
    plt.boxplot(cooldown_groups, labels=cooldown_labels)
    plt.title('Impact of Cooldown on Sharpe Ratio Difference')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)

plt.tight_layout()
plt.savefig(save_dir / f'param_impact_diff_boxplots_{start_date}_{end_date}.png', dpi=300, bbox_inches='tight')
plt.close()

# %% 计算每个参数的平均和中位数影响
print("\n各参数对夏普比率的平均影响：")
# 资金费用阈值的影响
print("\n资金费用阈值的影响:")
for fdlmt in funding_abs_limit_list:
    values = []
    for y_idx in range(len(cooldown_list)):
        for x_idx, limit in enumerate(funding_abs_limit_list):
            if limit == fdlmt and not np.isnan(sharpe_diff_matrix[y_idx, x_idx]):
                values.append(sharpe_diff_matrix[y_idx, x_idx])
    
    if values:
        mean_diff = np.mean(values)
        median_diff = np.median(values)
        pos_ratio = np.mean([1 if v > 0 else 0 for v in values])
        print(f"  资金费用阈值={fdlmt}: 平均差异={mean_diff:.4f}, 中位数差异={median_diff:.4f}, 优于基准比例={pos_ratio:.2%}")

# 冷却期的影响
print("\n冷却期的影响:")
for cd in cooldown_list:
    values = []
    for y_idx, cooldown in enumerate(cooldown_list):
        if cooldown == cd:
            for x_idx in range(len(funding_abs_limit_list)):
                if not np.isnan(sharpe_diff_matrix[y_idx, x_idx]):
                    values.append(sharpe_diff_matrix[y_idx, x_idx])
    
    if values:
        mean_diff = np.mean(values)
        median_diff = np.median(values)
        pos_ratio = np.mean([1 if v > 0 else 0 for v in values])
        print(f"  冷却期={cd}: 平均差异={mean_diff:.4f}, 中位数差异={median_diff:.4f}, 优于基准比例={pos_ratio:.2%}")

# %% 绘制每个参数组合的日度收益分布
top_n_params = sorted_params[:3]  # 选择表现最好的3个参数组合
plt.figure(figsize=(16, 8))

# 绘制基准的日度收益分布
plt.hist(org_profd['return'].values, bins=50, alpha=0.5, color='red', label=f'Base: {org_backtest_name}')

# 绘制表现最好的参数组合的日度收益分布
for i, param in enumerate(top_n_params):
    color = colors[i]
    profd = profd_dict[param]
    
    funding_abs_limit, cooldown = param
    param_str = f'fdlmt{funding_abs_limit}_cd{cooldown}'
    sharpe = metrics_dict[param]['sharpe_ratio']
    
    plt.hist(profd['return'].values, bins=50, alpha=0.5, color=color, 
             label=f'Top {i+1}: {param_str} (Sharpe: {sharpe:.4f})')

plt.title('Daily Return Distribution Comparison', fontsize=16)
plt.xlabel('Daily Return', fontsize=14)
plt.ylabel('Frequency', fontsize=14)
plt.grid(True, alpha=0.3)
plt.legend(loc='upper left', fontsize=12)
plt.tight_layout()
plt.savefig(save_dir / f'daily_return_distribution_{start_date}_{end_date}.png', dpi=300, bbox_inches='tight')
plt.close()

# %% 绘制回撤分析图
plt.figure(figsize=(16, 10))

# 定义回撤计算函数
def calc_max_drawdown(return_arr):
    net_arr = np.cumsum(return_arr)
    max_net_arr = np.maximum.accumulate(net_arr)
    drawdown_arr = max_net_arr - net_arr
    return np.max(drawdown_arr)

# 计算并绘制基准的回撤
cum_returns_org = org_profd['return'].cumsum()
max_net_org = np.maximum.accumulate(cum_returns_org.values)
drawdown_org = pd.Series(max_net_org - cum_returns_org.values, index=cum_returns_org.index)
plt.plot(drawdown_org.index, drawdown_org.values, color='red', linewidth=3, 
         label=f'Base: {org_backtest_name}')

# 为Top N参数组合绘制回撤
for i, param in enumerate(top_n_params):
    color = colors[i]
    profd = profd_dict[param]
    
    funding_abs_limit, cooldown = param
    param_str = f'fdlmt{funding_abs_limit}_cd{cooldown}'
    
    # 计算回撤
    cum_returns = profd['return'].cumsum()
    max_net = np.maximum.accumulate(cum_returns.values)
    drawdown = pd.Series(max_net - cum_returns.values, index=cum_returns.index)
    
    # 计算最大回撤并添加到标签中
    max_dd = calc_max_drawdown(profd['return'].values)
    
    plt.plot(drawdown.index, drawdown.values, color=color, linewidth=2, 
             label=f'Top {i+1}: {param_str} (Max DD: {max_dd:.4f})')

# 如果有指定目标日期范围，添加垂直线标记
if target_start_date and target_end_date:
    plt.axvline(x=pd.to_datetime(target_start_date), color='gray', linestyle='--', alpha=0.5)
    plt.axvline(x=pd.to_datetime(target_end_date), color='gray', linestyle='--', alpha=0.5)
    plt.axvspan(pd.to_datetime(target_start_date), pd.to_datetime(target_end_date), 
                alpha=0.1, color='gray', label='Target Period')

plt.title('Drawdown Comparison', fontsize=16)
plt.xlabel('Date', fontsize=14)
plt.ylabel('Drawdown (%)', fontsize=14)
plt.grid(True, alpha=0.3)
plt.legend(loc='lower left', fontsize=12)

# 格式化x轴日期
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=3))
plt.gcf().autofmt_xdate()

plt.tight_layout()
plt.savefig(save_dir / f'drawdown_comparison_{start_date}_{end_date}.png', dpi=300, bbox_inches='tight')
plt.close()