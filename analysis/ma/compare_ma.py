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
import matplotlib.dates as mdates
from matplotlib.lines import Line2D


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
compare_name = 'backtest__250318_compare_ma_methods'
model_name = 'merge_agg_250318_double3m_15d_73'
org_backtest_name = 'ma_simple'
org_backtest_name_to_compare = 'to_00125_maxmulti_2_mm_03_count_funding'
start_date = '20230701'
end_date = '20250401'
sp = 30
twap_list = ['twd30_sp30']

# 目标日期范围 - 可以根据需要调整
target_start_date = '20250101'
target_end_date = '20250401'

# 参数列表 - 用于分析不同的MA方法、窗口大小和百分比阈值
ma_method_list = ['ma', 'ewma']
ma_window_list = [24, 48, 144, 240, 336]
pct_list = [0.075, 0.1, 0.125, 0.15]

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

# 读取基准回测数据 (使用原始的ma_simple作为基准)
try:
    org_profd = load_profit_data(model_name, org_backtest_name_to_compare, start_date, end_date, sp)
    org_metrics = get_general_return_metrics(org_profd.loc[:, 'return'].values)
    print(f"基准回测 {org_backtest_name_to_compare} 的夏普比率: {org_metrics['sharpe_ratio']:.4f}")
except Exception as e:
    print(f"无法读取基准回测数据 {org_backtest_name_to_compare}: {e}")
    # 如果基准数据不存在，可以考虑使用第一个参数组合作为基准
    org_profd = None
    org_metrics = None

# %% 构建参数字典和读取所有参数组合的回测数据
param_dict = {}
for ma_method in ma_method_list:
    for ma_window in ma_window_list:
        for pct in pct_list:
            param = (ma_method, ma_window, pct)
            param_dict[param] = f'{org_backtest_name}-{ma_method}{ma_window}_pct{pct}'

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

# 如果基准数据不存在，使用第一个参数组合作为基准
if org_profd is None and profd_dict:
    first_param = next(iter(profd_dict.keys()))
    org_profd = profd_dict[first_param]
    org_metrics = metrics_dict[first_param]
    print(f"使用 {param_dict[first_param]} 作为基准回测")

# %% 构建3D热力图数据结构 (移动平均方法、窗口大小、百分比阈值)
# 初始化热力图数据结构
sharpe_matrix = {}
returns_matrix = {}
max_dd_matrix = {}

for ma_method in ma_method_list:
    # 为每种移动平均方法创建一个矩阵 (行为window，列为pct)
    sharpe_matrix[ma_method] = np.zeros((len(ma_window_list), len(pct_list)))
    returns_matrix[ma_method] = np.zeros((len(ma_window_list), len(pct_list)))
    max_dd_matrix[ma_method] = np.zeros((len(ma_window_list), len(pct_list)))
    
    # 用NaN填充矩阵
    sharpe_matrix[ma_method][:] = np.nan
    returns_matrix[ma_method][:] = np.nan
    max_dd_matrix[ma_method][:] = np.nan

# 填充热力图数据
for param, metrics in metrics_dict.items():
    ma_method, ma_window, pct = param
    
    # 找到在列表中的索引
    window_idx = ma_window_list.index(ma_window)
    pct_idx = pct_list.index(pct)
    
    # 填充夏普比率矩阵
    sharpe_matrix[ma_method][window_idx, pct_idx] = metrics['sharpe_ratio']
    
    # 计算目标日期范围内的总回报率
    profd = profd_dict[param]
    target_return = profd.loc[target_start_date:target_end_date, 'return'].sum()
    returns_matrix[ma_method][window_idx, pct_idx] = target_return
    
    # 计算最大回撤
    cum_returns = profd['return'].cumsum()
    max_net = np.maximum.accumulate(cum_returns.values)
    drawdown = max_net - cum_returns.values
    max_drawdown = np.max(drawdown)
    max_dd_matrix[ma_method][window_idx, pct_idx] = max_drawdown

# 计算基准回测在目标日期范围内的总回报率
org_target_return = org_profd.loc[target_start_date:target_end_date, 'return'].sum()
print(f"基准回测在目标日期范围内的总回报率: {org_target_return:.4f}")

# %% 绘制所有参数组合的累计收益曲线
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
    
    ma_method, ma_window, pct = param
    param_str = f'{ma_method}{ma_window}_pct{pct}'
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

# %% 创建热力图函数
def plot_heatmap(matrix, x_labels, y_labels, title, cmap, save_path, annot=True, fmt='.3f', 
                center=None, cbar_kws=None, mask=None):
    plt.figure(figsize=(12, 8))
    
    if cbar_kws is None:
        cbar_kws = {'label': 'Value'}
    
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
    plt.xlabel('Percentage Threshold', fontsize=14)
    plt.ylabel('MA Window', fontsize=14)
    
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

# %% 为每种移动平均方法绘制夏普比率热力图
for ma_method in ma_method_list:
    # 检查是否有数据
    if np.all(np.isnan(sharpe_matrix[ma_method])):
        print(f"没有 {ma_method} 方法的有效数据，跳过绘制热力图")
        continue
    
    # 创建缺失数据的掩码
    mask = np.isnan(sharpe_matrix[ma_method])
    
    # 绘制夏普比率热力图
    plot_heatmap(
        sharpe_matrix[ma_method], 
        [str(pct) for pct in pct_list], 
        [str(window) for window in ma_window_list],
        f'Sharpe Ratio Heatmap - {ma_method.upper()} ({start_date} to {end_date})',
        'viridis',  # 使用viridis色彩方案
        save_dir / f'sharpe_heatmap_{ma_method}_{start_date}_{end_date}.png',
        center=None,  # 不设置中心点
        cbar_kws={'label': '夏普比率'},
        mask=mask
    )
    
    # 绘制目标日期范围回报率热力图
    mask = np.isnan(returns_matrix[ma_method])
    plot_heatmap(
        returns_matrix[ma_method], 
        [str(pct) for pct in pct_list], 
        [str(window) for window in ma_window_list],
        f'Total Returns Heatmap - {ma_method.upper()} ({target_start_date} to {target_end_date})',
        'viridis',  # 使用viridis色彩方案
        save_dir / f'returns_heatmap_{ma_method}_{target_start_date}_{target_end_date}.png',
        center=None,  # 不设置中心点
        cbar_kws={'label': '总回报率'},
        mask=mask
    )
    
    # 绘制最大回撤热力图
    mask = np.isnan(max_dd_matrix[ma_method])
    plot_heatmap(
        max_dd_matrix[ma_method], 
        [str(pct) for pct in pct_list], 
        [str(window) for window in ma_window_list],
        f'Max Drawdown Heatmap - {ma_method.upper()} ({start_date} to {end_date})',
        'YlOrRd_r',  # 黄橙红色彩方案的反转，值越小（回撤越小）越好
        save_dir / f'max_drawdown_heatmap_{ma_method}_{start_date}_{end_date}.png',
        center=None,  
        cbar_kws={'label': '最大回撤'},
        mask=mask
    )

# %% 找出表现最好的参数组合
print("\n各MA方法的最佳参数组合：")
for ma_method in ma_method_list:
    sharpe_mat = sharpe_matrix[ma_method]
    valid_mask = ~np.isnan(sharpe_mat)
    
    if np.any(valid_mask):
        best_idx = np.unravel_index(np.nanargmax(sharpe_mat), sharpe_mat.shape)
        best_window = ma_window_list[best_idx[0]]
        best_pct = pct_list[best_idx[1]]
        best_sharpe = sharpe_mat[best_idx]
        
        print(f"  {ma_method.upper()}: 最佳窗口={best_window}, 百分比阈值={best_pct}, "
              f"夏普比率: {best_sharpe:.4f}")
    else:
        print(f"  {ma_method.upper()}: 没有有效数据")

# 找出所有参数组合中表现最好的
best_param = None
best_sharpe = -np.inf

for param, metrics in metrics_dict.items():
    sharpe = metrics['sharpe_ratio']
    if sharpe > best_sharpe:
        best_sharpe = sharpe
        best_param = param

if best_param:
    ma_method, ma_window, pct = best_param
    print(f"\n总体最佳参数组合: MA方法={ma_method}, 窗口={ma_window}, 百分比阈值={pct}, "
          f"夏普比率: {best_sharpe:.4f}")

# %% 绘制箱线图，分析参数对性能的影响
plt.figure(figsize=(15, 10))

# MA方法的影响
ma_method_groups = []
ma_method_labels = []
for method in ma_method_list:
    method_values = []
    for param, metrics in metrics_dict.items():
        if param[0] == method:
            method_values.append(metrics['sharpe_ratio'])
    
    if method_values:  # 只有在有数据时才添加
        ma_method_groups.append(method_values)
        ma_method_labels.append(f'method={method}')

if ma_method_groups:  # 只有在有数据时才绘图
    plt.subplot(1, 3, 1)
    plt.boxplot(ma_method_groups, labels=ma_method_labels)
    plt.title('Impact of MA Method on Sharpe Ratio')
    plt.grid(True, linestyle='--', alpha=0.7)
    if org_metrics:
        plt.axhline(y=org_metrics['sharpe_ratio'], color='r', linestyle='-', alpha=0.3)  # 添加基准线

# 窗口大小的影响
window_groups = []
window_labels = []
for window in ma_window_list:
    window_values = []
    for param, metrics in metrics_dict.items():
        if param[1] == window:
            window_values.append(metrics['sharpe_ratio'])
    
    if window_values:
        window_groups.append(window_values)
        window_labels.append(f'window={window}')

if window_groups:
    plt.subplot(1, 3, 2)
    plt.boxplot(window_groups, labels=window_labels)
    plt.title('Impact of Window Size on Sharpe Ratio')
    plt.grid(True, linestyle='--', alpha=0.7)
    if org_metrics:
        plt.axhline(y=org_metrics['sharpe_ratio'], color='r', linestyle='-', alpha=0.3)

# 百分比阈值的影响
pct_groups = []
pct_labels = []
for pct in pct_list:
    pct_values = []
    for param, metrics in metrics_dict.items():
        if param[2] == pct:
            pct_values.append(metrics['sharpe_ratio'])
    
    if pct_values:
        pct_groups.append(pct_values)
        pct_labels.append(f'pct={pct}')

if pct_groups:
    plt.subplot(1, 3, 3)
    plt.boxplot(pct_groups, labels=pct_labels)
    plt.title('Impact of Percentage Threshold on Sharpe Ratio')
    plt.grid(True, linestyle='--', alpha=0.7)
    if org_metrics:
        plt.axhline(y=org_metrics['sharpe_ratio'], color='r', linestyle='-', alpha=0.3)

plt.tight_layout()
plt.savefig(save_dir / f'param_impact_boxplots_{start_date}_{end_date}.png', dpi=300, bbox_inches='tight')
plt.close()

# %% 计算每个参数的平均和中位数影响
print("\n各参数对夏普比率的平均影响：")

# MA方法的影响
print("\nMA方法的影响:")
for method in ma_method_list:
    values = []
    for param, metrics in metrics_dict.items():
        if param[0] == method:
            values.append(metrics['sharpe_ratio'])
    
    if values:
        mean_sharpe = np.mean(values)
        median_sharpe = np.median(values)
        print(f"  MA方法={method}: 平均夏普比率={mean_sharpe:.4f}, 中位数夏普比率={median_sharpe:.4f}")

# 窗口大小的影响
print("\n窗口大小的影响:")
for window in ma_window_list:
    values = []
    for param, metrics in metrics_dict.items():
        if param[1] == window:
            values.append(metrics['sharpe_ratio'])
    
    if values:
        mean_sharpe = np.mean(values)
        median_sharpe = np.median(values)
        print(f"  窗口大小={window}: 平均夏普比率={mean_sharpe:.4f}, 中位数夏普比率={median_sharpe:.4f}")

# 百分比阈值的影响
print("\n百分比阈值的影响:")
for pct in pct_list:
    values = []
    for param, metrics in metrics_dict.items():
        if param[2] == pct:
            values.append(metrics['sharpe_ratio'])
    
    if values:
        mean_sharpe = np.mean(values)
        median_sharpe = np.median(values)
        print(f"  百分比阈值={pct}: 平均夏普比率={mean_sharpe:.4f}, 中位数夏普比率={median_sharpe:.4f}")

# %% 绘制每个参数组合的日度收益分布
top_n_params = sorted_params[:3]  # 选择表现最好的3个参数组合
plt.figure(figsize=(16, 8))

# 绘制基准的日度收益分布
plt.hist(org_profd['return'].values, bins=50, alpha=0.5, color='red', label=f'Base: {org_backtest_name}')

# 绘制表现最好的参数组合的日度收益分布
for i, param in enumerate(top_n_params):
    color = colors[i]
    profd = profd_dict[param]
    
    ma_method, ma_window, pct = param
    param_str = f'{ma_method}{ma_window}_pct{pct}'
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
    
    ma_method, ma_window, pct = param
    param_str = f'{ma_method}{ma_window}_pct{pct}'
    
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
plt.ylabel('Drawdown', fontsize=14)
plt.grid(True, alpha=0.3)
plt.legend(loc='lower left', fontsize=12)

# 格式化x轴日期
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=3))
plt.gcf().autofmt_xdate()

plt.tight_layout()
plt.savefig(save_dir / f'drawdown_comparison_{start_date}_{end_date}.png', dpi=300, bbox_inches='tight')
plt.close()
