# -*- coding: utf-8 -*-
"""
Created on Tue Apr 27 15:40:22 2025

@author: Data Analysis Team

比较不同资金费用限制策略的回测结果
"""
# %% imports
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates


# %% add sys path
file_path = Path(__file__).resolve()
file_dir = file_path.parents[0]
project_dir = file_path.parents[1]
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
compare_name = 'backtest__compare_funding_side_limit'
model_name = 'merge_agg_250318_double3m_15d_73'
org_backtest_name = 'to_00125_maxmulti_2_mm_03_count_funding'
start_date = '20230701'
end_date = '20250415'
sp = 30
twap_list = ['twd30_sp30']

# 目标日期范围 - 可以根据需要调整
target_start_date = '20250101'
target_end_date = '20250415'

# 资金费用限制列表
funding_limit_list = [0.02, 0.015, 0.01, 0.005, 0.002]

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
    profd['return'] = profd[f'raw_rtn_twd30_sp{sp}'] + profd['fee'] + profd['funding']
    return profd

# 读取基准回测数据
org_profd = load_profit_data(model_name, org_backtest_name, start_date, end_date, sp)

# 计算基准的metrics
org_metrics = get_general_return_metrics(org_profd.loc[:, 'return'].values)

print(f"基准回测 {org_backtest_name} 的夏普比率: {org_metrics['sharpe_ratio']:.4f}")

# %% 构建参数字典
param_dict = {}
for funding_limit in funding_limit_list:
    param_dict[funding_limit] = f'{org_backtest_name}_fdside-fdlmt{funding_limit}'

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

# %% 准备绘图数据
# 计算基准回测在目标日期范围内的总回报率和资金费用
org_target_return = org_profd.loc[target_start_date:target_end_date, 'return'].sum()
org_total_funding = org_profd.loc[target_start_date:target_end_date, 'funding'].sum()
print(f"基准回测在目标日期范围内的总回报率: {org_target_return:.4f}")
print(f"基准回测在目标日期范围内的总资金费用: {org_total_funding:.4f}")

# 创建对比数据结构
sharpe_diff = []
total_return_diff = []
total_funding = []
funding_limit_values = []

for param, profd in profd_dict.items():
    funding_limit_values.append(param)
    
    # 计算夏普比率差异
    sharpe_diff.append(metrics_dict[param]['sharpe_ratio'] - org_metrics['sharpe_ratio'])
    
    # 计算目标日期范围内的总回报率差异
    target_return = profd.loc[target_start_date:target_end_date, 'return'].sum()
    total_return_diff.append(target_return - org_target_return)
    
    # 计算目标日期范围内的总资金费用
    funding = profd.loc[target_start_date:target_end_date, 'funding'].sum()
    total_funding.append(funding)

# %% 绘制所有参数组合的累计收益曲线
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.lines import Line2D

# 创建一个新的图表
plt.figure(figsize=(16, 10))

# 准备颜色渐变
n_variants = len(profd_dict)
cmap = plt.cm.viridis
colors = [cmap(i/n_variants) for i in range(n_variants)]

# 添加原始基准的累计收益曲线
cum_returns_org = org_profd['return'].cumsum()
plt.plot(cum_returns_org.index, cum_returns_org.values, 
         color='red', linewidth=3, label=f'Base: {org_backtest_name}')

# 添加图例
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
    
    param_str = f'fdlmt{param}'
    sharpe = metrics_dict[param]['sharpe_ratio']
    
    # 标记是否是表现最好的（排名前3）
    if i < 3:
        label = f'Top {i+1}: {param_str} (Sharpe: {sharpe:.4f})'
        linestyle = '-'
        linewidth = 2
        alpha = 1.0
    else:
        label = f'{param_str} (Sharpe: {sharpe:.4f})'
        linestyle = '--'
        linewidth = 1.5
        alpha = 0.8
    
    plt.plot(cum_returns.index, cum_returns.values, 
             color=color, linestyle=linestyle, linewidth=linewidth, alpha=alpha, 
             label=label)
    
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
    
    param_str = f'fdlmt{param}'
    
    # 标记是否是表现最好的（排名前3）
    if i < 3:
        label = f'Top {i+1}: {param_str}'
        linestyle = '-'
        linewidth = 2
        alpha = 1.0
    else:
        label = f'{param_str}'
        linestyle = '--'
        linewidth = 1.5
        alpha = 0.8
    
    plt.plot(cum_funding.index, cum_funding.values, 
             color=color, linestyle=linestyle, linewidth=linewidth, alpha=alpha, 
             label=label)
    
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

# 保存图表
save_path = save_dir / f'cumulative_funding_comparison_{start_date}_{end_date}.png'
plt.savefig(save_path, dpi=300, bbox_inches='tight')
print(f"累计资金费用曲线图表已保存至: {save_path}")

# %% 创建对比条形图
plt.figure(figsize=(16, 10))

# 准备数据
labels = [f'fdlmt{param}' for param in sorted_params]
sharpe_data = [metrics_dict[param]['sharpe_ratio'] for param in sorted_params]
baseline_sharpe = org_metrics['sharpe_ratio']

# 创建条形图
x = np.arange(len(labels))
width = 0.35

fig, ax = plt.subplots(figsize=(14, 8))
bars = ax.bar(x, sharpe_data, width, label='策略夏普比率')
ax.axhline(y=baseline_sharpe, color='red', linestyle='--', label=f'基准夏普比率: {baseline_sharpe:.4f}')

# 添加标签
ax.set_xlabel('资金费用限制', fontsize=14)
ax.set_ylabel('夏普比率', fontsize=14)
ax.set_title('不同资金费用限制策略的夏普比率对比', fontsize=16)
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

# 为每个柱子添加标签
for bar, value in zip(bars, sharpe_data):
    height = bar.get_height()
    ax.annotate(f'{value:.4f}',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),  # 3点偏移
                textcoords="offset points",
                ha='center', va='bottom')

plt.tight_layout()
save_path = save_dir / f'sharpe_ratio_comparison_bar_{start_date}_{end_date}.png'
plt.savefig(save_path, dpi=300, bbox_inches='tight')
plt.close()

# %% 创建对比条形图 - 目标期间回报率差异
plt.figure(figsize=(16, 10))

# 准备数据
labels = [f'fdlmt{param}' for param in sorted_params]
returns_data = [profd_dict[param].loc[target_start_date:target_end_date, 'return'].sum() for param in sorted_params]
baseline_return = org_profd.loc[target_start_date:target_end_date, 'return'].sum()

# 创建条形图
x = np.arange(len(labels))
width = 0.35

fig, ax = plt.subplots(figsize=(14, 8))
bars = ax.bar(x, returns_data, width, label='策略总回报')
ax.axhline(y=baseline_return, color='red', linestyle='--', label=f'基准总回报: {baseline_return:.4f}')

# 添加标签
ax.set_xlabel('资金费用限制', fontsize=14)
ax.set_ylabel('目标期间总回报', fontsize=14)
ax.set_title(f'不同资金费用限制策略在目标期间({target_start_date}至{target_end_date})的总回报对比', fontsize=16)
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

# 为每个柱子添加标签
for bar, value in zip(bars, returns_data):
    height = bar.get_height()
    ax.annotate(f'{value:.4f}',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),  # 3点偏移
                textcoords="offset points",
                ha='center', va='bottom')

plt.tight_layout()
save_path = save_dir / f'total_return_comparison_bar_{target_start_date}_{target_end_date}.png'
plt.savefig(save_path, dpi=300, bbox_inches='tight')
plt.close()

# %% 绘制回报率差异和资金费用关系散点图
plt.figure(figsize=(12, 8))

# 准备数据
x_values = np.array(sorted_params)
y_values = np.array([metrics_dict[param]['sharpe_ratio'] - org_metrics['sharpe_ratio'] for param in sorted_params])

# 绘制散点图
plt.scatter(x_values, y_values, s=100, alpha=0.7)

# 添加趋势线
z = np.polyfit(x_values, y_values, 1)
p = np.poly1d(z)
plt.plot(x_values, p(x_values), "r--", alpha=0.7)

# 添加标签
for i, txt in enumerate(x_values):
    plt.annotate(f'fdlmt{txt}', 
                (x_values[i], y_values[i]),
                xytext=(5, 5),
                textcoords='offset points')

# 设置图表标题和标签
plt.title('资金费用限制与夏普比率差异的关系', fontsize=16)
plt.xlabel('资金费用限制', fontsize=14)
plt.ylabel('夏普比率差异 (策略-基准)', fontsize=14)
plt.grid(True, alpha=0.3)
plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)

# 保存图表
save_path = save_dir / f'funding_limit_vs_sharpe_diff_{start_date}_{end_date}.png'
plt.savefig(save_path, dpi=300, bbox_inches='tight')
plt.close()

# %% 绘制每个参数组合的日度收益分布
top_n_params = sorted_params[:3]  # 选择表现最好的3个参数组合
plt.figure(figsize=(16, 8))

# 绘制基准的日度收益分布
plt.hist(org_profd['return'].values, bins=50, alpha=0.5, color='red', label=f'Base: {org_backtest_name}')

# 绘制表现最好的参数组合的日度收益分布
for i, param in enumerate(top_n_params):
    color = colors[i]
    profd = profd_dict[param]
    
    param_str = f'fdlmt{param}'
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
    
    param_str = f'fdlmt{param}'
    
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

# %% 创建指标对比表格
metrics_data = []
for param in sorted_params:
    metric = metrics_dict[param]
    profd = profd_dict[param]
    
    target_return = profd.loc[target_start_date:target_end_date, 'return'].sum()
    total_return = profd['return'].sum()
    max_dd = calc_max_drawdown(profd['return'].values)
    total_funding_value = profd.loc[target_start_date:target_end_date, 'funding'].sum()
    
    metrics_data.append({
        'funding_limit': param,
        'sharpe_ratio': metric['sharpe_ratio'],
        'total_return': total_return,
        'target_period_return': target_return,
        'max_drawdown': max_dd,
        'target_period_funding': total_funding_value
    })

# 为基准添加数据
base_max_dd = calc_max_drawdown(org_profd['return'].values)
metrics_data.append({
    'funding_limit': 'baseline',
    'sharpe_ratio': org_metrics['sharpe_ratio'],
    'total_return': org_profd['return'].sum(),
    'target_period_return': org_target_return,
    'max_drawdown': base_max_dd,
    'target_period_funding': org_total_funding
})

# 创建DataFrame并保存
metrics_df = pd.DataFrame(metrics_data)
metrics_df.to_csv(save_dir / f'metrics_comparison_{start_date}_{end_date}.csv', index=False)
print(f"指标对比表格已保存至: {save_dir / f'metrics_comparison_{start_date}_{end_date}.csv'}")

# %% 打印关键发现和结论
print("\n===== 关键发现和结论 =====")

# 找出最佳参数
best_param = sorted_params[0]
best_sharpe = metrics_dict[best_param]['sharpe_ratio']
best_fdlmt = best_param
best_sharpe_diff = best_sharpe - org_metrics['sharpe_ratio']

print(f"1. 最佳资金费用限制参数为: {best_fdlmt}, 夏普比率: {best_sharpe:.4f}, 相比基准提升: {best_sharpe_diff:.4f}")

# 计算相关性
corr = np.corrcoef(np.array(sorted_params), 
                  np.array([metrics_dict[param]['sharpe_ratio'] for param in sorted_params]))[0, 1]
print(f"2. 资金费用限制与夏普比率的相关性: {corr:.4f}")

# 资金费用分析
best_funding = profd_dict[best_param].loc[target_start_date:target_end_date, 'funding'].sum()
funding_diff = best_funding - org_total_funding
print(f"3. 最佳参数在目标期间的资金费用: {best_funding:.4f}, 相比基准差异: {funding_diff:.4f}")

# 收益分析
best_return = profd_dict[best_param].loc[target_start_date:target_end_date, 'return'].sum()
return_diff = best_return - org_target_return
print(f"4. 最佳参数在目标期间的总回报: {best_return:.4f}, 相比基准差异: {return_diff:.4f}")

# 风险分析
best_max_dd = calc_max_drawdown(profd_dict[best_param]['return'].values)
org_max_dd = calc_max_drawdown(org_profd['return'].values)
dd_diff = best_max_dd - org_max_dd
print(f"5. 最佳参数的最大回撤: {best_max_dd:.4f}, 相比基准差异: {dd_diff:.4f}")

# 策略建议
print("\n===== 策略建议 =====")
print(f"1. 根据分析结果，建议使用资金费用限制参数: {best_fdlmt}")
print(f"2. 该参数可提高夏普比率 {best_sharpe_diff:.4f}，同时在目标期间增加总回报 {return_diff:.4f}")
print(f"3. 随着资金费用限制的{'增加' if corr > 0 else '减少'}，夏普比率总体上{'增加' if corr > 0 else '减少'}")
print("4. 建议进一步探索资金费用限制与其他参数的交互效应，以进一步优化策略表现")