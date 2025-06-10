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
compare_name = 'backtest__compare_filter_by_vol'
model_name = 'merge_agg_241227_cgy_zxt_double3m_15d_73'
org_backtest_name = 'to_00125_maxmulti_2_mm_03_pf_001'
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
    profd['return'] = profd[f'raw_rtn_twd30_sp{sp}'] + profd['fee']
    return profd

# 读取基准回测数据
org_profd = load_profit_data(model_name, org_backtest_name, start_date, end_date, sp)

# 计算基准的metrics
org_metrics = get_general_return_metrics(org_profd.loc[:, 'return'].values)

print(f"基准回测 {org_backtest_name} 的夏普比率: {org_metrics['sharpe_ratio']:.4f}")

# %% 读取所有参数组合的回测数据
ns = [2, 8, 16]  # 计算波动的区间，为30min的倍数，分别对应1h、4h、8h
ms = [2*24*30, 2*24*30*3]  # 计算波动zscore的回看窗口，分别对应1个月、3个月
thres_list = [5, 7, 10, 15]  # zscore标准差倍数
periods = [8, 48, 144, 480]  # 冷静期，即触发异常后，需要多少个窗口内没有再次触发才能恢复交易
k_values = [2, 3, 5, 100]  # top K：波动超出标准差倍数 且 波动在截面处于top K 才记为异常（100即忽略此参数，作为对比）

# 构建参数字典
param_dict = {}
for n in ns:
    for m in ms:
        for thres in thres_list:
            for period in periods:
                for k in k_values:
                    param_dict[(n, m, thres, period, k)] = f'{org_backtest_name}-przsc_n{n}_m{m}_th{thres}_p{period}_topk{k}'

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

# %% 创建热力图数据结构
# 横轴: n, m, thres 的组合
# 纵轴: period, k 的组合
x_labels = []
for n in ns:
    for m in ms:
        for thres in thres_list:
            x_labels.append(f'n{n}_m{m}_th{thres}')

y_labels = []
for period in periods:
    for k in k_values:
        y_labels.append(f'p{period}_k{k}')

# 初始化热力图矩阵 - 存储与基准回测的差异
sharpe_diff_matrix = np.zeros((len(y_labels), len(x_labels)))
returns_diff_matrix = np.zeros((len(y_labels), len(x_labels)))

# 计算基准回测在目标日期范围内的总回报率
org_target_return = org_profd.loc[target_start_date:target_end_date, 'return'].sum()
print(f"基准回测在目标日期范围内的总回报率: {org_target_return:.4f}")

# 填充矩阵 - 计算与基准的差异
for y_idx, (period, k) in enumerate([(p, k) for p in periods for k in k_values]):
    for x_idx, (n, m, thres) in enumerate([(n, m, t) for n in ns for m in ms for t in thres_list]):
        param = (n, m, thres, period, k)
        
        if param not in metrics_dict:
            # 如果没有读取到该参数组合的数据，设置为NaN
            sharpe_diff_matrix[y_idx, x_idx] = np.nan
            returns_diff_matrix[y_idx, x_idx] = np.nan
            continue
        
        # 计算夏普比率与基准的差异
        sharpe_diff_matrix[y_idx, x_idx] = metrics_dict[param]['sharpe_ratio'] - org_metrics['sharpe_ratio']
        
        # 计算目标日期范围内的回报率与基准的差异
        profd = profd_dict[param]
        target_return = profd.loc[target_start_date:target_end_date, 'return'].sum()
        returns_diff_matrix[y_idx, x_idx] = target_return - org_target_return
        
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

# 添加其他参数组合的累计收益曲线
legend_handles = []
legend_handles.append(Line2D([0], [0], color='red', linewidth=3, label=f'Base: {org_backtest_name}'))

# 按策略效果（夏普比率）对参数组合进行排序
sorted_params = sorted(metrics_dict.keys(), 
                       key=lambda x: metrics_dict[x]['sharpe_ratio'], 
                       reverse=True)

# 只画出表现最好的10个变种和最差的5个变种，以防图表过于拥挤
top_n = 10
bottom_n = 5
params_to_plot = sorted_params[:top_n] + sorted_params[-bottom_n:]

# 绘制所选参数组合的累计收益曲线
for i, param in enumerate(params_to_plot):
    color_idx = i if i < top_n else n_variants - (i - top_n)
    color = colors[min(color_idx, len(colors)-1)]
    
    profd = profd_dict[param]
    cum_returns = profd['return'].cumsum()
    
    n, m, thres, period, k = param
    param_str = f'n{n}_m{m}_th{thres}_p{period}_topk{k}'
    sharpe = metrics_dict[param]['sharpe_ratio']
    
    # 标记是否是表现最好的（排名前3）
    if i < 3:
        label = f'Top {i+1}: {param_str} (Sharpe: {sharpe:.4f})'
        linestyle = '-'
        linewidth = 2
    elif i < top_n:
        label = f'Top {i+1}: {param_str} (Sharpe: {sharpe:.4f})'
        linestyle = '--'
        linewidth = 1.5
    else:
        label = f'Bottom {i-top_n+1}: {param_str} (Sharpe: {sharpe:.4f})'
        linestyle = ':'
        linewidth = 1
    
    plt.plot(cum_returns.index, cum_returns.values, 
             color=color, linestyle=linestyle, linewidth=linewidth, alpha=0.8)
    
    legend_handles.append(Line2D([0], [0], color=color, linestyle=linestyle, 
                                 linewidth=linewidth, alpha=0.8, label=label))

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
print(f"图表已保存至: {save_path}")

# %% 创建热力图函数
def plot_heatmap(matrix, x_labels, y_labels, title, cmap, save_path, annot=True, fmt='.3f', 
                 center=0, cbar_kws=None, mask=None):
    plt.figure(figsize=(20, 16))
    
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
    plt.xlabel('n_m_threshold combo', fontsize=14)
    plt.ylabel('period_k combo', fontsize=14)
    
    # 调整刻度标签
    plt.xticks(rotation=90, fontsize=10)
    plt.yticks(rotation=0, fontsize=10)
    
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
    x_labels, 
    y_labels,
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
    x_labels, 
    y_labels,
    f'Total Returns Difference vs Baseline ({target_start_date} to {target_end_date})',
    'RdBu_r',  # 红蓝色彩方案
    save_dir / f'returns_diff_heatmap_{target_start_date}_{target_end_date}.png',
    center=0,  # 以0为中心点
    cbar_kws={'label': '回报率差异 (策略-基准)'},
    mask=mask
)

# %% 找出表现最好的参数组合
valid_mask = ~np.isnan(sharpe_diff_matrix)
if np.any(valid_mask):
    best_sharpe_idx = np.unravel_index(np.nanargmax(sharpe_diff_matrix), sharpe_diff_matrix.shape)
    print(f"最佳夏普比率参数组合: {x_labels[best_sharpe_idx[1]]} - {y_labels[best_sharpe_idx[0]]}, "
          f"夏普比率提升: {sharpe_diff_matrix[best_sharpe_idx]:.4f}")

valid_mask = ~np.isnan(returns_diff_matrix)
if np.any(valid_mask):
    best_returns_idx = np.unravel_index(np.nanargmax(returns_diff_matrix), returns_diff_matrix.shape)
    print(f"最佳回报率参数组合: {x_labels[best_returns_idx[1]]} - {y_labels[best_returns_idx[0]]}, "
          f"回报率提升: {returns_diff_matrix[best_returns_idx]:.4f}")

# %% 保存结果数据框以便进一步分析
results_df = pd.DataFrame(index=pd.MultiIndex.from_product([periods, k_values], names=['period', 'k']))

# 添加夏普比率差异
for y_idx, (period, k) in enumerate([(p, k) for p in periods for k in k_values]):
    for x_idx, (n, m, thres) in enumerate([(n, m, t) for n in ns for m in ms for t in thres_list]):
        param = (n, m, thres, period, k)
        col_name = f'n{n}_m{m}_th{thres}'
        
        if col_name not in results_df.columns:
            results_df[col_name] = np.nan
        
        if param in metrics_dict:
            results_df.loc[(period, k), col_name] = metrics_dict[param]['sharpe_ratio'] - org_metrics['sharpe_ratio']

# 保存结果表格
results_df.to_csv(save_dir / f'param_comparison_sharpe_diff_{start_date}_{end_date}.csv')

# %% 绘制箱线图，分析参数对性能的影响
plt.figure(figsize=(15, 10))

# 分别计算每个n值对应的夏普比率差异分布
n_groups = []
n_labels = []
for n in ns:
    n_values = []
    for y_idx in range(len(y_labels)):
        for x_idx, (n_val, m, thres) in enumerate([(n_v, m_v, t) for n_v in ns for m_v in ms for t in thres_list]):
            if n_val == n and not np.isnan(sharpe_diff_matrix[y_idx, x_idx]):
                n_values.append(sharpe_diff_matrix[y_idx, x_idx])
    if n_values:  # 只有在有数据时才添加
        n_groups.append(n_values)
        n_labels.append(f'n={n}')

if n_groups:  # 只有在有数据时才绘图
    plt.subplot(2, 3, 1)
    plt.boxplot(n_groups, labels=n_labels)
    plt.title('Impact of n on Sharpe Ratio Difference')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)  # 添加基准线

# 同样计算其他参数的影响
# m 值的影响
m_groups = []
m_labels = []
for m in ms:
    m_values = []
    for y_idx in range(len(y_labels)):
        for x_idx, (n, m_val, thres) in enumerate([(n_v, m_v, t) for n_v in ns for m_v in ms for t in thres_list]):
            if m_val == m and not np.isnan(sharpe_diff_matrix[y_idx, x_idx]):
                m_values.append(sharpe_diff_matrix[y_idx, x_idx])
    if m_values:
        m_groups.append(m_values)
        m_labels.append(f'm={m}')

if m_groups:
    plt.subplot(2, 3, 2)
    plt.boxplot(m_groups, labels=m_labels)
    plt.title('Impact of m on Sharpe Ratio Difference')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)

# thres 值的影响
thres_groups = []
thres_labels = []
for thres in thres_list:
    thres_values = []
    for y_idx in range(len(y_labels)):
        for x_idx, (n, m, t) in enumerate([(n_v, m_v, t_v) for n_v in ns for m_v in ms for t_v in thres_list]):
            if t == thres and not np.isnan(sharpe_diff_matrix[y_idx, x_idx]):
                thres_values.append(sharpe_diff_matrix[y_idx, x_idx])
    if thres_values:
        thres_groups.append(thres_values)
        thres_labels.append(f'thres={thres}')

if thres_groups:
    plt.subplot(2, 3, 3)
    plt.boxplot(thres_groups, labels=thres_labels)
    plt.title('Impact of threshold on Sharpe Ratio Difference')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)

# period 值的影响
period_groups = []
period_labels = []
for period in periods:
    period_values = []
    for y_idx, (p, k) in enumerate([(p_v, k_v) for p_v in periods for k_v in k_values]):
        if p == period:
            for x_idx in range(len(x_labels)):
                if not np.isnan(sharpe_diff_matrix[y_idx, x_idx]):
                    period_values.append(sharpe_diff_matrix[y_idx, x_idx])
    if period_values:
        period_groups.append(period_values)
        period_labels.append(f'period={period}')

if period_groups:
    plt.subplot(2, 3, 4)
    plt.boxplot(period_groups, labels=period_labels)
    plt.title('Impact of period on Sharpe Ratio Difference')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)

# k 值的影响
k_groups = []
k_labels = []
for k in k_values:
    k_values_list = []
    for y_idx, (p, k_val) in enumerate([(p_v, k_v) for p_v in periods for k_v in k_values]):
        if k_val == k:
            for x_idx in range(len(x_labels)):
                if not np.isnan(sharpe_diff_matrix[y_idx, x_idx]):
                    k_values_list.append(sharpe_diff_matrix[y_idx, x_idx])
    if k_values_list:
        k_groups.append(k_values_list)
        k_labels.append(f'k={k}')

if k_groups:
    plt.subplot(2, 3, 5)
    plt.boxplot(k_groups, labels=k_labels)
    plt.title('Impact of k on Sharpe Ratio Difference')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)

plt.tight_layout()
plt.savefig(save_dir / f'param_impact_diff_boxplots_{start_date}_{end_date}.png', dpi=300, bbox_inches='tight')
plt.close()

# %% 计算每个参数的平均和中位数影响
print("\n各参数对夏普比率的平均影响：")
for param_name, param_values in [('n', ns), ('m', ms), ('thres', thres_list), ('period', periods), ('k', k_values)]:
    print(f"\n{param_name}值的影响:")
    for val in param_values:
        values = []
        if param_name == 'n':
            for y_idx in range(len(y_labels)):
                for x_idx, (n, _, _) in enumerate([(n_v, m_v, t_v) for n_v in ns for m_v in ms for t_v in thres_list]):
                    if n == val and not np.isnan(sharpe_diff_matrix[y_idx, x_idx]):
                        values.append(sharpe_diff_matrix[y_idx, x_idx])
        elif param_name == 'm':
            for y_idx in range(len(y_labels)):
                for x_idx, (_, m, _) in enumerate([(n_v, m_v, t_v) for n_v in ns for m_v in ms for t_v in thres_list]):
                    if m == val and not np.isnan(sharpe_diff_matrix[y_idx, x_idx]):
                        values.append(sharpe_diff_matrix[y_idx, x_idx])
        elif param_name == 'thres':
            for y_idx in range(len(y_labels)):
                for x_idx, (_, _, t) in enumerate([(n_v, m_v, t_v) for n_v in ns for m_v in ms for t_v in thres_list]):
                    if t == val and not np.isnan(sharpe_diff_matrix[y_idx, x_idx]):
                        values.append(sharpe_diff_matrix[y_idx, x_idx])
        elif param_name == 'period':
            for y_idx, (p, _) in enumerate([(p_v, k_v) for p_v in periods for k_v in k_values]):
                if p == val:
                    for x_idx in range(len(x_labels)):
                        if not np.isnan(sharpe_diff_matrix[y_idx, x_idx]):
                            values.append(sharpe_diff_matrix[y_idx, x_idx])
        elif param_name == 'k':
            for y_idx, (_, k) in enumerate([(p_v, k_v) for p_v in periods for k_v in k_values]):
                if k == val:
                    for x_idx in range(len(x_labels)):
                        if not np.isnan(sharpe_diff_matrix[y_idx, x_idx]):
                            values.append(sharpe_diff_matrix[y_idx, x_idx])
        
        if values:
            mean_diff = np.mean(values)
            median_diff = np.median(values)
            pos_ratio = np.mean([1 if v > 0 else 0 for v in values])
            print(f"  {param_name}={val}: 平均差异={mean_diff:.4f}, 中位数差异={median_diff:.4f}, 优于基准比例={pos_ratio:.2%}")