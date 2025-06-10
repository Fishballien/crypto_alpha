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
import matplotlib as mpl
from matplotlib.gridspec import GridSpec
import matplotlib.dates as mdates
import seaborn as sns


# %% add sys path
file_path = Path(__file__).resolve()
file_dir = file_path.parents[0]
project_dir = file_path.parents[1]
sys.path.append(str(project_dir))


# %% import self-defined
from utils.dirutils import load_path_config
from test_and_eval.scores import get_general_return_metrics
from backtest.analysisutils import top_5_minus_bottom_95


# %%
path_config = load_path_config(project_dir)
model_dir = Path(path_config['result']) / 'model'
analysis_dir = Path(path_config['result']) / 'analysis'
twap_data_dir = Path(path_config['twap_price'])


# %%
compare_name = 'backtest__p02_vs_p04_vs_p04_fdlmt'
# 新的结构支持每个版本内有多个模型组合
compare_dict = {
    'p0.2': [
        {
            'name': 'p0.2_model',  # 子模型名称
            'model_name': 'merge_agg_241227_cgy_zxt_double3m_15d_73',
            'backtest_name': 'to_00125_maxmulti_2_mm_03_pf_001_count_funding',
        },
    ],
    'p0.4': [
        {
            'name': 'p0.4_model',
            'model_name': 'merge_agg_250127_cgy_zxt_double3m_15d_73',
            'backtest_name': 'to_00125_maxmulti_2_mm_03_pf_001_count_funding',
        },
    ],
    'p0.4_fdlmt': [
        {
            'name': 'p0.4_fdlmt0.01_cd48',
            'model_name': 'merge_agg_250127_cgy_zxt_double3m_15d_73',
            'backtest_name': 'to_00125_maxmulti_2_mm_03_pf_001_cnt_fd-fdlmt0.01_cd48',
        },
        {
            'name': 'p0.4_fdlmt0.002_cd48',
            'model_name': 'merge_agg_250127_cgy_zxt_double3m_15d_73',
            'backtest_name': 'to_00125_maxmulti_2_mm_03_pf_001_cnt_fd-fdlmt0.002_cd48',
        },
        {
            'name': 'p0.4_fdlmt0.01_cd16',
            'model_name': 'merge_agg_250127_cgy_zxt_double3m_15d_73',
            'backtest_name': 'to_00125_maxmulti_2_mm_03_pf_001_cnt_fd-fdlmt0.01_cd16',
        },
    ],
}

start_date = '20230701'
end_date = '20250307'

sp = 30
twap_list = ['twd30_sp30']


# %% dir
save_dir = analysis_dir / compare_name
save_dir.mkdir(parents=True, exist_ok=True)


# %% 加载所有子模型数据及计算等权组合
# 存储所有子模型的收益数据
all_model_profits = {}
# 存储每个版本的等权组合
version_equal_weighted = {}

for version, models in compare_dict.items():
    version_models_profit = {}
    
    # 加载每个子模型的数据
    for model_info in models:
        model_name = model_info['model_name']
        backtest_name = model_info['backtest_name']
        sub_model_name = model_info['name']
        
        path = model_dir / model_name / 'backtest' / backtest_name / f'profit_{model_name}__{backtest_name}.parquet'
        profit = pd.read_parquet(path)
        profit.index = pd.to_datetime(profit.index)
        profit = profit.loc[start_date:end_date]
        profd = profit.resample('1d').sum()
        profd['return'] = profd[f'raw_rtn_twd30_sp{sp}'] + profd['fee'] + profd['funding']
        
        # 存储子模型数据
        all_model_profits[sub_model_name] = profd
        version_models_profit[sub_model_name] = profd
    
    # 如果版本中只有一个模型，直接使用该模型
    if len(models) == 1:
        sub_model_name = models[0]['name']
        version_equal_weighted[version] = all_model_profits[sub_model_name]
    else:
        # 计算等权组合
        # 首先创建一个包含所有子模型return的DataFrame
        returns_df = pd.DataFrame({name: profit['return'] for name, profit in version_models_profit.items()})
        
        # 计算等权平均收益
        eq_weighted_return = returns_df.mean(axis=1)
        
        # 创建一个新的DataFrame，复制第一个子模型的结构，然后替换return列
        eq_weighted_profd = version_models_profit[list(version_models_profit.keys())[0]].copy()
        
        # 按比例分配fee和funding（分配给等权收益）
        if len(models) > 1:
            fee_sum = sum([p['fee'] for p in version_models_profit.values()]) / len(models)
            funding_sum = sum([p['funding'] for p in version_models_profit.values()]) / len(models)
            raw_rtn_column = f'raw_rtn_twd30_sp{sp}'
            raw_rtn_sum = sum([p[raw_rtn_column] for p in version_models_profit.values()]) / len(models)
            
            eq_weighted_profd['fee'] = fee_sum
            eq_weighted_profd['funding'] = funding_sum
            eq_weighted_profd[raw_rtn_column] = raw_rtn_sum
        
        # 替换return列为等权平均收益
        eq_weighted_profd['return'] = eq_weighted_return
        
        # 存储等权组合
        version_equal_weighted[version] = eq_weighted_profd


# %% 计算所有模型（包括子模型和等权组合）的指标
all_metrics = {}

# 计算所有子模型的指标
for name, profd in all_model_profits.items():
    all_metrics[name] = get_general_return_metrics(profd.loc[:, 'return'].values)

# 计算所有等权组合的指标
for version, profd in version_equal_weighted.items():
    all_metrics[version] = get_general_return_metrics(profd.loc[:, 'return'].values)


# %% 加载价格数据计算市场指标
curr_px_path = twap_data_dir / f'curr_price_sp{sp}.parquet'
curr_price = pd.read_parquet(curr_px_path)
main_columns = curr_price.columns
to_mask = curr_price.isna()

rtn = curr_price.pct_change(int(240/sp), fill_method=None).replace([np.inf, -np.inf], np.nan)
cross_sectional_volatility = rtn.std(axis=1).resample('1d').mean()
cross_sectional_kurt = rtn.kurtosis(axis=1).resample('1d').mean()
cross_sectional_top_bottom_diff = rtn.abs().apply(top_5_minus_bottom_95, axis=1).resample('1d').mean()


# %% 增强版可视化函数
def plot_enhanced_multi_model_visualization(
    version_equal_weighted, all_model_profits, all_metrics,
    cross_sectional_volatility, cross_sectional_kurt, cross_sectional_top_bottom_diff,
    twap_list, compare_name, save_dir, show_average=False):
    """
    创建增强版多模型可视化，支持每个版本内有多个子模型
    
    Parameters:
    -----------
    version_equal_weighted : dict
        各版本等权组合的收益dataframe
    all_model_profits : dict
        所有子模型的收益dataframe
    all_metrics : dict
        所有模型（含子模型和等权组合）的性能指标
    cross_sectional_volatility : Series
        横截面波动率时间序列
    cross_sectional_kurt : Series
        横截面峰度时间序列
    cross_sectional_top_bottom_diff : Series
        横截面top-bottom差异时间序列
    twap_list : list
        TWAP名称列表
    compare_name : str
        比较图名称
    save_dir : Path
        保存图表的目录
    show_average : bool, optional
        是否显示平均线(default: False)
    """

    # 设置主题以获得更好的美观效果
    plt.style.use('seaborn-v0_8-whitegrid')
    sns.set_context("notebook", font_scale=1.2)

    # 为版本和子模型使用不同的调色板
    version_colors = sns.color_palette('viridis', len(version_equal_weighted))
    sub_model_colors = {}
    
    # 为每个版本的子模型分配略微不同的颜色（基于版本主颜色但稍微淡一些）
    for i, (version, _) in enumerate(version_equal_weighted.items()):
        base_color = version_colors[i]
        # 查找该版本的所有子模型
        sub_models = [m['name'] for m in compare_dict[version]]
        
        if len(sub_models) == 1:
            # 如果只有一个子模型，颜色与版本相同
            sub_model_colors[sub_models[0]] = base_color
        else:
            # 如果有多个子模型，为每个子模型生成淡化版本的颜色
            for j, sub_model in enumerate(sub_models):
                # 淡化颜色（添加一些白色）
                r, g, b = base_color
                # 越高的j，颜色越淡
                alpha = 0.6 + (j * 0.1)  # 控制淡化程度
                sub_model_colors[sub_model] = (r*alpha + (1-alpha), g*alpha + (1-alpha), b*alpha + (1-alpha))
    
    accent_colors = sns.color_palette('Set2', 3)

    # 更好的字体
    plt.rcParams['font.family'] = 'DejaVu Sans'
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['axes.labelsize'] = 12

    # 创建具有改进布局的图形
    fig = plt.figure(figsize=(20, 16))
    gs = GridSpec(4, 1, figure=fig, height_ratios=[4, 1, 1, 1], hspace=0.5)

    # 主标题
    fig.suptitle(compare_name, fontsize=24, fontweight='bold', y=0.95)

    # 面板1：累积收益
    ax0 = fig.add_subplot(gs[0])
    combined_returns = pd.DataFrame()

    # 先绘制子模型（薄线）
    for sub_model_name, profd in all_model_profits.items():
        # 找出该子模型所属的版本
        parent_version = None
        for version, models in compare_dict.items():
            if any(m['name'] == sub_model_name for m in models):
                parent_version = version
                break
        
        # 只有多模型版本才绘制子模型线
        if len(compare_dict[parent_version]) > 1:
            return_text = f"{sub_model_name}: R: {all_metrics[sub_model_name]['return']:.2%}, MaxDD: {all_metrics[sub_model_name]['max_dd']:.2%}, Sharpe: {all_metrics[sub_model_name]['sharpe_ratio']:.2f}"
            
            for twap_name in twap_list:
                # 使用淡色和细线绘制子模型
                cum_returns = (profd[f'raw_rtn_{twap_name}'] + profd['fee'] + profd['funding']).cumsum()
                ax0.plot(cum_returns, label=return_text, linewidth=1.5, 
                        color=sub_model_colors[sub_model_name], alpha=0.7, 
                        linestyle='-')

    # 然后绘制等权版本模型（粗线）
    for idx, (version, profd) in enumerate(version_equal_weighted.items()):
        return_text = f"{version}: R: {all_metrics[version]['return']:.2%}, MaxDD: {all_metrics[version]['max_dd']:.2%}, Sharpe: {all_metrics[version]['sharpe_ratio']:.2f}"
        
        for twap_name in twap_list:
            combined_returns[f"{version}_{twap_name}"] = profd[f'raw_rtn_{twap_name}'] + profd['fee'] + profd['funding']
            
            # 主收益线 - 更粗更鲜艳
            cum_returns = (profd[f'raw_rtn_{twap_name}'] + profd['fee'] + profd['funding']).cumsum()
            ax0.plot(cum_returns, label=return_text, linewidth=3, color=version_colors[idx], alpha=0.9)
            
            # 费用和资金费率作为更薄更透明的线
            ax0.plot((profd['fee']).abs().cumsum(), linewidth=1.5, 
                    color=version_colors[idx], linestyle='--', alpha=0.5, 
                    label=f"{version}: Fee")
            ax0.plot((profd['funding']).abs().cumsum(), linewidth=1.5, 
                    color=version_colors[idx], linestyle=':', alpha=0.5,
                    label=f"{version}: Funding")
            
            # 添加终点注释
            last_date = profd.index[-1]
            last_value = cum_returns[-1]
            ax0.scatter(last_date, last_value, s=100, color=version_colors[idx], zorder=5, edgecolor='white')

    # 可选平均线
    if show_average:
        avg_return = combined_returns.mean(axis=1)
        avg_metric = get_general_return_metrics(avg_return.values)
        avg_return_text = f"Average: R: {avg_metric['return']:.2%}, MaxDD: {avg_metric['max_dd']:.2%}, Sharpe: {avg_metric['sharpe_ratio']:.2f}"
        avg_cum_return = avg_return.cumsum()

        ax0.plot(avg_cum_return, label=avg_return_text, 
                linewidth=4, color='black', alpha=0.8, zorder=10)

    # 添加零线参考的阴影区域
    ax0.axhline(y=0, color='grey', linestyle='-', alpha=0.3, linewidth=1)

    # 添加带自定义样式的网格
    ax0.grid(True, linestyle=':', alpha=0.6)

    # 改进图例放置和样式
    ax0.legend(loc="upper left", fontsize=12, framealpha=0.9, 
            edgecolor='lightgrey', fancybox=True)

    # 更好的x轴日期格式
    ax0.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax0.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.setp(ax0.xaxis.get_majorticklabels(), rotation=45, ha='right')

    ax0.set_title('Cumulative Returns', fontsize=18, pad=15)
    ax0.set_ylabel('Return', fontsize=14)

    # 使用第一个版本的索引作为基准
    first_version = list(version_equal_weighted.keys())[0]
    reference_index = version_equal_weighted[first_version].index

    # 添加面板2：横截面波动率
    ax2 = fig.add_subplot(gs[1], sharex=ax0)
    volatility_line = ax2.plot(cross_sectional_volatility.loc[reference_index], 
                            label='Cross-Sectional Volatility', 
                            color=accent_colors[0], linewidth=2)

    # 在波动率线下添加阴影区域
    ax2.fill_between(cross_sectional_volatility.loc[reference_index].index, 
                    0, cross_sectional_volatility.loc[reference_index], 
                    color=accent_colors[0], alpha=0.2)

    ax2.grid(True, linestyle=':', alpha=0.6)
    ax2.legend(loc="upper left", fontsize=12, framealpha=0.9)
    ax2.set_ylabel('Volatility', fontsize=14)
    ax2.set_title('Market Volatility', fontsize=14, pad=10)

    # 添加面板3：横截面峰度
    ax3 = fig.add_subplot(gs[2], sharex=ax0)
    kurt_line = ax3.plot(cross_sectional_kurt.loc[reference_index], 
                        label='Cross-Sectional Kurtosis', 
                        color=accent_colors[1], linewidth=2)

    # 在峰度线下添加阴影区域
    ax3.fill_between(cross_sectional_kurt.loc[reference_index].index, 
                    0, cross_sectional_kurt.loc[reference_index], 
                    color=accent_colors[1], alpha=0.2)

    ax3.grid(True, linestyle=':', alpha=0.6)
    ax3.legend(loc="upper left", fontsize=12, framealpha=0.9)
    ax3.set_ylabel('Kurtosis', fontsize=14)
    ax3.set_title('Distribution Kurtosis', fontsize=14, pad=10)

    # 添加面板4：横截面Top-Bottom差异
    ax4 = fig.add_subplot(gs[3], sharex=ax0)
    top_bottom_line = ax4.plot(cross_sectional_top_bottom_diff.loc[reference_index], 
                            label='Cross-Sectional Top-Bottom', 
                            color=accent_colors[2], linewidth=2)

    # 在top-bottom线下添加阴影区域
    ax4.fill_between(cross_sectional_top_bottom_diff.loc[reference_index].index, 
                    0, cross_sectional_top_bottom_diff.loc[reference_index], 
                    color=accent_colors[2], alpha=0.2)

    ax4.grid(True, linestyle=':', alpha=0.6)
    ax4.legend(loc="upper left", fontsize=12, framealpha=0.9)
    ax4.set_ylabel('Difference', fontsize=14)
    ax4.set_xlabel('Date', fontsize=14)
    ax4.set_title('Top-Bottom Spread', fontsize=14, pad=10)

    # 调整布局以获得更好的间距
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # 为标题调整

    # 添加创建日期/时间脚注
    from datetime import datetime
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    fig.text(0.02, 0.01, f"Generated: {now}", fontsize=10, color='gray')

    # 以更高分辨率和更好的压缩保存
    plt.savefig(save_dir / f"{compare_name}_multi_model_enhanced.png", 
               dpi=150, bbox_inches="tight", 
               facecolor='white', edgecolor='none')

    plt.show()
    
    return fig


# %% 创建可视化
# 创建没有平均线的可视化
fig = plot_enhanced_multi_model_visualization(
    version_equal_weighted=version_equal_weighted,
    all_model_profits=all_model_profits,
    all_metrics=all_metrics,
    cross_sectional_volatility=cross_sectional_volatility,
    cross_sectional_kurt=cross_sectional_kurt,
    cross_sectional_top_bottom_diff=cross_sectional_top_bottom_diff,
    twap_list=twap_list,
    compare_name=compare_name,
    save_dir=save_dir,
    show_average=False  # 设置为False隐藏平均线
)

# 可选：创建带平均线的第二个可视化
# 取消下面代码的注释以生成带平均线的另一个图表
"""
fig_with_avg = plot_enhanced_multi_model_visualization(
    version_equal_weighted=version_equal_weighted,
    all_model_profits=all_model_profits,
    all_metrics=all_metrics,
    cross_sectional_volatility=cross_sectional_volatility,
    cross_sectional_kurt=cross_sectional_kurt,
    cross_sectional_top_bottom_diff=cross_sectional_top_bottom_diff,
    twap_list=twap_list,
    compare_name=f"{compare_name}_with_average",
    save_dir=save_dir,
    show_average=True  # 设置为True显示平均线
)
"""