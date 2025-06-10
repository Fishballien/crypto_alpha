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
compare_name = 'backtest__p02_vs_p04_count_funding'
compare_dict = {
    # 'tf+zxt': {
    #     'model_name': 'merge_agg_241029_double3m_15d_73',
    #     'backtest_name': 'to_00125_maxmulti_2_mm_03_pf_001',
    #     },
    # 'cgy+tf+zxt': {
    #     'model_name': 'merge_agg_241109_double3m_15d_73',
    #     'backtest_name': 'to_00125_maxmulti_2_mm_03_pf_001',
    #     },
    # 'm1': {
    #     'model_name': 'merge_agg_241113_double3m_15d_73',
    #     'backtest_name': 'to_00125_maxmulti_2_mm_03_pf_001',
    #     },
    # 'cgy+tf': {
    #     'model_name': 'merge_agg_241114_tf_cgy_double3m_15d_73',
    #     'backtest_name': 'to_00125_maxmulti_2_mm_03_pf_001',
    #     },
    # 'm2': {
    #     'model_name': 'merge_agg_241114_zxt_cgy_double3m_15d_73',
    #     'backtest_name': 'to_00125_maxmulti_2_mm_03_pf_001',
    #     },
    # 'cgy': {
    #     'model_name': 'merge_agg_241214_cgy_double3m_15d_73',
    #     'backtest_name': 'to_00125_maxmulti_2_mm_03_pf_001',
    #     },
    # 'cgy+zxt': {
    #     'model_name': 'merge_agg_241214_cgy_zxt_double3m_15d_73',
    #     'backtest_name': 'to_00125_maxmulti_2_mm_03_pf_001',
    #     },
    'p0.2': {
        'model_name': 'merge_agg_241227_cgy_zxt_double3m_15d_73',
        'backtest_name': 'to_00125_maxmulti_2_mm_03_pf_001_count_funding',
        },
    'p0.4': {
        'model_name': 'merge_agg_250127_cgy_zxt_double3m_15d_73',
        'backtest_name': 'to_00125_maxmulti_2_mm_03_pf_001_count_funding',
        },
    }

start_date = '20230701'
end_date = '20250307'

sp = 30
twap_list = ['twd30_sp30']


# %% dir
save_dir = analysis_dir / compare_name
save_dir.mkdir(parents=True, exist_ok=True)


# %%
profd_dict = {}

for name, info in compare_dict.items():
    model_name = info['model_name']
    backtest_name = info['backtest_name']
    path = model_dir / model_name / 'backtest' / backtest_name / f'profit_{model_name}__{backtest_name}.parquet'
    profit = pd.read_parquet(path)
    profit.index = pd.to_datetime(profit.index)
    profit = profit.loc[start_date:end_date]
    profd = profit.resample('1d').sum()
    profd['return'] = profd[f'raw_rtn_twd30_sp{sp}'] + profd['fee'] + profd['funding']
    profd_dict[name] = profd
    
    
metrics = {name: get_general_return_metrics(profd.loc[:, 'return'].values)
                for name, profd in profd_dict.items()}
    

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

    
# %% Plot Visualization with Enhanced Aesthetics
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.gridspec import GridSpec
import matplotlib.dates as mdates
import seaborn as sns

def plot_enhanced_visualization(profd_dict, metrics, cross_sectional_volatility, 
                               cross_sectional_kurt, cross_sectional_top_bottom_diff,
                               twap_list, compare_name, save_dir, 
                               show_average=False):  # Added parameter for average line
    """
    Create enhanced visualization with optional average line
    
    Parameters:
    -----------
    profd_dict : dict
        Dictionary containing profit dataframes
    metrics : dict
        Dictionary containing performance metrics
    cross_sectional_volatility : Series
        Time series of cross-sectional volatility
    cross_sectional_kurt : Series
        Time series of cross-sectional kurtosis
    cross_sectional_top_bottom_diff : Series
        Time series of cross-sectional top-bottom difference
    twap_list : list
        List of TWAP names
    compare_name : str
        Name for the comparison plot
    save_dir : Path
        Directory to save the plot
    show_average : bool, optional
        Whether to show the average line (default: False)
    """

    # Set theme for better aesthetics
    plt.style.use('seaborn-v0_8-whitegrid')
    sns.set_context("notebook", font_scale=1.2)

    # Better color palette
    colors = sns.color_palette('viridis', len(profd_dict))
    accent_colors = sns.color_palette('Set2', 3)

    # Better fonts
    plt.rcParams['font.family'] = 'DejaVu Sans'
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['axes.labelsize'] = 12

    # Create figure with improved layout
    fig = plt.figure(figsize=(20, 16))
    gs = GridSpec(4, 1, figure=fig, height_ratios=[4, 1, 1, 1], hspace=0.5)

    # Main title with custom styling
    fig.suptitle(compare_name, fontsize=24, fontweight='bold', y=0.95)

    # Panel 1: Cumulative Returns
    ax0 = fig.add_subplot(gs[0])
    combined_returns = pd.DataFrame()

    # Plot with better line styling and annotations
    for idx, (name, profd) in enumerate(profd_dict.items()):
        return_text = f"{name}: Return: {metrics[name]['return']:.2%}, MaxDD: {metrics[name]['max_dd']:.2%}, Sharpe: {metrics[name]['sharpe_ratio']:.2f}"
        
        for twap_name in twap_list:
            combined_returns[f"{name}_{twap_name}"] = profd[f'raw_rtn_{twap_name}'] + profd['fee'] + profd['funding']
            
            # Main return line - thicker and more vibrant
            cum_returns = (profd[f'raw_rtn_{twap_name}'] + profd['fee'] + profd['funding']).cumsum()
            ax0.plot(cum_returns, label=return_text, linewidth=3, color=colors[idx], alpha=0.9)
            
            # Fee and funding as thinner, more transparent lines
            ax0.plot((profd['fee']).abs().cumsum(), linewidth=1.5, 
                    color=colors[idx], linestyle='--', alpha=0.5, 
                    label=f"{name}: Fee")
            ax0.plot((profd['funding']).abs().cumsum(), linewidth=1.5, 
                    color=colors[idx], linestyle=':', alpha=0.5,
                    label=f"{name}: Funding")
            
            # Add end point annotation
            last_date = profd.index[-1]
            last_value = cum_returns[-1]
            ax0.scatter(last_date, last_value, s=100, color=colors[idx], zorder=5, edgecolor='white')

    # Optional average line
    if show_average:
        avg_return = combined_returns.mean(axis=1)
        avg_metric = get_general_return_metrics(avg_return.values)
        avg_return_text = f"Average: Return: {avg_metric['return']:.2%}, MaxDD: {avg_metric['max_dd']:.2%}, Sharpe: {avg_metric['sharpe_ratio']:.2f}"
        avg_cum_return = avg_return.cumsum()

        ax0.plot(avg_cum_return, label=avg_return_text, 
                linewidth=4, color='black', alpha=0.8, zorder=10)

    # Add a shaded area for zero line reference
    ax0.axhline(y=0, color='grey', linestyle='-', alpha=0.3, linewidth=1)

    # Add grid with custom styling
    ax0.grid(True, linestyle=':', alpha=0.6)

    # Improve legend placement and styling
    ax0.legend(loc="upper left", fontsize=12, framealpha=0.9, 
            edgecolor='lightgrey', fancybox=True)

    # Better x-axis date formatting
    ax0.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax0.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.setp(ax0.xaxis.get_majorticklabels(), rotation=45, ha='right')

    ax0.set_title('Cumulative Returns', fontsize=18, pad=15)
    ax0.set_ylabel('Return', fontsize=14)

    # Add Panel 2: Cross-Sectional Volatility with improved styling
    ax2 = fig.add_subplot(gs[1], sharex=ax0)
    volatility_line = ax2.plot(cross_sectional_volatility.loc[profd_dict[list(profd_dict.keys())[0]].index], 
                            label='Cross-Sectional Volatility', 
                            color=accent_colors[0], linewidth=2)

    # Add shaded area under volatility line
    ax2.fill_between(cross_sectional_volatility.loc[profd_dict[list(profd_dict.keys())[0]].index].index, 
                    0, cross_sectional_volatility.loc[profd_dict[list(profd_dict.keys())[0]].index], 
                    color=accent_colors[0], alpha=0.2)

    ax2.grid(True, linestyle=':', alpha=0.6)
    ax2.legend(loc="upper left", fontsize=12, framealpha=0.9)
    ax2.set_ylabel('Volatility', fontsize=14)
    ax2.set_title('Market Volatility', fontsize=14, pad=10)

    # Add Panel 3: Cross-Sectional Kurtosis with improved styling
    ax3 = fig.add_subplot(gs[2], sharex=ax0)
    kurt_line = ax3.plot(cross_sectional_kurt.loc[profd_dict[list(profd_dict.keys())[0]].index], 
                        label='Cross-Sectional Kurtosis', 
                        color=accent_colors[1], linewidth=2)

    # Add shaded area under kurtosis line
    ax3.fill_between(cross_sectional_kurt.loc[profd_dict[list(profd_dict.keys())[0]].index].index, 
                    0, cross_sectional_kurt.loc[profd_dict[list(profd_dict.keys())[0]].index], 
                    color=accent_colors[1], alpha=0.2)

    ax3.grid(True, linestyle=':', alpha=0.6)
    ax3.legend(loc="upper left", fontsize=12, framealpha=0.9)
    ax3.set_ylabel('Kurtosis', fontsize=14)
    ax3.set_title('Distribution Kurtosis', fontsize=14, pad=10)

    # Add Panel 4: Cross-Sectional Top-Bottom Difference with improved styling
    ax4 = fig.add_subplot(gs[3], sharex=ax0)
    top_bottom_line = ax4.plot(cross_sectional_top_bottom_diff.loc[profd_dict[list(profd_dict.keys())[0]].index], 
                            label='Cross-Sectional Top-Bottom', 
                            color=accent_colors[2], linewidth=2)

    # Add shaded area under top-bottom line
    ax4.fill_between(cross_sectional_top_bottom_diff.loc[profd_dict[list(profd_dict.keys())[0]].index].index, 
                    0, cross_sectional_top_bottom_diff.loc[profd_dict[list(profd_dict.keys())[0]].index], 
                    color=accent_colors[2], alpha=0.2)

    ax4.grid(True, linestyle=':', alpha=0.6)
    ax4.legend(loc="upper left", fontsize=12, framealpha=0.9)
    ax4.set_ylabel('Difference', fontsize=14)
    ax4.set_xlabel('Date', fontsize=14)
    ax4.set_title('Top-Bottom Spread', fontsize=14, pad=10)

    # Add annotations for important market events (customize as needed)
    # Example:
    # ax0.annotate('Important Event', xy=('2023-09-01', 0.1), xytext=('2023-08-15', 0.2),
    #             arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
    #             fontsize=12, bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="grey", alpha=0.8))

    # Adjust layout for better spacing
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust for title

    # Add watermark (optional)
    # fig.text(0.5, 0.5, 'CONFIDENTIAL', fontsize=60, color='gray',
    #          ha='center', va='center', alpha=0.2, rotation=45)

    # Add creation date/time footnote
    from datetime import datetime
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    fig.text(0.02, 0.01, f"Generated: {now}", fontsize=10, color='gray')

    # Save with higher resolution and better compression
    plt.savefig(save_dir / f"{compare_name}_enhanced.png", 
               dpi=150, bbox_inches="tight", 
               facecolor='white', edgecolor='none')

    plt.show()
    
    return fig
    

# %%
# Create visualization with average line turned OFF
fig = plot_enhanced_visualization(
    profd_dict=profd_dict, 
    metrics=metrics, 
    cross_sectional_volatility=cross_sectional_volatility, 
    cross_sectional_kurt=cross_sectional_kurt, 
    cross_sectional_top_bottom_diff=cross_sectional_top_bottom_diff,
    twap_list=twap_list, 
    compare_name=compare_name, 
    save_dir=save_dir,
    show_average=False  # Set to False to hide the average line
)

# Optional: To create a second visualization with average line turned ON
# Uncomment the below code to generate another plot with the average line
"""
fig_with_avg = plot_enhanced_visualization(
    profd_dict=profd_dict, 
    metrics=metrics, 
    cross_sectional_volatility=cross_sectional_volatility, 
    cross_sectional_kurt=cross_sectional_kurt, 
    cross_sectional_top_bottom_diff=cross_sectional_top_bottom_diff,
    twap_list=twap_list, 
    compare_name=f"{compare_name}_with_average", 
    save_dir=save_dir,
    show_average=True  # Set to True to show the average line
)
"""