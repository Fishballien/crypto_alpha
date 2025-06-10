# -*- coding: utf-8 -*-
"""
Created on Mon Apr  7 14:39:41 2025

@author: Xintang Zheng

星星: ★ ☆ ✪ ✩ 🌟 ⭐ ✨ 🌠 💫 ⭐️
勾勾叉叉: ✓ ✔ ✕ ✖ ✅ ❎
报警啦: ⚠ ⓘ ℹ ☣
箭头: ➔ ➜ ➙ ➤ ➥ ↩ ↪
emoji: 🔔 ⏳ ⏰ 🔒 🔓 🛑 🚫 ❗ ❓ ❌ ⭕ 🚀 🔥 💧 💡 🎵 🎶 🧭 📅 🤔 🧮 🔢 📊 📈 📉 🧠 📝

"""
# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils.datautils import align_columns

# Load data
pred_path = r'D:/crypto/multi_factor/factor_test_by_alpha/results/model/merge_agg_250318_double3m_15d_73/predict/predict_merge_agg_250318_double3m_15d_73.parquet'
twap_path = 'D:/mnt/Data/Crypto/ProcessedData/updated_twap/twd30_sp30.parquet'

predict_res = pd.read_parquet(pred_path)
twap = pd.read_parquet(twap_path)

# Mask missing values
to_mask = twap.isna() | predict_res.isna()
to_mask = to_mask.astype(bool)
main_columns = twap.columns
predict_res = align_columns(main_columns, predict_res)
predict_res = predict_res.mask(to_mask)

def calculate_rank(df):
    """Calculate percentile rank, adjusting for column count."""
    return df.rank(axis=1, pct=True).sub(0.5 / df.count(axis=1), axis=0).replace([np.inf, -np.inf], np.nan)

def calculate_turnover(rank, top_pct=0.1, bottom_pct=0.1, rebalance_freq='1D'):
    """
    Calculate turnover based on top and bottom percentile quantiles.
    
    Parameters:
    -----------
    rank : pd.DataFrame
        Dataframe with ranked values (0-1 scale), with datetime index and assets in columns
    top_pct : float, default=0.1
        Percentage for top quantile (e.g., 0.1 for top 10%)
    bottom_pct : float, default=0.1
        Percentage for bottom quantile (e.g., 0.1 for bottom 10%)
    rebalance_freq : str, default='1D'
        Frequency to rebalance portfolio, e.g., '1D' for daily
        
    Returns:
    --------
    tuple
        (hsrd, avg_hsrd) - daily turnover series and average daily turnover
    """
    
    # Create position dataframe based on rank
    # Long positions for top percentile
    top_positions = (rank > (1 - top_pct)).astype(float)
    
    # Short positions for bottom percentile
    bottom_positions = (rank <= bottom_pct).astype(float)
    
    # Calculate weights for each group (equal weighting within groups)
    # For top group, divide by count to get equal weights summing to 1
    top_count = top_positions.sum(axis=1)
    top_weights = top_positions.div(top_count, axis=0).fillna(0)
    
    # For bottom group, divide by count to get equal weights summing to 1
    bottom_count = bottom_positions.sum(axis=1)
    bottom_weights = bottom_positions.div(bottom_count, axis=0).fillna(0)
    
    # Scale weights to ensure absolute sum is 1 and sum is 0
    # Top group gets positive weights of 0.5 total
    scaled_top = top_weights * 0.5
    
    # Bottom group gets negative weights of -0.5 total
    scaled_bottom = bottom_weights * (-0.5)
    
    # Combine to get final positions
    ps = scaled_top + scaled_bottom
    
    # Calculate turnover
    hsr = ((ps - ps.shift(1)).abs().sum(axis=1) / 
           (2 * ps.shift(1).abs().sum(axis=1))).replace([np.inf, -np.inf], np.nan)
    
    # Calculate daily turnover
    hsrd = hsr.resample('1d').sum()

    # Calculate average daily turnover
    avg_hsrd = hsrd.mean()
    
    return hsrd, avg_hsrd

# Calculate the original rank and turnover
rank_original = calculate_rank(predict_res)
hsrd_original, avg_hsrd_original = calculate_turnover(rank_original)

# Moving average periods to analyze
ma_mode = 'ewma'
ma_periods = [8, 16, 24, 48, 144, 240]

# Store results
results = {'Original': avg_hsrd_original}
all_hsrd = {'Original': hsrd_original}

# Calculate turnover for each MA period
for period in ma_periods:
    # Calculate moving average
    if ma_mode == 'ma':
        predict_ma = predict_res.rolling(window=period, min_periods=1).mean()
    elif ma_mode == 'ewma':
        predict_ma = predict_res.ewm(span=period, adjust=False).mean()
    
    # Calculate rank for MA prediction
    rank_ma = calculate_rank(predict_ma)
    
    # Calculate turnover
    hsrd_ma, avg_hsrd_ma = calculate_turnover(rank_ma)
    
    # Store results
    results[f'MA-{period}'] = avg_hsrd_ma
    all_hsrd[f'MA-{period}'] = hsrd_ma

# Convert results to DataFrame for easier plotting
results_df = pd.DataFrame.from_dict(results, orient='index', columns=['Avg_Daily_Turnover'])

# Add period as a numeric column for plotting
results_df['Period'] = [0] + ma_periods  # 0 for original

# Create a scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(results_df['Period'], results_df['Avg_Daily_Turnover'], s=100, alpha=0.7)

# Add labels for each point
for idx, row in results_df.iterrows():
    label = idx
    plt.annotate(label, (row['Period'], row['Avg_Daily_Turnover']), 
                 xytext=(5, 5), textcoords='offset points')

# Connect points with a line
plt.plot(results_df['Period'], results_df['Avg_Daily_Turnover'], 'k--', alpha=0.5)

# Add labels and title
plt.xlabel('Moving Average Period (days)', fontsize=12)
plt.ylabel('Average Daily Turnover', fontsize=12)
plt.title(f'Effect of Smoothing (Method: {ma_mode}) on Portfolio Turnover', fontsize=14)


# Set y-axis to logarithmic scale
plt.yscale('log')

# Add grid
plt.grid(True, linestyle='--', alpha=0.7)

# Ensure x-axis starts from 0
plt.xlim(left=-1)

# Show the plot
plt.tight_layout()
plt.show()

# Create a more detailed DataFrame with statistics
detailed_results = pd.DataFrame({
    'MA_Period': [0] + ma_periods,  # 0 for original
    'Avg_Daily_Turnover': [results[k] for k in ['Original'] + [f'MA-{p}' for p in ma_periods]],
    'Max_Daily_Turnover': [all_hsrd[k].max() for k in ['Original'] + [f'MA-{p}' for p in ma_periods]],
    'Min_Daily_Turnover': [all_hsrd[k].min() for k in ['Original'] + [f'MA-{p}' for p in ma_periods]],
    'Std_Daily_Turnover': [all_hsrd[k].std() for k in ['Original'] + [f'MA-{p}' for p in ma_periods]]
})

# Print the detailed results
print(detailed_results)