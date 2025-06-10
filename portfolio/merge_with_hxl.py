# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 16:36:27 2025

@author: Xintang Zheng

星星: ★ ☆ ✪ ✩ 🌟 ⭐ ✨ 🌠 💫 ⭐️
勾勾叉叉: ✓ ✔ ✕ ✖ ✅ ❎
报警啦: ⚠ ⓘ ℹ ☣
箭头: ➔ ➜ ➙ ➤ ➥ ↩ ↪
emoji: 🔔 ⏳ ⏰ 🔒 🔓 🛑 🚫 ❗ ❓ ❌ ⭕ 🚀 🔥 💧 💡 🎵 🎶 🧭 📅 🤔 🧮 🔢 📊 📈 📉 🧠 📝

"""
# %% imports
import pandas as pd
import matplotlib.pyplot as plt


# %%
model_name = 'merge_agg_241227_cgy_zxt_double3m_15d_73'
hxl_name = 'cc_cta_v2_ProfitDaily_20250306'
alpha_pft_path = f'D:/crypto/multi_factor/factor_test_by_alpha/results/model/{model_name}/backtest/to_00125_maxmulti_2_mm_03_pf_001/profit_{model_name}__to_00125_maxmulti_2_mm_03_pf_001.parquet'
cta_btc_pft_path = f'D:/crypto/multi_factor/factor_test_by_alpha/cta_fr_hxl/{hxl_name}.csv'


# %%
cta_btc_pft = pd.read_csv(cta_btc_pft_path)
cta_btc_pft['time'] = pd.to_datetime(cta_btc_pft['Unnamed: 0']) # data_ts
cta_btc_pft.set_index('time', inplace=True)
cta_btc_pft.index = cta_btc_pft.index.tz_convert(None)
cta_btc_pft['return'].cumsum().plot() # cc cta profit daily


# %%
alpha_pft = pd.read_parquet(alpha_pft_path)
profd = alpha_pft.resample('1d').sum()
profd['return'] = profd['raw_rtn_twd30_sp30'] + profd['fee']


# %%
# Align dates from 2023-07-01 to min(max dates of both series)
start_date = '2023-07-01'
end_date = min(cta_btc_pft.index.max(), profd.index.max())
aligned_cta_btc = cta_btc_pft.loc[start_date:end_date]
aligned_profd = profd.loc[start_date:end_date]

# Calculate Sharpe Ratio function
def calculate_sharpe(series):
    return series.mean() / series.std() * (365 ** 0.5)  # Annualized Sharpe assuming daily data

# Calculate individual Sharpe Ratios
cta_btc_sharpe = calculate_sharpe(aligned_cta_btc['return'])
alpha_sharpe = calculate_sharpe(aligned_profd['return'])

# Calculate equal-weighted combined return
combined_return = (aligned_cta_btc['return'] + aligned_profd['return']) / 2
combined_cumsum = combined_return.cumsum()
combined_sharpe = calculate_sharpe(combined_return)

# Plot cumulative returns with Sharpe Ratios
plt.figure(figsize=(12, 8))
aligned_cta_btc['return'].cumsum().plot(label=f'CTA BTC&ETH (Sharpe: {cta_btc_sharpe:.2f})')
aligned_profd['return'].cumsum().plot(label=f'Alpha Profit (Sharpe: {alpha_sharpe:.2f})')
combined_cumsum.plot(label=f'Equal-Weighted Combined (Sharpe: {combined_sharpe:.2f})')

plt.title('CTA BTC&ETH vs Alpha Profit vs Combined Cumulative Returns')
plt.xlabel('Datetime')
plt.ylabel('Cumulative Returns')
plt.legend()
plt.grid(':')
plt.tight_layout()
plt.show()




