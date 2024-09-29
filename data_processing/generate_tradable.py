# -*- coding: utf-8 -*-
"""
Created on Fri Aug 16 14:58:55 2024

@author: Xintang Zheng

星星: ★ ☆ ✪ ✩ 🌟 ⭐ ✨ 🌠 💫 ⭐️
勾勾叉叉: ✓ ✔ ✕ ✖ ✅ ❎
报警啦: ⚠ ⓘ ℹ ☣
箭头: ➔ ➜ ➙ ➤ ➥ ↩ ↪
emoji: 🔔 ⏳ ⏰ 🔒 🔓 🛑 🚫 ❗ ❓ ❌ ⭕ 🚀 🔥 💧 💡 🎵 🎶 🧭 📅 🤔 🧮 🔢 📊 📈 📉 🧠 📝

"""
# %% imports
import pandas as pd
import pickle
from datetime import datetime


# %%
tardis_path = r'D:\crypto\data\test local\binance-futures_exchange_details.pkl'
binance_path = r'D:/crypto/data/binance_sample/symbol_info_usd.pkl'
combine_path = r'D:\mnt\Data\xintang\pool\binance_usd_multi\combined_symbol_on_board_info.pkl'


# %%
with open(tardis_path, 'rb') as f:
    tardis_info = pickle.load(f)
    
    
with open(binance_path, 'rb') as f:
    binance_info = dict(pickle.load(f))
    
    
# %%
all_symbols = list(set(list(tardis_info.keys()) + list(binance_info.keys())))
all_symbols = [s for s in all_symbols if s.endswith('USDT')]

combined_info = {}

for symbol in all_symbols:
    tardis_symbol_info = tardis_info.get(symbol)
    if tardis_symbol_info:
        tardis_avlbsince = tardis_symbol_info['availableSince']
    else:
        tardis_avlbsince = None
    binance_onboard = binance_info.get(symbol)
    
    tardis_avlb_date = datetime.strptime(tardis_avlbsince, '%Y-%m-%dT%H:%M:%S.%fZ').date()
    
    if tardis_avlbsince is not None and binance_onboard is not None:
        # 计算两个日期之间的差异
        date_diff = tardis_avlb_date - binance_onboard
        
        # 判断差异超过一天且币安时间晚于tardis时间，说明可能是二次上市
        if abs(date_diff.days) > 1 and binance_onboard > tardis_avlb_date:
            # 记录两个日期
            onboard_dates = [binance_onboard, tardis_avlb_date]
        else:
            # 否则按tardis时间为准，记录一个日期，时间为零点
            onboard_dates = [datetime.combine(tardis_avlb_date, datetime.min.time())]
    elif tardis_avlbsince is not None:
        onboard_dates = [tardis_avlb_date]
    elif binance_onboard is not None:
        onboard_dates = [binance_onboard]
        
    combined_info[symbol] = onboard_dates
    
with open(combine_path, 'wb') as f:
    pickle.dump(combined_info, f)