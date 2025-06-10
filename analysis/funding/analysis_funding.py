# -*- coding: utf-8 -*-
"""
Created on Thu May 15 17:35:41 2025

@author: Xintang Zheng

星星: ★ ☆ ✪ ✩ 🌟 ⭐ ✨ 🌠 💫 ⭐️
勾勾叉叉: ✓ ✔ ✕ ✖ ✅ ❎
报警啦: ⚠ ⓘ ℹ ☣
箭头: ➔ ➜ ➙ ➤ ➥ ↩ ↪
emoji: 🔔 ⏳ ⏰ 🔒 🔓 🛑 🚫 ❗ ❓ ❌ ⭕ 🚀 🔥 💧 💡 🎵 🎶 🧭 📅 🤔 🧮 🔢 📊 📈 📉 🧠 📝

"""
import pandas as pd
import numpy as np

path = 'D:/mnt/Data/Crypto/Funding/funding_rates_wide.parquet'
funding = pd.read_parquet(path).astype(np.float64).fillna(0)
path1 = 'D:/mnt/Data/Crypto/Funding/binance_usd_funding_rates_30min.parquet'
funding_est = pd.read_parquet(path1)
# funding['BTCUSDT'].plot()