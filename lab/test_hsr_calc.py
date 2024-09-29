# -*- coding: utf-8 -*-
"""
Created on Thu Aug 15 17:08:47 2024

@author: Xintang Zheng

星星: ★ ☆ ✪ ✩ 🌟 ⭐ ✨ 🌠 💫 ⭐️
勾勾叉叉: ✓ ✔ ✕ ✖ ✅ ❎
报警啦: ⚠ ⓘ ℹ ☣
箭头: ➔ ➜ ➙ ➤ ➥ ↩ ↪
emoji: 🔔 ⏳ ⏰ 🔒 🔓 🛑 🚫 ❗ ❓ ❌ ⭕ 🚀 🔥 💧 💡 🎵 🎶 🧭 📅 🤔 🧮 🔢 📊 📈 📉 🧠 📝

"""
import pandas as pd
import numpy as np
from pathlib import Path

pp_by_sp = 1
path = Path(r'D:/crypto/multi_factor/factor_test_by_alpha/debug/funding_filled_mmt12h_ma.parquet')


factor_processed = pd.read_parquet(path)

factor_rank = factor_processed.rank(axis=1, pct=True
                                    ).sub(0.5 / factor_processed.count(axis=1), axis=0
                                          ).replace([np.inf, -np.inf], np.nan
                                                    ) #.fillna(0)
fct_n_pct = 2 * (factor_rank - 0.5)

ps = fct_n_pct #.mask(to_mask)
hsr = ((ps - ps.shift(pp_by_sp)).abs().sum(axis=1) / (2 * ps.shift(pp_by_sp).abs().sum(axis=1))).replace([np.inf, -np.inf], np.nan)
hsrm = hsr.resample('M').mean()


# 发现1：funding rate中性化去极值后，可能所有值都一样（算不出std），会导致中性化后都是0