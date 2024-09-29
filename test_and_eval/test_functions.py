# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 10:07:13 2024

@author: Xintang Zheng

星星: ★ ☆ ✪ ✩ 🌟 ⭐ ✨ 🌠 💫 ⭐️
勾勾叉叉: ✓ ✔ ✕ ✖ ✅ ❎
报警啦: ⚠ ⓘ ℹ ☣
箭头: ➔ ➜ ➙ ➤ ➥ ↩ ↪
emoji: 🔔 ⏳ ⏰ 🔒 🔓 🛑 🚫 ❗ ❓ ❌ ⭕ 🚀 🔥 💧 💡 🎵 🎶 🧭 📅 🤔 🧮 🔢 📊 📈 📉 🧠 📝

"""
# %% imports
import numpy as np


# %%
def calc_rank(factor):
    factor_rank = factor.rank(axis=1, pct=True
                              ).sub(0.5 / factor.count(axis=1), axis=0
                                    ).replace([np.inf, -np.inf], np.nan
                                          ) #.fillna(0)
    fct_n_pct = 2 * (factor_rank - 0.5)
    return fct_n_pct
        

def calc_gp(fct_n_pct, rtn):
    gps = (fct_n_pct * rtn).mean(axis=1).shift(1).replace([0.0], np.nan)
    direction = 1 if np.nansum(gps) > 0 else -1
    gps *= direction
    gpd = gps.resample('D').sum(min_count=1).fillna(0)
    return gps, gpd


def calc_hsr(fct_n_pct):
    ps = fct_n_pct
    hsr = ((ps - ps.shift(1)).abs().sum(axis=1) / (2 * ps.abs().sum(axis=1))).replace([np.inf, -np.inf], np.nan)
    return hsr


def calc_ic(factor, predict):
    ic = factor.corrwith(predict, axis=1, method='spearman'
                         ).replace([np.inf, -np.inf], np.nan).fillna(0)
    return ic