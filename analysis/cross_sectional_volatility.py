# -*- coding: utf-8 -*-
"""
Created on Mon Aug 12 13:27:27 2024

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
from tqdm import tqdm
import matplotlib.pyplot as plt
from functools import partial
from datetime import timedelta


# %% add sys path
file_path = Path(__file__).resolve()
file_dir = file_path.parents[0]
project_dir = file_path.parents[1]
sys.path.append(str(project_dir))


# %% import self-defined
from utils.dirutils import load_path_config


# %%
path_config = load_path_config(project_dir)
processed_data_dir = Path(path_config['processed_data'])
result_dir = Path(path_config['result']) / 'model'
twap_data_dir = Path(path_config['twap_price'])


# %%
# load twap & calc rtn
curr_px_path = twap_data_dir / 'curr_price_sp240.parquet'
curr_price = pd.read_parquet(curr_px_path)


# %%
rtn = curr_price.pct_change(1).replace([np.inf, -np.inf], np.nan)
rtn_std = rtn.std(axis=1).resample('1d').mean()



plt.figure(figsize=(20, 6))
plt.plot(rtn_std[rtn_std.index >= '20230701'], label='Standard Deviation')
# plt.yscale('log')
plt.grid(True, which="both", linestyle='--', linewidth=0.5)
plt.xlabel('Index')
plt.ylabel('Standard Deviation')
plt.title('Standard Deviation')  # with Logarithmic Y-Axis
plt.legend()
plt.show()
