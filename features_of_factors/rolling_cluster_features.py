# -*- coding: utf-8 -*-
"""
Created on Fri Aug 23 16:55:52 2024

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
import yaml
import pickle
from datetime import datetime
from tqdm import tqdm


# %% add sys path
file_path = Path(__file__).resolve()
file_dir = file_path.parents[0]
project_dir = file_path.parents[1]
sys.path.append(str(project_dir))


# %%
from utils.timeutils import RollingPeriods, period_shortcut
from cluster_features import Cluster


# %%
cluster_name = 'pool_240827_1'

rolling_params = {
    'fstart': datetime(2021, 1, 1),
    'pstart': datetime(2023, 7, 1),
    'puntil': datetime(2024, 4, 30),
    'rrule_kwargs': {"freq": "M", "interval": 1, "bymonthday": [1, 16]},
    'window_kwargs': {'months': 24},
    'end_by': 'time', 
    }


# %% cluster
rolling = RollingPeriods(**rolling_params)
fit_periods = rolling.fit_periods
cluster = Cluster(cluster_name)

for (date_start, date_end) in tqdm(fit_periods, desc='cluster'):
    cluster.cluster_one_period(date_start, date_end)