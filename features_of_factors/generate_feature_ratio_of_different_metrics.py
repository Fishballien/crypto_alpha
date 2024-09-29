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
import numpy as np
import yaml
import matplotlib.pyplot as plt
import pickle


# %% add sys path
file_path = Path(__file__).resolve()
file_dir = file_path.parents[0]
project_dir = file_path.parents[1]
sys.path.append(str(project_dir))


# %%
pool_name = 'pool_240827'
metric_1 = 'sharpe_ratio_0'
metric_2 = 'hsr'
periods = ['pred15days', 'pred1month']


# %% dir
path_config_path = project_dir / '.path_config.yaml'
with path_config_path.open('r') as file:
    path_config = yaml.safe_load(file)
processed_data_dir = Path(path_config['processed_data'])
result_dir = Path(path_config['result'])

feature_dir = processed_data_dir / 'features_of_factors' / pool_name


# %%
with open(feature_dir / 'features_of_factors.pkl', 'rb') as f:
    features = pickle.load(f)
    
    
# %%
for p in periods:
    # m1 = pd.read_parquet(feature_dir / f'{p}_{metric_1}.parquet')
    # m2 = pd.read_parquet(feature_dir / f'{p}_{metric_2}.parquet')
    # ratio = m1 / m2
    ratio_name = f'r_{p}_{metric_1}_div_{metric_2}'
    # ratio.to_parquet(feature_dir / f'{ratio_name}.parquet')
    features.remove(ratio_name)
    
with open(feature_dir / 'features_of_factors.pkl', 'wb') as f:
    pickle.dump(features, f)