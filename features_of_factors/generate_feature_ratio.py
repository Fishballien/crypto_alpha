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
pool_name = 'pool_240820'
metrics = ['gp_sharpe_ratio_0']
period_combos = [
    ('1month', '1year'),
    ('1month', '2years'),
    ('3months', '1year'),
    ('3months', '2years'),
    ('10days', '3months'),
    ('10days', '1year'),
    ]


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
for metric in metrics:
    for p_short, p_long in period_combos:
        f_short = pd.read_parquet(feature_dir / f'lb{p_short}_{metric}.parquet')
        f_long = pd.read_parquet(feature_dir / f'lb{p_long}_{metric}.parquet')
        ratio = f_short / f_long
        ratio_name = f'r_lb{p_short}_lb{p_long}_{metric}'
        ratio.to_parquet(feature_dir / f'{ratio_name}.parquet')
        features.append(ratio_name)
    
with open(feature_dir / 'features_of_factors.pkl', 'wb') as f:
    pickle.dump(features, f)