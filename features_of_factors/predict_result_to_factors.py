# -*- coding: utf-8 -*-
"""
Created on Tue Aug 27 10:10:14 2024

@author: Xintang Zheng

星星: ★ ☆ ✪ ✩ 🌟 ⭐ ✨ 🌠 💫 ⭐️
勾勾叉叉: ✓ ✔ ✕ ✖ ✅ ❎
报警啦: ⚠ ⓘ ℹ ☣
箭头: ➔ ➜ ➙ ➤ ➥ ↩ ↪
emoji: 🔔 ⏳ ⏰ 🔒 🔓 🛑 🚫 ❗ ❓ ❌ ⭕ 🚀 🔥 💧 💡 🎵 🎶 🧭 📅 🤔 🧮 🔢 📊 📈 📉 🧠 📝

"""
# %%
import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import yaml
import matplotlib.pyplot as plt
import pickle
from datetime import datetime


# %% add sys path
file_path = Path(__file__).resolve()
file_dir = file_path.parents[0]
project_dir = file_path.parents[1]
sys.path.append(str(project_dir))


# %%
pool_name = 'pool_240820'
model_name = 'ridge_v0_pool_240820_1'
date = datetime(2024, 4, 1)


# %% dir
path_config_path = project_dir / '.path_config.yaml'
with path_config_path.open('r') as file:
    path_config = yaml.safe_load(file)
processed_data_dir = Path(path_config['processed_data'])
result_dir = Path(path_config['result'])

feature_dir = processed_data_dir / 'features_of_factors' / pool_name
predict_dir = result_dir / 'features_of_factors' / pool_name / 'model' / model_name / 'predict'


# %%
factor_mapping = pd.read_parquet(feature_dir / "factor_mapping.parquet")
predict = pd.read_parquet(predict_dir / f'predict_{model_name}.parquet')

predict.sort_index(axis=1, inplace=True)

latest_pred = predict[predict.index <= date].iloc[-1]
factor_eval = factor_mapping.copy()
factor_eval['predict'] = latest_pred
thresh = factor_eval['predict'].quantile(0.5)