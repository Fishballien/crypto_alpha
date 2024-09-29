# -*- coding: utf-8 -*-
"""
Created on Fri Aug 30 10:22:11 2024

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
import matplotlib.pyplot as plt


# %% add sys path
file_path = Path(__file__).resolve()
file_dir = file_path.parents[0]
project_dir = file_path.parents[1]
sys.path.append(str(project_dir))


# %%
pool_name = 'pool_240827'
metric_1 = 'pred1month_sharpe_ratio_0'
metric_2 = 'pred1month_hsr'


# %% dir
path_config_path = project_dir / '.path_config.yaml'
with path_config_path.open('r') as file:
    path_config = yaml.safe_load(file)
processed_data_dir = Path(path_config['processed_data'])
result_dir = Path(path_config['result'])

feature_dir = processed_data_dir / 'features_of_factors' / pool_name
name = f'{metric_1}_vs_{metric_2}'
save_dir = result_dir / 'features_of_factors' / pool_name / 'analysis' / name
save_dir.mkdir(parents=True, exist_ok=True)

    
# %%
m1 = pd.read_parquet(feature_dir / f'{metric_1}.parquet')
m2 = pd.read_parquet(feature_dir / f'{metric_2}.parquet')

for i, idx in enumerate(m1.index):
    m1_arr = m1.loc[idx]
    m2_arr = m2.loc[idx]
    
    # Plotting the curve
    plt.figure(figsize=(10, 6))

    plt.scatter(m1_arr, m2_arr)
    
    plt.grid(True)
    plt.title(f'{idx}')
    plt.xlabel(metric_1)
    plt.ylabel(metric_2)
    plt.tight_layout()  # 自动调整子图参数以适应图形区域
    plt.savefig(save_dir / f'{i}.jpg')
    plt.show()

