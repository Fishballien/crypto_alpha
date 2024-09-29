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


# %%
pool_name = 'pool_240827'

rolling_params = {
    'fstart': datetime(2021, 1, 1),
    'pstart': datetime(2023, 7, 1),
    'puntil': datetime(2024, 4, 30),
    'rrule_kwargs': {"freq": "M", "interval": 1, "bymonthday": [1, 16]},
    'window_kwargs': {'months': 24},
    'end_by': 'time', 
    }


# %% dir
path_config_path = project_dir / '.path_config.yaml'
with path_config_path.open('r') as file:
    path_config = yaml.safe_load(file)
processed_data_dir = Path(path_config['processed_data'])
result_dir = Path(path_config['result'])

feature_dir = processed_data_dir / 'features_of_factors' / pool_name
test_dir = result_dir / 'features_of_factors' / pool_name / 'test'
data_dir = test_dir / 'data'
eval_dir = result_dir / 'features_of_factors' / pool_name / 'eval'
eval_dir.mkdir(parents=True, exist_ok=True)


# %%
with open(feature_dir / 'features_of_factors.pkl', 'rb') as f:
    features = pickle.load(f)


# %% load feature
rolling = RollingPeriods(**rolling_params)
fit_periods = rolling.fit_periods

for (date_start, date_end) in tqdm(fit_periods, desc='eval'):
    period_name = period_shortcut(date_start, date_end)
    res_list = []
    for i_f, feature_name in enumerate(features):
        res = {'pool_name': pool_name, 'feature_name': feature_name}
        ic = pd.read_parquet(data_dir / f'{feature_name}.parquet')
        ic = ic[(ic.index >= date_start) & (ic.index < date_end)]
        direction = 1 if ic['pred1month_return_annualized_0'].mean() > 0 else -1
        res.update({'direction': direction})
        # print(date_end, ic)
        for pred_target in ic.columns:
            res.update({pred_target: (ic[pred_target] * direction).mean()})
        res_list.append(res)
    eval_period = pd.DataFrame(res_list)
    eval_period.to_csv(eval_dir / f'feature_eval_{period_name}.csv', index=None)