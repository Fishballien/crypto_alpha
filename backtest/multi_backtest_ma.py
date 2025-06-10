# -*- coding: utf-8 -*-
"""
Created on Tue Mar 11 11:28:54 2025

@author: Xintang Zheng

星星: ★ ☆ ✪ ✩ 🌟 ⭐ ✨ 🌠 💫 ⭐️
勾勾叉叉: ✓ ✔ ✕ ✖ ✅ ❎
报警啦: ⚠ ⓘ ℹ ☣
箭头: ➔ ➜ ➙ ➤ ➥ ↩ ↪
emoji: 🔔 ⏳ ⏰ 🔒 🔓 🛑 🚫 ❗ ❓ ❌ ⭕ 🚀 🔥 💧 💡 🎵 🎶 🧭 📅 🤔 🧮 🔢 📊 📈 📉 🧠 📝

"""
# %% imports
import sys
import toml
from pathlib import Path
import concurrent.futures
import os
from tqdm import tqdm


# %% add sys path
file_path = Path(__file__).resolve()
file_dir = file_path.parents[0]
project_dir = file_path.parents[1]
sys.path.append(str(project_dir))


# %% import self-defined
from utils.dirutils import load_path_config
from backtest.backtester import backtest


# %%
model_name = 'merge_agg_250318_double3m_15d_73'
org_backtest_name = 'ma_simple'
ma_method_list = ['ma', 'ewma']
ma_window_list = [8, 24, 48, 144, 240, 336]
pct_list = [0.075, 0.1, 0.125, 0.15]


# %%
path_config = load_path_config(project_dir)
param_dir = Path(path_config['param']) / 'backtest'
backtest_param = toml.load(param_dir / f'{org_backtest_name}.toml')


# %%
task_name_list = []
for ma_method in ma_method_list:
    for ma_window in ma_window_list:
        for pct in pct_list:
            param = {k: v for k, v in backtest_param.items()}
            param['ma_window'] = ma_window
            param['top_pct'] = pct
            param['bottom_pct'] = pct
            param_name = f'{org_backtest_name}-{ma_method}{ma_window}_pct{pct}'
            with open(param_dir / f"{param_name}.toml", "w") as f:
                toml.dump(param, f)
            task_name_list.append(param_name)


# %%
def process_task(model_name, task_name):
    # 在子进程中禁用tqdm输出
    os.environ['TQDM_DISABLE'] = '1'
    try:
        return backtest(model_name, task_name)
    except Exception as e:
        return f"Error in {task_name}: {str(e)}"

# 使用ProcessPoolExecutor进行并行处理，同时用tqdm显示总进度
with concurrent.futures.ProcessPoolExecutor(max_workers=10) as executor:
    # 提交所有任务
    futures = {executor.submit(process_task, model_name, task_name): task_name 
               for task_name in task_name_list}
    
    # 用tqdm显示总体进度
    total_tasks = len(task_name_list)
    results = {}
    
    # 使用tqdm显示总进度条
    for future in tqdm(concurrent.futures.as_completed(futures), 
                       total=total_tasks, 
                       desc="Total Progress", 
                       unit="task"):
        task_name = futures[future]
        try:
            result = future.result()
            results[task_name] = result
        except Exception as exc:
            print(f"Task {task_name} generated an exception: {exc}")
            results[task_name] = None