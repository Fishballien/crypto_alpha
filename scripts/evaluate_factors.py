# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 18:00:33 2024

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
from datetime import datetime
import argparse
import warnings
warnings.filterwarnings("ignore")


# %% add sys path
file_path = Path(__file__).resolve()
file_dir = file_path.parents[0]
project_dir = file_path.parents[1]
sys.path.append(str(project_dir))


# %% import self-defined
from test_and_eval.factor_evaluation import FactorEvaluation


# %% main
def main():
    '''read args'''
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--eval_name', type=str, help='eval_name')
    parser.add_argument('-sd', '--start_date', type=str, help='start_date')
    parser.add_argument('-ed', '--end_date', type=str, default=None, help='end_date')
    parser.add_argument('-dsd', '--data_start_date', type=str, default=None, help='data_start_date')
    parser.add_argument('-ded', '--data_end_date', type=str, default=None, help='data_end_date')
    parser.add_argument('-wkr', '--n_workers', type=int, default=1, help='n_workers')
    args = parser.parse_args()
    eval_name, start_date, end_date = args.eval_name, args.start_date, args.end_date
    data_start_date, data_end_date = args.data_start_date, args.data_end_date
    n_workers = args.n_workers
    
    # trans date
    date_start = datetime.strptime(start_date, '%Y%m%d')
    date_end = datetime.strptime(end_date, '%Y%m%d')
    data_start_date = datetime.strptime(data_start_date, '%Y%m%d') if data_start_date is not None else None
    data_end_date = datetime.strptime(data_end_date, '%Y%m%d') if data_end_date is not None else None
    
    # run
    fe = FactorEvaluation(eval_name, n_workers=n_workers)
    fe.eval_one_period(date_start, date_end, data_start_date, data_end_date)
    
        
# %%
if __name__=='__main__':
    main()