# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 14:59:39 2024

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
import argparse
import warnings
warnings.filterwarnings("ignore")


# %% add sys path
file_path = Path(__file__).resolve()
file_dir = file_path.parents[0]
project_dir = file_path.parents[1]
sys.path.append(str(project_dir))


# %% import self-defined
from features_of_factors.generate_features_of_factors import GenerateFeaturesOfFactors


# %% main
def main():
    '''read args'''
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--param_name', type=str, help='param_name')
    parser.add_argument('-dwkr', '--data_workers', type=int, help='data_workers')
    parser.add_argument('-cwkr', '--calc_workers', type=int, help='calc_workers')

    args = parser.parse_args()
    param_name = args.param_name
    data_workers = args.data_workers
    calc_workers = args.calc_workers
    
    mm = GenerateFeaturesOfFactors(param_name, data_workers, calc_workers)
    mm.generate_all()
    

# %% main
if __name__ == "__main__":
    main()
