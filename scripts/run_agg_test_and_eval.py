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
# %%
'''
process_name + factor_name & root_dir 找到因子路径
                           & tag_name 找到测试路径
'''
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
from test_and_eval.agg_test_and_eval import AggTestEval


# %% main
def main():
    '''read args'''
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', '--agg_eval_name', type=str, help='agg_eval_name')
    parser.add_argument('-twkr', '--test_wkr', type=int, help='test_wkr')
    parser.add_argument('-ewkr', '--eval_wkr', type=int, help='eval_wkr')

    args = parser.parse_args()
    agg_eval_name, test_wkr, eval_wkr = args.agg_eval_name, args.test_wkr, args.eval_wkr
    
    agg_test_eval = AggTestEval(agg_eval_name, test_wkr, eval_wkr)
    agg_test_eval.run()
    

# %% main
if __name__ == "__main__":
    main()
