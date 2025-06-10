# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 11:23:20 2024

@author: Xintang Zheng

星星: ★ ☆ ✪ ✩ 🌟 ⭐ ✨ 🌠 💫 ⭐️
勾勾叉叉: ✓ ✔ ✕ ✖ ✅ ❎
报警啦: ⚠ ⓘ ℹ ☣
箭头: ➔ ➜ ➙ ➤ ➥ ↩ ↪
emoji: 🔔 ⏳ ⏰ 🔒 🔓 🛑 🚫 ❗ ❓ ❌ ⭕ 🚀 🔥 💧 💡 🎵 🎶 🧭 📅 🤔 🧮 🔢 📊 📈 📉 🧠 📝

"""
# %%
import sys
from pathlib import Path
from datetime import datetime, timedelta
from functools import partial
import toml
import traceback


# %% add sys path
file_path = Path(__file__).resolve()
file_dir = file_path.parents[0]
project_dir = file_path.parents[1]
sys.path.append(str(project_dir))


# %%
from scripts.rolling_fit_pred_backtest import main
from synthesis.merge_models import MergeModel
from backtest.backtester import backtest
from utils.logutils import FishStyleLogger
from utils.dirutils import load_path_config


# %%
class DailyUpdate:
    
    def __init__(self, task_name):
        path_config = load_path_config(project_dir)
        param_dir = Path(path_config['param']) / 'daily_update'
        self.param = toml.load(param_dir / f'{task_name}.toml')
        self.log = FishStyleLogger()
        self.msgs = []
        
    def _predict(self, date):
        model_name_list = self.param['model_name_list']
        mode = self.param['mode']
        n_workers = self.param['n_workers']
        
        puntil = (datetime.strptime(date, '%Y%m%d') + timedelta(days=1)).strftime('%Y%m%d')
        predict_func = partial(main, puntil=puntil, mode=mode, n_workers=n_workers)
        for model_name in model_name_list:
            self.log.info(f'Predict Start: {model_name}')
            predict_func(test_name=model_name)
            self.log.info(f'Predict Finished: {model_name}')
            
    def _merge(self):
        merge_model_list = self.param['merge_model_list']
        
        for model_name in merge_model_list:
            self.log.info(f'Merge Start: {model_name}')
            mm = MergeModel(model_name)
            mm.run()
            self.log.info(f'Merge Finished: {model_name}')
            
    def _backtest(self):
        backtest_list = self.param['backtest_list']
        for backtest_pair in backtest_list:
            self.log.info(f'Backtest Start: {backtest_pair}')
            backtest(**backtest_pair)
            self.log.info(f'Backtest Finished: {backtest_pair}')
            
    def run(self, date):
        try:
            self._predict(date)
            self._merge()
            self._backtest()
        except:
            e_format = traceback.format_exc()
            self.log.exception()
            self.msgs.append({
                    'type': 'msg',
                    'content': {
                        'level': 'error',
                        'title': 'daily update model error',
                        'msg': e_format,
                        }
                    })