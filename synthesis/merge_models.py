# -*- coding: utf-8 -*-
"""
Created on Mon Aug  5 16:26:39 2024

@author: Xintang Zheng

星星: ★ ☆ ✪ ✩ 🌟 ⭐ ✨ 🌠 💫 ⭐️
勾勾叉叉: ✓ ✔ ✕ ✖ ✅ ❎
报警啦: ⚠ ⓘ ℹ ☣
箭头: ➔ ➜ ➙ ➤ ➥ ↩ ↪
emoji: 🔔 ⏳ ⏰ 🔒 🔓 🛑 🚫 ❗ ❓ ❌ ⭕ 🚀 🔥 💧 💡 🎵 🎶 🧭 📅 🤔 🧮 🔢 📊 📈 📉 🧠 📝

"""
# %% imports
import pandas as pd
import numpy as np
from pathlib import Path
import toml


from utils.dirutils import load_path_config
from test_and_eval.factor_tester_adaptive import FactorTest
from data_processing.feature_engineering import quantile_transform_with_nan


# %%
class MergeModel:
    
    def __init__(self, new_model_name):
        self.new_model_name = new_model_name
        self._load_public_paths()
        self._init_dir()
        self._load_params()
        
    def _load_public_paths(self):
        file_path = Path(__file__).resolve()
        project_dir = file_path.parents[1]
        path_config = load_path_config(project_dir)
        
        result_root_dir = Path(path_config['result'])
        self.model_dir = result_root_dir / 'model'
        self.param_dir = Path(path_config['param']) / 'model'
        
    def _init_dir(self):
        self.merge_save_dir = self.model_dir / self.new_model_name
        self.merge_model_dir = self.merge_save_dir/ 'model'
        self.merge_model_dir.mkdir(parents=True, exist_ok=True)
        self.merge_predict_dir = self.merge_save_dir/ 'predict'
        self.merge_predict_dir.mkdir(parents=True, exist_ok=True)
        
    def _load_params(self):
        params = toml.load(self.param_dir / f'{self.new_model_name}.toml')
        self.model_list = params['model_list']
        self.weight_list = params['weight_list']
        self.quantile_transform = params.get('quantile_transform', False)
        self.predict_params = params['predict_params']
        self.outlier_n = params.get('outlier_n', 30)
        
        self.adjusted_weight = self.weight_list / np.sum(self.weight_list)
        
    def _merge_predict(self):
        model_list = self.model_list
        adjusted_weight = self.adjusted_weight
        model_dir = self.model_dir
        merge_predict_dir = self.merge_predict_dir
        new_model_name = self.new_model_name
        
        merge_res = None
        
        for model_name, w in zip(model_list, adjusted_weight):
            predict_path = model_dir / model_name / 'predict' / f'predict_{model_name}.parquet'
            predict_res = pd.read_parquet(predict_path)
            if self.quantile_transform:
                original_columns = predict_res.columns
                predict_res = predict_res.apply(lambda row: pd.Series(quantile_transform_with_nan(row.values)), axis=1)
                predict_res.columns = original_columns
            if merge_res is None:
                merge_res = w * predict_res
            else:
                merge_res += w * predict_res
                
        merge_res.to_parquet(merge_predict_dir / f'predict_{new_model_name}.parquet')
        self._test_predicted()
        
    def _test_predicted(self):
        process_name = None
        factor_data_dir = self.merge_predict_dir
        result_dir = self.merge_predict_dir
        params = self.predict_params
        
        ft = FactorTest(process_name, None, factor_data_dir, result_dir=result_dir, params=params)
        ft.test_one_factor(f'predict_{self.new_model_name}')
