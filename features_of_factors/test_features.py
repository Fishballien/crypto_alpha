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
import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import yaml
import matplotlib.pyplot as plt
import pickle
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed


# %% add sys path
file_path = Path(__file__).resolve()
file_dir = file_path.parents[0]
project_dir = file_path.parents[1]
sys.path.append(str(project_dir))


# %% class
class TestFeatures:
    
    def __init__(self, pool_name, feature_dir, save_dir, predict_target_dir=None):
        self.pool_name = pool_name
        self.feature_dir = feature_dir
        self.save_dir = save_dir
        self.predict_target_dir = predict_target_dir
        
        self._init_dir()
        self._load_predict_target()
        
    def _init_dir(self):
        self.data_dir = self.save_dir / 'data'
        self.plot_dir = self.save_dir / 'plot'
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.plot_dir.mkdir(parents=True, exist_ok=True)
        self.predict_target_dir = self.feature_dir if self.predict_target_dir is None else self.predict_target_dir
        
    def _load_predict_target(self):
        predict_target_list = [
            'pred1month_sharpe_ratio_0',
            'pred1month_return_annualized_0',
            'pred15days_sharpe_ratio_0',
            'pred15days_return_annualized_0',
            'r_pred1month_sharpe_ratio_0_div_hsr',
            'r_pred15days_sharpe_ratio_0_div_hsr',
            ]
        predict_target_mapping = {predict_target: pd.read_parquet(self.predict_target_dir / f'{predict_target}.parquet')
                                  for predict_target in predict_target_list}
        self.predict_target_mapping = predict_target_mapping
        
    def test_one_feature(self, feature_name, check_exists=True):
        test_result_path = self.data_dir / f'{feature_name}.parquet'
        if check_exists and os.path.exists(test_result_path):
            return
        
        feature = pd.read_parquet(self.feature_dir / f'{feature_name}.parquet')
        
        df_ic = pd.DataFrame()
        for predict_target in self.predict_target_mapping:
            predict = self.predict_target_mapping[predict_target]
            predict = predict.reindex(index=feature.index, columns=feature.columns)
            df_ic[predict_target] = feature.corrwith(predict, axis=1, method='spearman'
                                                     ).replace([np.inf, -np.inf], np.nan).fillna(0)
        
        df_ic.to_parquet(test_result_path)
        
        # Plotting the curve
        n = len(df_ic.columns)
        index = np.arange(len(df_ic))  # x 轴的索引位置
        bar_width = 0.15  # 每个柱的宽度
        
        # Plotting the curve
        plt.figure(figsize=(10, 6))
        
        # 遍历每个 predict_target，并列画出柱状图
        for i, predict_target in enumerate(df_ic.columns):
            plt.bar(index + i * bar_width, df_ic[predict_target], bar_width, label=f'{predict_target}')
        
        plt.grid(True)
        plt.ylim([-1, 1])
        plt.axhline(y=0, linestyle='--')
        plt.title(f'{feature_name}')
        plt.xlabel('Index')
        plt.ylabel('Correlation')
        plt.xticks(index + bar_width * (n - 1) / 2, df_ic.index, rotation=45, ha='right', fontsize=10)  # 调整 x 轴的刻度位置
        plt.legend()
        plt.tight_layout()  # 自动调整子图参数以适应图形区域
        plt.savefig(self.plot_dir / f'{feature_name}.jpg')
        plt.show()
        
        
def main(n_workers=1, check_exists=True):
    tester = TestFeatures(pool_name, feature_dir, save_dir)
    if n_workers is None or n_workers == 1:
        for feature_name in tqdm(features, desc='testing'):
            tester.test_one_feature(feature_name, check_exists=check_exists)
    else:
        # 创建进程池
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            # 提交所有任务到进程池
            futures = {executor.submit(tester.test_one_feature, feature_name, check_exists=check_exists): feature_name 
                       for feature_name in features}
            
            # 使用tqdm跟踪进度
            for future in tqdm(as_completed(futures), total=len(features), desc='testing'):
                feature_name = futures[future]
                try:
                    future.result()  # 如果有需要可以捕获结果
                except Exception as e:
                    print(f"Error testing feature {feature_name}: {e}")


# %%
pool_name = 'pool_240827'


# %% dir
path_config_path = project_dir / '.path_config.yaml'
with path_config_path.open('r') as file:
    path_config = yaml.safe_load(file)
processed_data_dir = Path(path_config['processed_data'])
result_dir = Path(path_config['result'])

feature_dir = processed_data_dir / 'features_of_factors' / pool_name
save_dir = result_dir / 'features_of_factors' / pool_name / 'test'


# %%
with open(feature_dir / 'features_of_factors.pkl', 'rb') as f:
    features = pickle.load(f)


# %% main
if __name__=='__main__':
    n_workers = 100
    check_exists = False
    main(n_workers, check_exists)

