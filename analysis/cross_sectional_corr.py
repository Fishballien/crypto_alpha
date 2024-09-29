# -*- coding: utf-8 -*-
"""
Created on Mon Aug 12 13:27:27 2024

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
import numpy as np
import numpy as np
import pandas as pd
import scipy.cluster.hierarchy as spc
from scipy.spatial.distance import squareform


# %% add sys path
file_path = Path(__file__).resolve()
file_dir = file_path.parents[0]
project_dir = file_path.parents[1]
sys.path.append(str(project_dir))


# %% import self-defined
from utils.dirutils import load_path_config


# %% params
cluster_params = {'t': 0.6, 'criterion': 'distance'}


# %%
path_config = load_path_config(project_dir)
processed_data_dir = Path(path_config['processed_data'])
result_dir = Path(path_config['result']) / 'model'
twap_data_dir = Path(path_config['twap_price'])
analysis_dir = Path(path_config['result']) / 'analysis'


# %%
# load twap & calc rtn
curr_px_path = twap_data_dir / 'curr_price_sp240.parquet'
curr_price = pd.read_parquet(curr_px_path)


# %%
rtn = curr_price.pct_change(1, fill_method=None).replace([np.inf, -np.inf], np.nan)


# %%
# 计算币种间的相关性矩阵
corr_matrix = rtn.corr()

# 转换为距离矩阵，距离越近表示相关性越高
distance_matrix = 1 - np.abs(corr_matrix)

# 处理数据精度问题，确保对称性并填充对角线
distance_matrix = np.triu(distance_matrix) + np.triu(distance_matrix, 1).T
# distance_matrix = (distance_matrix + distance_matrix.T) / 2
np.fill_diagonal(distance_matrix, 0)
distance_matrix = np.nan_to_num(distance_matrix, nan=0.0)

# 转换为压缩距离矩阵
condensed_distance_matrix = squareform(distance_matrix)

# 执行层次聚类
linkage_method = 'complete'  # 或 'average' 等其他方法
linkage = spc.linkage(condensed_distance_matrix, method=linkage_method)

# 聚类，定义簇数或其他参数
idx = spc.fcluster(linkage, **cluster_params)

# 将币种及其分类结果存入 DataFrame
cluster_df = pd.DataFrame({
    'Symbol': corr_matrix.columns,
    'Cluster': idx
})

# 将 cluster_params 转换为字符串以添加到文件名中
params_str = '_'.join([f"{key}={value}" for key, value in cluster_params.items()])

# 保存相关性矩阵为 CSV 文件
corr_matrix.to_csv(analysis_dir / f'correlation_matrix_{params_str}.csv')

# 保存币种及其分类结果为 CSV 文件
cluster_df.to_csv(analysis_dir / f'clustered_symbols_{params_str}.csv', index=False)

print("相关性矩阵和币种分类结果已保存。")

