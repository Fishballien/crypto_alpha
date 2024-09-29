# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 17:40:18 2024

@author: Xintang Zheng

星星: ★ ☆ ✪ ✩ 🌟 ⭐ ✨ 🌠 💫 ⭐️
勾勾叉叉: ✓ ✔ ✕ ✖ ✅ ❎
报警啦: ⚠ ⓘ ℹ ☣
箭头: ➔ ➜ ➙ ➤ ➥ ↩ ↪
emoji: 🔔 ⏳ ⏰ 🔒 🔓 🛑 🚫 ❗ ❓ ❌ ⭕ 🚀 🔥 💧 💡 🎵 🎶 🧭 📅 🤔 🧮 🔢 📊 📈 📉 🧠 📝

"""
# %% imports
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster


from utils.speedutils import timeit


# %% group features
@timeit
def group_features_by_correlation(X, correlation_threshold, linkage_method='ward'):
    """
    根据特征间相关性进行聚类分组。
    
    参数:
    X (pd.DataFrame): 特征矩阵，列名为 `group_{i}`。
    correlation_threshold (float): 0-1之间的相关性阈值，决定分组数。
    linkage_method (str): 聚类方法，可选 'ward', 'average', 'complete'。

    返回:
    dict: 最终的分组结果，每个组包含相应的特征名称。
    """
    # 计算相关矩阵
    corr = np.corrcoef(X.values, rowvar=False)
    
    # 将相关性矩阵转化为距离矩阵
    distance_matrix = 1 - corr
    
    # 使用层次聚类进行分组
    Z = linkage(distance_matrix, method=linkage_method)
    
    # 根据给定的相关性阈值确定分组数
    max_distance = 1 - correlation_threshold
    clusters = fcluster(Z, max_distance, criterion='distance')
    
    # 创建符合 Group Lasso 格式的分组列表
    group_lasso_groups = clusters.tolist()
    
    return group_lasso_groups
