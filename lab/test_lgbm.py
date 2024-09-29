# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 16:35:35 2024

@author: Xintang Zheng

"""

import lightgbm as lgb
import numpy as np

# 示例数据
X = np.random.rand(100, 10)
y = np.random.randn(100)  # 均值为0的目标变量

train_data = lgb.Dataset(X, label=y)
params = {
    'objective': 'regression',
    'metric': 'mse',
    'learning_rate': 0.01,
    'num_leaves': 31,
    'max_depth': 15,
    'n_estimators': 1000
}

# 训练模型
model = lgb.train(params, train_data)
print('finished')

# 输出日志中会有 [LightGBM] [Info] Start training from score 0.000000
