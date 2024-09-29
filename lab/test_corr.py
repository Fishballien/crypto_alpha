# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 11:13:03 2024

@author: Xintang Zheng

"""
import pandas as pd
import numpy as np

# 创建示例数据
dates = pd.date_range(start='2021-01-01', periods=10, freq='D')
data_x = np.random.rand(10, 3)  # dfx 的数据，10行3列
data_y = np.random.rand(10, 3)  # dfy 的数据，10行3列

dfx = pd.DataFrame(data_x, columns=['X1', 'X2', 'X3'], index=dates)
dfy = pd.DataFrame(data_y, columns=['Y1', 'Y2', 'Y3'], index=dates)

# 初始化列表来存储日期和相关性值
correlations = []

# 将数据按天分组，并对每一天的数据进行操作
for (date, group_x), (_, group_y) in zip(dfx.groupby(dfx.index.date), dfy.groupby(dfy.index.date)):
    # 拼接每天的数据
    concat_data_x = group_x.values.flatten()
    concat_data_y = group_y.values.flatten()
    
    # 计算相关性
    corr_matrix = np.corrcoef(concat_data_x, concat_data_y)
    correlation = corr_matrix[0, 1]  # 取相关系数矩阵中非对角线的值
    correlations.append((date, correlation))

# 将列表转换为 DataFrame
result_df = pd.DataFrame(correlations, columns=['Date', 'Correlation'])
result_df['Date'] = pd.to_datetime(result_df['Date'])
result_df.set_index('Date', inplace=True)

# 输出 DataFrame
print(result_df)


