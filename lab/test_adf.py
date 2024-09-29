# -*- coding: utf-8 -*-
"""
Created on Fri Jun 14 13:39:33 2024

@author: Xintang Zheng

"""
# =============================================================================
# import numpy as np
# import pandas as pd
# from statsmodels.tsa.stattools import adfuller
# 
# # 生成一个随机的时间序列数据
# np.random.seed(0)
# time_series_data = np.random.randn(100)  # 生成100个随机数作为时间序列数据
# 
# # 创建一个Pandas Series
# data = pd.Series(time_series_data)
# 
# # 进行ADF检验
# result = adfuller(data)
# 
# # 提取并打印结果
# print('ADF Statistic:', result[0])
# print('p-value:', result[1])
# print('Critical Values:', result[4])
# 
# # 判断结果
# if result[1] < 0.05:
#     print("时间序列是平稳的")
# else:
#     print("时间序列是非平稳的")
# =============================================================================


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller

# 生成一些示例数据
np.random.seed(0)
dates = pd.date_range('20230101', periods=100)
df_skewd = pd.DataFrame(np.random.randn(100, 1), index=dates, columns=['Column1'])

# 进行ADF检验
adf_result = adfuller(df_skewd['Column1'])

# 创建子图
fig = plt.figure()
spec = fig.add_gridspec(12, 12)
ax6 = fig.add_subplot(spec[-11:, -11:])

# 绘制线图并设置标签
df_skewd.plot.line(ax=ax6)
labels = [f'factor mean  ADF Statistic: {adf_result[0]:.2f}, p-value: {adf_result[1]:.3f}']
ax6.legend(labels, loc="upper left")

# # 自定义标签
# custom_label = f'factor mean  ADF Statistic: {adf_result[0]:.2f}, p-value: {adf_result[1]:.3f}'
# handles, _ = ax6.get_legend_handles_labels()
# ax6.legend(handles, [custom_label])

# 添加标题和轴标签
ax6.set_title('Line Plot with Custom Labels')
ax6.set_xlabel('Date')
ax6.set_ylabel('Value')
# ax6.legend(loc="upper left")

# 显示图形
plt.show()
