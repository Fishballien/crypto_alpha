# -*- coding: utf-8 -*-
"""
Created on Thu Mar 20 22:34:24 2025

@author: Xintang Zheng

星星: ★ ☆ ✪ ✩ 🌟 ⭐ ✨ 🌠 💫 ⭐️
勾勾叉叉: ✓ ✔ ✕ ✖ ✅ ❎
报警啦: ⚠ ⓘ ℹ ☣
箭头: ➔ ➜ ➙ ➤ ➥ ↩ ↪
emoji: 🔔 ⏳ ⏰ 🔒 🔓 🛑 🚫 ❗ ❓ ❌ ⭕ 🚀 🔥 💧 💡 🎵 🎶 🧭 📅 🤔 🧮 🔢 📊 📈 📉 🧠 📝

"""
import pandas as pd
import numpy as np
from scipy import stats
import time
import matplotlib.pyplot as plt
from functools import wraps
import gc

def timing_decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__} 执行耗时: {end_time - start_time:.4f} 秒")
        return result, end_time - start_time
    return wrapper

# 生成测试数据
def generate_sparse_dataframes(num_rows, num_cols, sparsity=0.9):
    """
    生成稀疏测试数据
    
    参数:
    - num_rows: 行数
    - num_cols: 列数
    - sparsity: 稀疏度，NaN的比例
    
    返回:
    - factor_df: 因子数据
    - return_df: 收益率数据
    """
    # 创建日期索引
    dates = pd.date_range(start='2020-01-01', periods=num_rows, freq='30min')
    
    # 创建列名
    columns = [f'asset_{i}' for i in range(num_cols)]
    
    # 创建因子数据
    factor_data = np.random.randn(num_rows, num_cols)
    return_data = np.random.randn(num_rows, num_cols) * 0.05  # 收益率通常较小
    
    # 添加稀疏性
    mask = np.random.random(size=(num_rows, num_cols)) < sparsity
    factor_data[mask] = np.nan
    return_data[mask] = np.nan
    
    # 转换为DataFrame
    factor_df = pd.DataFrame(factor_data, index=dates, columns=columns)
    return_df = pd.DataFrame(return_data, index=dates, columns=columns)
    
    return factor_df, return_df

# 原始方法
@timing_decorator
def original_method(factor_df, return_df):
    """原始IC计算方法"""
    ics = factor_df.corrwith(return_df, axis=1, method='spearman').replace([np.inf, -np.inf], np.nan).fillna(0)
    icsm = ics.resample('ME').mean()
    icsd = ics.resample('1d').mean()
    return icsm, icsd

# 方法1：预过滤有效列
@timing_decorator
def filtered_method(factor_df, return_df):
    """预过滤有效列的方法"""
    # 找出两个数据框中有效数据的共同列
    valid_cols_factor = factor_df.columns[~factor_df.isna().all()]
    valid_cols_return = return_df.columns[~return_df.isna().all()]
    common_valid_cols = list(set(valid_cols_factor) & set(valid_cols_return))
    
    # 使用共同的有效列计算相关系数
    ics = factor_df[common_valid_cols].corrwith(return_df[common_valid_cols], 
                                             axis=1, 
                                             method='spearman').replace([np.inf, -np.inf], np.nan).fillna(0)
    icsm = ics.resample('ME').mean()
    icsd = ics.resample('1d').mean()
    return icsm, icsd

# 方法2：向量化按行计算
@timing_decorator
def vectorized_method(factor_df, return_df):
    """向量化按行计算的方法"""
    def row_spearman(row_idx):
        # 获取当前行数据
        f_row = factor_df.loc[row_idx]
        r_row = return_df.loc[row_idx]
        
        # 找出共同的非NaN点
        mask = ~(f_row.isna() | r_row.isna())
        
        # 如果有足够的数据点，计算相关系数
        if mask.sum() >= 5:
            corr, _ = stats.spearmanr(f_row[mask], r_row[mask])
            return corr if not np.isnan(corr) else 0
        else:
            return 0
    
    # 应用到每个日期
    ics = pd.Series([row_spearman(idx) for idx in factor_df.index], index=factor_df.index)
    icsm = ics.resample('ME').mean()
    icsd = ics.resample('1d').mean()
    return icsm, icsd

# 方法3：低级NumPy实现
@timing_decorator
def numpy_method(factor_df, return_df):
    """使用NumPy的低级实现"""
    result = pd.Series(index=factor_df.index)
    
    for idx in factor_df.index:
        x = factor_df.loc[idx].values
        y = return_df.loc[idx].values
        mask = ~(np.isnan(x) | np.isnan(y))
        if mask.sum() >= 5:
            rho, _ = stats.spearmanr(x[mask], y[mask])
            result[idx] = rho if not np.isnan(rho) else 0
        else:
            result[idx] = 0
    
    icsm = result.resample('ME').mean()
    icsd = result.resample('1d').mean()
    return icsm, icsd

# 运行性能测试
def run_performance_test(data_sizes=[(1000, 200), (5000, 500)], sparsity_levels=[0.8, 0.9, 0.95]):
    """
    运行性能测试
    
    参数:
    - data_sizes: (行数, 列数)的列表
    - sparsity_levels: 稀疏度列表
    """
    results = []
    
    for rows, cols in data_sizes:
        for sparsity in sparsity_levels:
            print(f"\n测试数据集: {rows}行 x {cols}列, 稀疏度: {sparsity}")
            
            # 生成测试数据
            factor_df, return_df = generate_sparse_dataframes(rows, cols, sparsity)
            
            # 测试各方法
            _, time_original = original_method(factor_df, return_df)
            _, time_filtered = filtered_method(factor_df, return_df)
            _, time_vectorized = vectorized_method(factor_df, return_df)
            _, time_numpy = numpy_method(factor_df, return_df)
            
            # 记录结果
            results.append({
                'rows': rows,
                'cols': cols,
                'sparsity': sparsity,
                'original': time_original,
                'filtered': time_filtered,
                'vectorized': time_vectorized,
                'numpy': time_numpy
            })
            
            # 清理内存
            gc.collect()
    
    return results

# 绘制结果图表
def plot_results(results):
    """绘制测试结果图表"""
    import matplotlib.pyplot as plt
    import pandas as pd
    
    df_results = pd.DataFrame(results)
    
    # 为每组数据大小创建一个图
    for (rows, cols), group in df_results.groupby(['rows', 'cols']):
        plt.figure(figsize=(10, 6))
        
        # 绘制各方法在不同稀疏度下的性能
        plt.plot(group['sparsity'], group['original'], 'o-', label='原始方法')
        plt.plot(group['sparsity'], group['filtered'], 's-', label='预过滤方法')
        plt.plot(group['sparsity'], group['vectorized'], '^-', label='向量化方法')
        plt.plot(group['sparsity'], group['numpy'], 'd-', label='NumPy方法')
        
        plt.title(f'数据大小: {rows}行 x {cols}列')
        plt.xlabel('稀疏度')
        plt.ylabel('执行时间 (秒)')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f'ic_perf_comparison_{rows}x{cols}.png')
        plt.show()

# 主函数
if __name__ == "__main__":
    # 定义不同的测试数据大小
    data_sizes = [
        # (1000, 100),    # 小数据集
        # (5000, 200),    # 中等数据集
        (70000, 400)    # 大数据集 (可能需要较长时间)
    ]
    
    # 定义不同的稀疏度
    sparsity_levels = [0.6, 0.7, 0.8, 0.9, 0.95]
    
    # 运行测试
    print("开始性能测试...")
    results = run_performance_test(data_sizes, sparsity_levels)
    
    # 打印结果表格
    print("\n性能测试结果摘要:")
    df_results = pd.DataFrame(results)
    print(df_results)
    
    # 保存结果
    df_results.to_csv('ic_calculation_performance_results.csv')
    print("结果已保存到 ic_calculation_performance_results.csv")
    
    # 绘制结果图表
    plot_results(results)
