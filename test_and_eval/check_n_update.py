# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 16:45:34 2024

@author: Xintang Zheng

星星: ★ ☆ ✪ ✩ 🌟 ⭐ ✨ 🌠 💫 ⭐️
勾勾叉叉: ✓ ✔ ✕ ✖ ✅ ❎
报警啦: ⚠ ⓘ ℹ ☣
箭头: ➔ ➜ ➙ ➤ ➥ ↩ ↪
emoji: 🔔 ⏳ ⏰ 🔒 🔓 🛑 🚫 ❗ ❓ ❌ ⭕ 🚀 🔥 💧 💡 🎵 🎶 🧭 📅 🤔 🧮 🔢 📊 📈 📉 🧠 📝

"""
# %% imports
from datetime import timedelta
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed


from utils.dirutils import get_filenames_by_extension


# %% update
def format_timedelta_threshold(timedelta_threshold):
    """格式化时间阈值，根据天数或小时输出合适的描述"""
    if timedelta_threshold.days > 0:
        return f"{timedelta_threshold.days}天"
    elif timedelta_threshold.seconds >= 3600:
        hours = timedelta_threshold.seconds // 3600
        return f"{hours}小时"
    elif timedelta_threshold.seconds >= 60:
        minutes = timedelta_threshold.seconds // 60
        return f"{minutes}分钟"
    else:
        return f"{timedelta_threshold.seconds}秒"


def print_daily_diff_stats(difference_mask, combined_index, comparison_df):
    """按天聚合并返回每天不一致行的数量、总行数，以及每行差异比例在一天中的均值，格式为字典"""
    diff_summary = {}
    if difference_mask.any():
        # 将索引转换为日期，并统计每一天不一致的行数
        diff_df = pd.DataFrame({'diff': difference_mask}, index=combined_index)
        diff_df['date'] = diff_df.index.date
        
        daily_diff_count = diff_df.groupby('date')['diff'].sum()  # 每天不一致的数量
        daily_total_count = diff_df.groupby('date')['diff'].count()  # 每天的总数
        
        # 对每一列按日期分组，然后计算每行的比例均值
        row_diff_mean_by_day = comparison_df.groupby(comparison_df.index.date).mean()  # 按日期计算每列的均值
        row_diff_mean = row_diff_mean_by_day.mean(axis=1)  # 对每行计算均值
        
        # 生成差异字典，包含每日不一致比例和每行差异比例的均值
        for date, diff_count in daily_diff_count.items():
            if diff_count > 0:
                total_count = daily_total_count.loc[date]
                daily_mean = row_diff_mean.loc[date]  # 获取当天每行差异比例的均值
                # if daily_mean == 1:
                #     breakpoint()
                diff_summary[date.strftime('%Y-%m-%d')] = (f"{int(diff_count)}/{total_count}", round(daily_mean, 5))
    
    return diff_summary


def add_dataframe_to_dataframe_reindex(df, new_data):
    """
    使用 reindex 将新 DataFrame 的数据添加到目标 DataFrame 中，支持动态扩展列和行，原先没有值的地方填充 NaN。

    参数:
    df (pd.DataFrame): 目标 DataFrame。
    new_data (pd.DataFrame): 要添加的新 DataFrame。

    返回值:
    df (pd.DataFrame): 更新后的 DataFrame。
    """
    # 同时扩展行和列，并确保未填充的空值为 NaN，按排序
    df = df.reindex(index=df.index.union(new_data.index, sort=True),
                    columns=df.columns.union(new_data.columns, sort=True),
                    fill_value=np.nan)
    
    # 使用 loc 添加新数据
    df.loc[new_data.index, new_data.columns] = new_data

    return df


class CheckNUpdate:
    
    def __init__(self, params={}, n_workers=1, log=None):
        self.params = params.copy()
        self.n_workers = n_workers
        self.log = log
        
        self._preprocess_params()
        
    def _preprocess_params(self):
        params = self.params
        timedelta_threshold = params.get('timedelta_threshold', {'minutes': 0})
        params['timedelta_threshold'] = timedelta(**timedelta_threshold)
        params['precision'] = params.get('precision', 1e-5)
    
    def check_n_update(self, factor_name, pre_update_data, incremental_data):
        timedelta_threshold = self.params['timedelta_threshold']
        
        # 检查数据重叠
        pre_start, pre_end, inc_start, inc_end = self._check_data_overlap(
            factor_name, pre_update_data, incremental_data)

        # 处理 before_threshold 数据差异
# =============================================================================
#         self._process_threshold_data(factor_name, pre_update_data, incremental_data, inc_start, 
#                                      is_before_threshold=True)
# =============================================================================
        
        # 处理 after_threshold 数据差异，只到 pre_end
        self._process_threshold_data(factor_name, pre_update_data, incremental_data, inc_start, 
                                     is_before_threshold=False, pre_end=pre_end)
        
        updated_data = self._update_to_updated(pre_update_data, incremental_data, 
                                               inc_start, timedelta_threshold, pre_end)
        
        return updated_data
    
    def _check_data_overlap(self, factor_name, pre_update_data, incremental_data):
        pre_start = pre_update_data.index[0]
        pre_end = pre_update_data.index[-1]
        inc_start = incremental_data.index[0]
        inc_end = incremental_data.index[-1]
        
        if inc_end < pre_start or inc_start > pre_end:
            self.log.error(f"{factor_name} 数据没有重叠部分")
            raise
        
        return pre_start, pre_end, inc_start, inc_end

    def _process_threshold_data(self, factor_name, pre_update_data, incremental_data, inc_start, is_before_threshold, 
                               pre_end=None):
        timedelta_threshold = self.params['timedelta_threshold']
        precision = self.params['precision']
        
        threshold_time = inc_start + timedelta_threshold
        timedelta_threshold_in_format = format_timedelta_threshold(timedelta_threshold)
        if is_before_threshold:
            inc_data = incremental_data.loc[inc_start:threshold_time]  # 修正逻辑，from inc_start 开始
            pre_data = pre_update_data.loc[inc_start:threshold_time]
            threshold_desc = f"前{timedelta_threshold_in_format}"
        else:
            inc_data = incremental_data.loc[threshold_time:pre_end]  # 修正逻辑，处理到 pre_end
            pre_data = pre_update_data.loc[threshold_time:pre_end]
            threshold_desc = f"前{timedelta_threshold_in_format}后的"
        
        combined_index = pre_data.index.union(inc_data.index)
        combined_columns = pre_data.columns.union(inc_data.columns)
        
        pre_data = pre_data.reindex(index=combined_index, columns=combined_columns, fill_value=np.nan)
        inc_data = inc_data.reindex(index=combined_index, columns=combined_columns, fill_value=np.nan)
        
        comparison = ~np.isclose(pre_data, inc_data, atol=precision, equal_nan=True)
        difference_rows = comparison.any(axis=1)
        comparison_df = pd.DataFrame(comparison, index=combined_index, columns=combined_columns)
        
        diff_summary = print_daily_diff_stats(difference_rows, combined_index, comparison_df)
        
        if diff_summary:
            self.log.warning(f"{factor_name}{threshold_desc}数据差异: {diff_summary}")
    
    def _update_to_updated(self, pre_update_data, incremental_data, inc_start, timedelta_threshold, pre_end):
        updated_data = add_dataframe_to_dataframe_reindex(
            pre_update_data, incremental_data.loc[(inc_start+timedelta_threshold):])
        
        return updated_data