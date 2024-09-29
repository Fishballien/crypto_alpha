# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 14:13:55 2024

@author: Xintang Zheng

"""
# %% imports
import numpy as np
import numba as nb
import pandas as pd
import traceback
from pathlib import Path
import re
from scipy.stats import yeojohnson


from utils.speedutils import gc_collect_after


# %%
@nb.njit("float64[:](int64[:], int64[:], float64[:], int32, float64)")
def downsampling(org_timeline, tgt_timeline, org_value_arr, g, default):
    tgt_value_arr = np.empty(len(tgt_timeline), org_value_arr.dtype)
    len_of_org = len(org_timeline)
    curr_org_idx = 0
    for target_i, target_t in enumerate(tgt_timeline):
        while curr_org_idx < len_of_org and org_timeline[curr_org_idx] <= target_t:
            curr_org_idx += 1
        org_idx = curr_org_idx - 1 if g == 0 else curr_org_idx 
        tgt_value_arr[target_i] = (default if org_idx < 0 or org_idx == len_of_org
                                   else org_value_arr[org_idx])
    return tgt_value_arr


# %% old 
# =============================================================================
# def align_columns(main_df, sub_df):
#     sub_aligned = pd.DataFrame(columns=main_df.columns, index=sub_df.index)
#     inner_columns = main_df.columns.intersection(sub_df.columns)
#     sub_aligned[inner_columns] = sub_df[inner_columns]
#     return sub_aligned
# =============================================================================

# =============================================================================
# def align_columns(main_col, sub_df):
#     sub_aligned = pd.DataFrame(columns=main_col, index=sub_df.index)
#     inner_columns = main_col.intersection(sub_df.columns)
#     sub_aligned[inner_columns] = sub_df[inner_columns]
#     return sub_aligned
# =============================================================================

# =============================================================================
# def align_index_with_main(main_index, sub_df):
#     sub_aligned = pd.DataFrame(columns=sub_df.columns, index=main_index)
#     inner_index = main_index.intersection(sub_df.index)
#     sub_aligned.loc[inner_index, :] = sub_df.loc[inner_index, :]
#     return sub_aligned
# =============================================================================


def align_columns(main_col, sub_df):
    sub_aligned = sub_df.reindex(columns=main_col)
    return sub_aligned


def align_index(df_1, df_2):
    inner_index = df_1.index.intersection(df_2.index)
    return df_1.loc[inner_index, :], df_2.loc[inner_index, :]


def align_index_with_main(main_index, sub_df):
    # 使用reindex直接对齐索引
    sub_aligned = sub_df.reindex(index=main_index)
    return sub_aligned


def align_to_primary(df_primary, df_secondary, key_column1, key_column2):
    # 检查主 DataFrame 是否有重复值
    assert not df_primary.duplicated(subset=[key_column1, key_column2]).any(), "df_primary contains duplicate rows based on key_column1 and key_column2"
    
    # 取主 DataFrame 的键值组合
    primary_keys = df_primary[[key_column1, key_column2]]
    
    # 根据主 DataFrame 的键值组合重新索引次要 DataFrame
    df_secondary_aligned = df_secondary.set_index([key_column1, key_column2]).reindex(pd.MultiIndex.from_frame(primary_keys)).reset_index()
    
    return df_secondary_aligned


# %%
# =============================================================================
# @gc_collect_after
# def get_one_factor(process_name=None, factor_name=None, sp=None, factor_data_dir=None, date_start=None, date_end=None, 
#                    ref_order_col=None, ref_index=None, debug=0):
#     if process_name.startswith('gp') and 'sp' not in process_name:
#         split_p_name = process_name.split('/')
#         split_p_name[1] += '_sp240'
#         process_name = '/'.join(split_p_name)
#     if sp is not None and 'sp' in process_name:
#         process_name = replace_sp(process_name, sp)
#     if debug:
#         breakpoint()
#     # process_name = 'ma15_sp240'
#     factor_dir = factor_data_dir / process_name
#     factor_path = factor_dir / f'{factor_name}.parquet'
#     factor = pd.read_parquet(factor_path)
#     factor = factor[(factor.index >= date_start) & (factor.index < date_end)]
#     if ref_order_col is not None:
#         factor = align_columns(ref_order_col, factor)
#     if ref_index is not None:
#         factor = align_index_with_main(ref_index, factor)
#     return factor
# =============================================================================


@gc_collect_after # TODO：之后要改加入root dir
def load_one_group(group_num, group_info, get_one_factor_func, normalization_func, factor_data_dir, to_mask):
    try:
        # group_name = f'group_{int(group_num)}'
        len_of_group = len(group_info)
        group_factor = None
        for id_, index in enumerate(group_info.index):
            process_name, factor_name, direction = group_info.loc[index, ['process_name', 'factor', 'direction']]
            try:
                # 尝试获取特定行和列的值
                factor_dir = Path(group_info.loc[index, 'root_dir'])
            except KeyError:
                # 如果列不存在，返回默认值
                factor_dir = factor_data_dir
            factor = get_one_factor_func(process_name, factor_name, factor_data_dir=factor_dir)
            if not process_name.startswith('gp'):
                factor_mask = factor.isna() | to_mask
                factor_masked = factor.mask(factor_mask)
                factor = normalization_func(factor_masked)
                if factor is None:
                    len_of_group -= 1
                    # factor = get_one_factor_func(process_name, factor_name, factor_data_dir=factor_dir, debug=1)
                    # breakpoint()
                    print(process_name, factor_name)
                    continue
                factor = factor.mask(factor_mask) # TODO: 改为用参数设置mask
            factor = factor * direction
            if group_factor is None:
                group_factor = factor
            else:
                group_factor += factor
        group_factor = group_factor / len_of_group
        # if not process_name.startswith('gp'):
        group_factor = normalization_func(group_factor)
        # else:
        #     group_factor = group_factor.replace([np.inf, -np.inf], np.nan)
    except:
        traceback.print_exc()
    return int(group_num), group_factor


# =============================================================================
# @gc_collect_after # TODO：之后要改加入root dir
# def load_one_group_of_features(group_num, group_info, get_one_factor_func, normalization_func, factor_data_dir, to_mask):
#     try:
#         # group_name = f'group_{int(group_num)}'
#         len_of_group = len(group_info)
#         group_factor = None
#         for id_, index in enumerate(group_info.index):
#             pool_name, factor_name, direction = group_info.loc[index, ['pool_name', 'feature_name', 'direction']]
#             try:
#                 # 尝试获取特定行和列的值
#                 factor_dir = Path(group_info.loc[index, 'root_dir'])
#             except KeyError:
#                 # 如果列不存在，返回默认值
#                 factor_dir = factor_data_dir
#             factor = get_one_factor_func(pool_name, factor_name, factor_data_dir=factor_dir)
#             if not pool_name.startswith('gp'):
#                 factor_mask = factor.isna() | to_mask
#                 factor_masked = factor.mask(factor_mask)
#                 factor = normalization_func(factor_masked)
#                 if factor is None:
#                     len_of_group -= 1
#                     # factor = get_one_factor_func(process_name, factor_name, factor_data_dir=factor_dir, debug=1)
#                     # breakpoint()
#                     print(pool_name, factor_name)
#                     continue
#                 factor = factor.mask(factor_mask) # TODO: 改为用参数设置mask
#             factor = factor * direction
#             if group_factor is None:
#                 group_factor = factor
#             else:
#                 group_factor += factor
#         group_factor = group_factor / len_of_group
#         # if not process_name.startswith('gp'):
#         group_factor = normalization_func(group_factor)
#         # else:
#         #     group_factor = group_factor.replace([np.inf, -np.inf], np.nan)
#     except:
#         traceback.print_exc()
#     return int(group_num), group_factor
# =============================================================================


# %% test
import time
import traceback
from pathlib import Path
import pandas as pd

# 2024-08-26 15:54:02 INFO Start Fit: 220701_230701
# Process name handling time: 0.000000 seconds
# Factor loading time: 3.584571 seconds
# Date filtering time: 0.014960 seconds
# Column alignment time: 34.752625 seconds
# Index alignment time: 26.161740 seconds
# [Group 1] Row 0: Data reading time: 64.810261 seconds
# [Group 1] Row 0: Normalization time: 4.244433 seconds
# [Group 1] Row 0: group_factor += factor time: 0.000000 seconds

# to

# 2024-08-26 16:57:00 INFO Start Fit: 220701_230701
# Process name handling time: 0.000000 seconds
# Factor loading time: 3.599091 seconds
# Date filtering time: 0.012251 seconds
# Column alignment time: 0.005116 seconds
# Index alignment time: 0.004546 seconds
# [Group 1] Row 0: Data reading time: 3.784578 seconds
# [Group 1] Row 0: Normalization time: 0.080814 seconds
# [Group 1] Row 0: group_factor += factor time: 0.000000 seconds

# =============================================================================
# @gc_collect_after
# def get_one_factor(process_name=None, factor_name=None, sp=None, factor_data_dir=None, date_start=None, date_end=None, 
#                    ref_order_col=None, ref_index=None, debug=0):
#     
#     start_time = time.time()
#     
#     if process_name.startswith('gp') and 'sp' not in process_name:
#         split_p_name = process_name.split('/')
#         split_p_name[1] += '_sp240'
#         process_name = '/'.join(split_p_name)
#     if sp is not None and 'sp' in process_name:
#         process_name = replace_sp(process_name, sp)
#     process_name_time = time.time() - start_time
#     print(f"Process name handling time: {process_name_time:.6f} seconds")
#     
#     if debug:
#         breakpoint()
#     
#     start_time = time.time()
#     factor_dir = factor_data_dir / process_name
#     factor_path = factor_dir / f'{factor_name}.parquet'
#     factor = pd.read_parquet(factor_path)
#     factor_loading_time = time.time() - start_time
#     print(f"Factor loading time: {factor_loading_time:.6f} seconds")
#     
#     start_time = time.time()
#     factor = factor[(factor.index >= date_start) & (factor.index < date_end)]
#     date_filtering_time = time.time() - start_time
#     print(f"Date filtering time: {date_filtering_time:.6f} seconds")
#     
#     if ref_order_col is not None:
#         start_time = time.time()
#         factor = align_columns(ref_order_col, factor)
#         column_alignment_time = time.time() - start_time
#         print(f"Column alignment time: {column_alignment_time:.6f} seconds")
#     
#     if ref_index is not None:
#         start_time = time.time()
#         factor = align_index_with_main(ref_index, factor)
#         index_alignment_time = time.time() - start_time
#         print(f"Index alignment time: {index_alignment_time:.6f} seconds")
#     
#     return factor
# 
# 
# @gc_collect_after  # TODO：之后要改加入root dir
# def load_one_group_of_features(group_num, group_info, get_one_factor_func, normalization_func, factor_data_dir, to_mask):
#     try:
#         len_of_group = len(group_info)
#         group_factor = None
# 
#         for id_, index in enumerate(group_info.index):
#             pool_name, factor_name, direction = group_info.loc[index, ['pool_name', 'feature_name', 'direction']]
#             
#             try:
#                 # 尝试获取特定行和列的值
#                 factor_dir = Path(group_info.loc[index, 'root_dir'])
#             except KeyError:
#                 # 如果列不存在，返回默认值
#                 factor_dir = factor_data_dir
#             
#             # 记录数据读取时间
#             start_time = time.time()
#             factor = get_one_factor_func(pool_name, factor_name, factor_data_dir=factor_dir)
#             read_time = time.time() - start_time
#             print(f"[Group {group_num}] Row {id_}: Data reading time: {read_time:.6f} seconds")
# 
#             if not pool_name.startswith('gp'):
#                 factor_mask = factor.isna() | to_mask
#                 factor_masked = factor.mask(factor_mask)
#                 
#                 # 记录 normalization 时间
#                 start_time = time.time()
#                 factor = normalization_func(factor_masked)
#                 norm_time = time.time() - start_time
#                 print(f"[Group {group_num}] Row {id_}: Normalization time: {norm_time:.6f} seconds")
#                 
#                 if factor is None:
#                     len_of_group -= 1
#                     print(f"[Group {group_num}] Row {id_}: Normalization returned None for {pool_name}, {factor_name}")
#                     continue
#                 factor = factor.mask(factor_mask)  # TODO: 改为用参数设置mask
#             
#             factor = factor * direction
# 
#             # 记录 group_factor += factor 时间
#             start_time = time.time()
#             if group_factor is None:
#                 group_factor = factor
#             else:
#                 group_factor += factor
#             add_time = time.time() - start_time
#             print(f"[Group {group_num}] Row {id_}: group_factor += factor time: {add_time:.6f} seconds")
# 
#         if len_of_group > 0:
#             # 计算 group_factor 的平均值
#             group_factor = group_factor / len_of_group
# 
#             # 记录最后一次 normalization 时间
#             start_time = time.time()
#             group_factor = normalization_func(group_factor)
#             final_norm_time = time.time() - start_time
#             print(f"[Group {group_num}]: Final normalization time: {final_norm_time:.6f} seconds")
#         else:
#             group_factor = None
#             print(f"[Group {group_num}]: No valid factors to compute group_factor.")
# 
#     except Exception as e:
#         traceback.print_exc()
#     
#     return int(group_num), group_factor
# 
# =============================================================================

# %%
def replace_sp(org_name, target_sp):
    match = re.search(r'(\d+)', target_sp)
    if match:
        target_num = match.group(1)
        return re.sub(r'(_sp)\d+', rf'\g<1>{target_num}', org_name)
    else:
        raise ValueError
        

def extract_sp(name):
    match = re.search(r'_sp(\d+)', name)
    if match:
        target_num = match.group(1)
        return int(target_num)
    else:
        raise ValueError
        
        
# %%
def yeojohnson_transform(row):
    transformed_row, _ = yeojohnson(row)
    return pd.Series(transformed_row, index=row.index)


# %% qcut
def qcut_row(row, q=10):
    # qcut 会在行中产生 NaN 值的情况下失败，因此需要先处理 NaN 值
    row_nonan = row.dropna()
    qcut_result = pd.qcut(row_nonan, q=q, labels=False, duplicates='drop')
    result = pd.Series(qcut_result, index=row_nonan.index)
    return result.reindex_like(row)