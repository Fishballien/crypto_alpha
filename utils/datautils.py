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
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from tqdm import tqdm


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


# %% 
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
@gc_collect_after
def get_one_factor(process_name=None, factor_name=None, sp=None, factor_data_dir=None, date_start=None, date_end=None, 
                   ref_order_col=None, ref_index=None, fix_changed_root=False, debug=0):
    if process_name.startswith('gp') and 'sp' not in process_name:
        split_p_name = process_name.split('/')
        split_p_name[1] += '_sp240'
        process_name = '/'.join(split_p_name)
    if sp is not None and 'sp' in process_name:
        process_name = replace_sp(process_name, sp)
    if fix_changed_root:
        str_dir = str(factor_data_dir)
        if str_dir.endswith('neu'):
            factor_data_dir = Path(str_dir + '_4h')
    if debug:
        breakpoint()
    # process_name = 'ma15_sp240'
    factor_dir = factor_data_dir / process_name
    factor_path = factor_dir / f'{factor_name}.parquet'
    try:
        factor = pd.read_parquet(factor_path)
    except:
        print(factor_path)
        traceback.print_exc()
    if 'timestamp' in factor.columns:
        factor.set_index('timestamp', inplace=True)
    factor = factor[(factor.index >= date_start) & (factor.index < date_end)]
    if ref_order_col is not None:
        factor = align_columns(ref_order_col, factor)
    if ref_index is not None:
        factor = align_index_with_main(ref_index, factor)
    return factor


# =============================================================================
# @gc_collect_after # TODO：之后要改加入root dir
# def load_one_group(group_num, group_info, get_one_factor_func, normalization_func, factor_data_dir, to_mask):
#     try:
#         # group_name = f'group_{int(group_num)}'
#         len_of_group = len(group_info)
#         group_factor = None
#         for id_, index in enumerate(group_info.index):
#             process_name, factor_name, direction = group_info.loc[index, ['process_name', 'factor', 'direction']]
#             try:
#                 # 尝试获取特定行和列的值
#                 factor_dir = Path(group_info.loc[index, 'root_dir'])
#             except KeyError:
#                 # 如果列不存在，返回默认值
#                 factor_dir = factor_data_dir
#             factor = get_one_factor_func(process_name, factor_name, factor_data_dir=factor_dir)
#             if not process_name.startswith('gp'):
#                 factor_mask = factor.isna() | to_mask
#                 factor_masked = factor.mask(factor_mask)
#                 factor = normalization_func(factor_masked)
#                 if factor is None:
#                     len_of_group -= 1
#                     # factor = get_one_factor_func(process_name, factor_name, factor_data_dir=factor_dir, debug=1)
#                     # breakpoint()
#                     print(process_name, factor_name)
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


# %% new for features of factors
def load_all_features_of_factors(cluster_info, get_one_factor_func, n_workers):
    tasks = []
    
    for index in cluster_info.index:
        pool_name, feature_name = cluster_info.loc[index, ['pool_name', 'feature_name']]
        tasks.append((index, pool_name, feature_name))
        
    factor_dict = {}
    if n_workers is None or n_workers == 1:
        for (index, pool_name, feature_name) in tasks:
            factor_dict[index] = get_one_factor_func(pool_name, feature_name)
    else:
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = {executor.submit(get_one_factor_func, *task[1:]): task[0] for task in tasks}

            for future in tqdm(as_completed(futures), total=len(futures), desc='load all factors 📊'):
                idx_to_save = futures[future]
                factor = future.result()
                factor_dict[idx_to_save] = factor
    return factor_dict
                
                
@gc_collect_after # TODO：之后要改加入root dir
def load_one_group_of_features(group_num, group_info, factor_dict, normalization_func, to_mask):
    try:
        # group_name = f'group_{int(group_num)}'
        len_of_group = len(group_info)
        group_factor = None
        for id_, index in enumerate(group_info.index):
            pool_name, factor_name, direction = group_info.loc[index, ['pool_name', 'feature_name', 'direction']]
            factor = factor_dict[index]
            if not pool_name.startswith('gp'):
                factor_mask = factor.isna() | to_mask
                factor_masked = factor.mask(factor_mask)
                factor = normalization_func(factor_masked)
                if factor is None:
                    len_of_group -= 1
                    print(pool_name, factor_name)
                    continue
                factor = factor.mask(factor_mask) # TODO: 改为用参数设置mask
            factor = factor * direction
            if group_factor is None:
                group_factor = factor
            else:
                group_factor += factor
        group_factor = group_factor / len_of_group
        group_factor = normalization_func(group_factor)
    except:
        traceback.print_exc()
    return int(group_num), group_factor


# %% new for factors
def load_all_factors(cluster_info, get_one_factor_func, factor_data_dir, n_workers):
    tasks = []
    
    for index in cluster_info.index:
        process_name, factor_name = cluster_info.loc[index, ['process_name', 'factor']]
        try:
            # 尝试获取特定行和列的值
            factor_dir = Path(cluster_info.loc[index, 'root_dir'])
        except KeyError:
            # 如果列不存在，返回默认值
            factor_dir = factor_data_dir
        tasks.append((index, process_name, factor_name, factor_dir))
        
    factor_dict = {}
    if n_workers is None or n_workers == 1:
        for (index, pool_name, feature_name) in tasks:
            factor_dict[index] = get_one_factor_func(pool_name, feature_name, process_name, factor_name)
    else:
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = {executor.submit(get_one_factor_func, *task[1:-1], factor_data_dir=task[-1]): 
                       task[0] for task in tasks}

            for future in tqdm(as_completed(futures), total=len(futures), desc='load all factors 📊'):
                idx_to_save = futures[future]
                factor = future.result()
                factor_dict[idx_to_save] = factor
    return factor_dict
                
                
@gc_collect_after # TODO：之后要改加入root dir
def load_one_group(group_num, group_info, factor_dict={}, normalization_func=None, to_mask=None):
    try:
        # group_name = f'group_{int(group_num)}'
        len_of_group = len(group_info)
        group_factor = None
        for id_, index in enumerate(group_info.index):
            process_name, factor_name, direction = group_info.loc[index, ['process_name', 'factor', 'direction']]
            factor = factor_dict[index]
            if not process_name.startswith('gp'):
                factor_mask = factor.isna() | to_mask
                factor_masked = factor.mask(factor_mask)
                factor = normalization_func(factor_masked)
                if factor is None:
                    len_of_group -= 1
                    print(process_name, factor_name)
                    continue
                factor = factor.mask(factor_mask) # TODO: 改为用参数设置mask
            factor = factor * direction
            if group_factor is None:
                group_factor = factor
            else:
                group_factor += factor
        group_factor = group_factor / len_of_group
        group_factor = normalization_func(group_factor)
    except:
        traceback.print_exc()
    return int(group_num), group_factor


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


# %% merge
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