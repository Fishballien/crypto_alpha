# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 17:09:24 2024

@author: Xintang Zheng

"""
# %% imports
import yaml
import toml
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
from functools import partial
from scipy.spatial.distance import squareform
import scipy.cluster.hierarchy as spc
from queue import Queue
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed


from utils.timeutils import period_shortcut
from utils.datautils import get_one_factor, align_index, align_to_primary
from synthesis.filter_methods import *
from data_processing.feature_engineering import normalization


# %%
def get_df_distance(df_i, df_j, i, j):
    df_j = df_j[df_i.columns]
    p = df_i.corrwith(df_j, axis=1).mean()
    # if np.isnan(p):
    #     breakpoint()
    distance = 1 - abs(p)
    return i, j, distance


def load_one_factor(selected_factor_info, n_fct, get_factor_func, outlier):
    process_name, factor_name = selected_factor_info.loc[n_fct, ['process_name', 'factor']]
    factor = get_factor_func(process_name, factor_name)
    factor = normalization(factor, outlier)
    return n_fct, factor

    
class Cluster:
    
    def __init__(self, cluster_name, t_workers=None, p_workers=None):
        self.cluster_name = cluster_name

        self.t_workers = t_workers
        self.p_workers = p_workers
        
        self._load_paths()
        self._load_cluster_params()
        self._load_test_name()
        self._init_dirs()
        self._init_filter_func()
        self._init_pool_if_needed()
        
    def _load_paths(self):
        file_path = Path(__file__).resolve()
        file_dir = file_path.parents[1]
        path_config_path = file_dir / '.path_config.yaml'
        with path_config_path.open('r') as file:
            path_config = yaml.safe_load(file)
        
        self.processed_data_dir = Path(path_config['processed_data'])
        self.factor_data_dir = Path(path_config['factor_data'])
        self.twap_data_dir = Path(path_config['twap_price'])
        self.result_dir = Path(path_config['result'])
        self.param_dir = Path(path_config['param'])
        
    def _load_cluster_params(self):
        self.params = toml.load(self.param_dir / 'cluster' / f'{self.cluster_name}.toml')
        
    def _load_test_name(self):
        feval_name = self.params.get('feval_name')
        pool_name = self.params.get('pool_name')
        
        if feval_name is not None and pool_name is None:
            feval_params = toml.load(self.param_dir / 'feval' / f'{feval_name}.toml')
            self.test_name = feval_params['test_name']
        elif pool_name is not None and feval_name is None:
            pool_param_dir = self.param_dir / 'features_of_factors' / 'generate'
            pool_params = toml.load(pool_param_dir / f'{pool_name}.toml')
            self.test_name = pool_params['test_data']['test_name']
        
    def _init_dirs(self):
        feval_name = self.params.get('feval_name')
        
        self.feval_dir = (self.result_dir / 'factor_evaluation' / feval_name 
                          if feval_name is not None else None)
        self.cluster_dir = self.result_dir / 'cluster' / f'{self.cluster_name}'
        self.cluster_dir.mkdir(parents=True, exist_ok=True)
        self.test_dir = self.result_dir / 'test'
        
    def _init_filter_func(self):
        filter_name = self.params['filter_func']
        rec_filter_name = self.params.get('rec_filter_func', None)
        super_rec_filter_name = self.params.get('super_rec_filter_func', None)
        self.filter_func = globals()[filter_name]
        self.rec_filter_func = globals()[rec_filter_name] if rec_filter_name is not None else None
        self.super_rec_filter_func = globals()[super_rec_filter_name] if super_rec_filter_name is not None else None
        
    def _init_pool_if_needed(self):
        pool_name = self.params.get('pool_name')
        model_name = self.params.get('model_name')
        if pool_name is None:
            return
        feature_dir = self.processed_data_dir / 'features_of_factors' / pool_name
        predict_dir = self.result_dir / 'features_of_factors' / pool_name / 'model' / model_name / 'predict'
        
        factor_mapping = pd.read_parquet(feature_dir / "factor_mapping.parquet")
        corr_filter = pd.read_parquet(feature_dir / "corr_filter.parquet")
        corr_filter.sort_index(axis=1, inplace=True)
        directions = pd.read_parquet(feature_dir / "direction.parquet")
        directions.sort_index(axis=1, inplace=True)
        predict = pd.read_parquet(predict_dir / f'predict_{model_name}.parquet')
        predict.sort_index(axis=1, inplace=True)
        predict = predict.mask(~corr_filter)
        self.factor_mapping = factor_mapping
        self.directions = directions
        self.predict = predict
    
    def cluster_one_period(self, date_start, date_end, rec_date_start, rec_date_end,
                           super_rec_date_start, super_rec_date_end):
        filter_func = self.filter_func
        rec_filter_func = self.rec_filter_func
        super_rec_filter_func = self.super_rec_filter_func
        feval_dir = self.feval_dir
        cluster_dir = self.cluster_dir
        params = self.params
        feval_name = self.params.get('feval_name')
        corr_target = params.get('corr_target')
        pool_name = params.get('pool_name')
        cluster_params = params.get('cluster_params')
        linkage_method = params.get('linkage_method', 'average')
        
        period_name = period_shortcut(date_start, date_end)
        
        if feval_name is not None and pool_name is None:
            factor_eval = pd.read_csv(feval_dir / f'factor_eval_{period_name}.csv')
        elif pool_name is not None and feval_name is None:
            print(date_end, self.predict[self.predict.index <= date_end])
            latest_direction = self.directions[self.directions.index <= date_end].iloc[-1]
            latest_pred = self.predict[self.predict.index <= date_end].iloc[-1]
            factor_eval = self.factor_mapping.copy()
            factor_eval['direction'] = latest_direction
            factor_eval['predict'] = latest_pred
        else:
            raise NotImplementedError()
        
        selected_idx = filter_func(factor_eval)
        if rec_filter_func is not None:
            rec_period_name = period_shortcut(rec_date_start, rec_date_end)
            rec_factor_eval = pd.read_csv(feval_dir / f'factor_eval_{rec_period_name}.csv')
            rec_factor_eval = align_to_primary(factor_eval, rec_factor_eval, 'process_name', 'factor')
            selected_idx_rec = rec_filter_func(rec_factor_eval)
            selected_idx = selected_idx & selected_idx_rec
        if super_rec_filter_func is not None:
            super_rec_period_name = period_shortcut(super_rec_date_start, super_rec_date_end)
            super_rec_factor_eval = pd.read_csv(feval_dir / f'factor_eval_{super_rec_period_name}.csv')
            super_rec_factor_eval = align_to_primary(factor_eval, super_rec_factor_eval, 'process_name', 'factor')
            selected_idx_super_rec = super_rec_filter_func(super_rec_factor_eval)
            selected_idx = selected_idx & selected_idx_super_rec
        info_list = ['root_dir',  'test_name', 'tag_name', 'process_name', 'factor', 'direction']
        selected_factor_info = factor_eval[selected_idx][info_list].reset_index(drop=True)
        
        if corr_target is not None:
            if corr_target == 'factor':
                distance_matrix = self._calc_corr_by_factor_value(selected_factor_info, period_name, date_start, date_end)
            elif corr_target == 'ic':
                distance_matrix = self._calc_corr_by_ic_value(selected_factor_info, period_name, date_start, date_end)
                    
            try:
                condensed_distance_matrix = squareform(distance_matrix)
            except:
                breakpoint()
            try:
                linkage = spc.linkage(condensed_distance_matrix, method=linkage_method) # complete # average
            except:
                breakpoint()
            idx = spc.fcluster(linkage, **cluster_params)
        else:
            idx = list(range(len(selected_factor_info)))
        
        selected_factor_info['group'] = idx
        selected_factor_info.to_csv(cluster_dir / f'cluster_info_{period_name}.csv', index=None)
        
    def _calc_corr_by_factor_value(self, selected_factor_info, period_name, date_start, date_end):
        params = self.params
        outlier = params['outlier']
        factor_data_dir = self.factor_data_dir
        get_factor_func = partial(get_one_factor, factor_data_dir=factor_data_dir, date_start=date_start, date_end=date_end)
        
        factor_dict = {}
    
        # 判断是否需要线程池加载
        if self.t_workers is None or self.t_workers <= 1:
            # 不使用多线程
            for n_fct in tqdm(selected_factor_info.index, desc=f'load_data_{period_name}'):
                process_name, factor_name = selected_factor_info.loc[n_fct, ['process_name', 'factor']]
                factor = get_factor_func(process_name, factor_name)
                factor = normalization(factor, outlier)
                factor_dict[n_fct] = factor
        else:
            # 使用多线程
            with ThreadPoolExecutor(max_workers=self.t_workers) as t_executor:
                load_tasks = [t_executor.submit(load_one_factor, selected_factor_info, n_fct, get_factor_func, outlier)
                              for n_fct in selected_factor_info.index]
                for task in tqdm(as_completed(load_tasks), total=len(load_tasks), desc=f'load_data_{period_name}'):
                    n_fct, factor = task.result()
                    factor_dict[n_fct] = factor
    
        len_selected = len(selected_factor_info)
        distance_matrix = np.zeros((len_selected, len_selected))
    
        # 判断是否需要进程池计算
        if self.p_workers is None or self.p_workers <= 1:
            # 不使用多进程
            for i in range(len_selected):
                for j in range(i + 1, len_selected):
                    factor_i, factor_j = factor_dict[i], factor_dict[j]
                    i, j, distance = get_df_distance(factor_i, factor_j, i, j)
                    distance_matrix[i, j] = distance
                    distance_matrix[j, i] = distance
        else:
            # 使用多进程
            with ProcessPoolExecutor(max_workers=self.p_workers) as p_executor:
                all_tasks = []
                for i in range(len_selected):
                    for j in range(i + 1, len_selected):
                        factor_i, factor_j = factor_dict[i], factor_dict[j]
                        all_tasks.append(p_executor.submit(get_df_distance, factor_i, factor_j, i, j))
                
                res_queue = Queue()
                for task in tqdm(as_completed(all_tasks), total=len(all_tasks), desc=f'calc_cluster_{period_name}'):
                    res = task.result()
                    if res is not None:
                        res_queue.put(res)
                
                while not res_queue.empty():
                    i, j, distance = res_queue.get()
                    distance_matrix[i, j] = distance
                    distance_matrix[j, i] = distance
    
        return distance_matrix
    
    def _calc_corr_by_ic_value(self, selected_factor_info, period_name, date_start, date_end):
        params = self.params
        data_to_use = params.get('data_to_use', 'icd')
        ic_col = params['ic_col']
        test_dir = self.test_dir
        
        date_range = pd.date_range(start=date_start, end=date_end, freq='D')

        ic_list = []
        for n_fct in selected_factor_info.index:
            tag_name, process_name, factor_name = selected_factor_info.loc[n_fct, ['tag_name', 'process_name', 'factor']]
            process_dir = (test_dir / self.test_name / tag_name if isinstance(tag_name, str)
                          else test_dir / self.test_name)
            data_dir = process_dir / process_name / 'data' 
            try:
                df_icd = pd.read_parquet(data_dir / f'{data_to_use}_{factor_name}.parquet')
                ic_series = df_icd[(df_icd.index >= date_start) & (df_icd.index <= date_end)
                                   ].reindex(date_range)[ic_col].fillna(0)
            except:
                print('missing', f'{data_to_use}_{factor_name}.parquet')
                ic_series = pd.Series(np.zeros(len(date_range)), index=date_range)
            ic_list.append(ic_series)
            
        ic_matrix = np.array(ic_list)
        corr_matrix = np.corrcoef(ic_matrix)
        distance_matrix = 1 - np.abs(corr_matrix)
        
        # 处理数据精度问题
        distance_matrix = (distance_matrix + distance_matrix.T) / 2
        
        # 强制对称性
        for i in range(distance_matrix.shape[0]):
            for j in range(i + 1, distance_matrix.shape[1]):
                avg_value = (distance_matrix[i, j] + distance_matrix[j, i]) / 2
                distance_matrix[i, j] = avg_value
                distance_matrix[j, i] = avg_value
        
        # 确保对角线为零
        np.fill_diagonal(distance_matrix, 0)
        
# =============================================================================
#         for i in range(len_selected): # TODO: 改进
#             for j in range(i + 1, len_selected):
#                 ic_i, ic_j = ic_dict[i], ic_dict[j]
#                 ic_i, ic_j = align_index(ic_i, ic_j)
#                 corr = np.corrcoef(ic_i[ic_col], ic_j[ic_col])[0, 1]
#                 distance = 1 - abs(corr)
#                 distance_matrix[i, j] = distance
#                 distance_matrix[j, i] = distance
# =============================================================================
        return distance_matrix