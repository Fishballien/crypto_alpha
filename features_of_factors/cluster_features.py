# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 17:09:24 2024

@author: Xintang Zheng

"""
# %% imports
import yaml
import toml
from pathlib import Path
import pandas as pd
import numpy as np
from scipy.spatial.distance import squareform
import scipy.cluster.hierarchy as spc


from utils.timeutils import period_shortcut
from filter_methods import *


# %%
class Cluster:
    
    def __init__(self, cluster_name):
        self.cluster_name = cluster_name

        self._load_paths()
        self._load_cluster_params()
        self._init_dirs()
        self._init_filter_func()
        
    def _load_paths(self):
        file_path = Path(__file__).resolve()
        file_dir = file_path.parents[1]
        path_config_path = file_dir / '.path_config.yaml'
        with path_config_path.open('r') as file:
            path_config = yaml.safe_load(file)

        self.result_dir = Path(path_config['result'])
        self.param_dir = Path(path_config['param'])
        
    def _load_cluster_params(self):
        self.params = toml.load(self.param_dir / 'features_of_factors' / 'cluster' / f'{self.cluster_name}.toml')

    def _init_dirs(self):
        pool_name = self.params['pool_name']
        
        pool_dir = self.result_dir / 'features_of_factors' / pool_name
        self.test_dir = pool_dir / 'test'
        self.feval_dir = pool_dir / 'eval'
        self.cluster_dir = pool_dir / 'cluster' / f'{self.cluster_name}'
        self.cluster_dir.mkdir(parents=True, exist_ok=True)
        
    def _init_filter_func(self):
        filter_name = self.params['filter_func']
        self.filter_func = globals()[filter_name]
     
    def cluster_one_period(self, date_start, date_end):
        filter_func = self.filter_func
        feval_dir = self.feval_dir
        cluster_dir = self.cluster_dir
        params = self.params
        corr_target = params.get('corr_target')
        cluster_params = params.get('cluster_params')
        linkage_method = params.get('linkage_method', 'average')
        
        period_name = period_shortcut(date_start, date_end)
        factor_eval = pd.read_csv(feval_dir / f'feature_eval_{period_name}.csv')
        
        selected_idx = filter_func(factor_eval)
        info_list = ['pool_name',  'feature_name', 'direction']
        selected_factor_info = factor_eval[selected_idx][info_list].reset_index(drop=True)
        
        if corr_target is not None:
            if corr_target == 'ic':
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
    
    def _calc_corr_by_ic_value(self, selected_factor_info, period_name, date_start, date_end):
        params = self.params
        ic_col = params['ic_col']
        test_dir = self.test_dir

        ic_list = []
        for n_fct in selected_factor_info.index:
            pool_name, feature_name = selected_factor_info.loc[n_fct, ['pool_name', 'feature_name']]
            data_dir = test_dir / 'data' 
            df_icd = pd.read_parquet(data_dir / f'{feature_name}.parquet')
            ic_series = df_icd[(df_icd.index >= date_start) & (df_icd.index < date_end)][ic_col].fillna(0)
            # breakpoint()
            ic_list.append(ic_series)
            
        ic_matrix = np.array(ic_list)
        corr_matrix = np.corrcoef(ic_matrix)
        distance_matrix = 1 - np.abs(corr_matrix)
        
        # 处理数据精度问题
        distance_matrix = (distance_matrix + distance_matrix.T) / 2
        np.fill_diagonal(distance_matrix, 0)

        return distance_matrix