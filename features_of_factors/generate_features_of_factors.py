# -*- coding: utf-8 -*-
"""
Created on Thu Aug 22 13:20:22 2024

@author: Xintang Zheng

星星: ★ ☆ ✪ ✩ 🌟 ⭐ ✨ 🌠 💫 ⭐️
勾勾叉叉: ✓ ✔ ✕ ✖ ✅ ❎
报警啦: ⚠ ⓘ ℹ ☣
箭头: ➔ ➜ ➙ ➤ ➥ ↩ ↪
emoji: 🔔 ⏳ ⏰ 🔒 🔓 🛑 🚫 ❗ ❓ ❌ ⭕ 🚀 🔥 💧 💡 🎵 🎶 🧭 📅 🤔 🧮 🔢 📊 📈 📉 🧠 📝

"""
# %% rules
'''
pool_name对应唯一的因子池
factor的direction，根据每期回看2y来决定
'''
# %% imports
import os
from pathlib import Path
import pandas as pd
import numpy as np
from collections import defaultdict
from tqdm import tqdm
import toml
import yaml
import pickle
from functools import partial
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from random import shuffle


from utils.dirutils import DirectoryProcessor
from utils.timeutils import RollingPeriods, translate_rolling_params, parse_relativedelta
from test_and_eval.scores import *


# %% features
def fac_direction(factor_dataset, date_start, date_end, idx_to_save, f_id):
    gp = factor_dataset[f_id]
    if gp is None:
        return np.nan
    gp_prd = gp[(gp.index >= date_start) & (gp.index <= date_end)]
    cumrtn_lag_0 = gp_prd['long_short_0'].sum()
    direction = 1 if cumrtn_lag_0 > 0 else -1
    return direction


# =============================================================================
# def fac_direction(f_dataset, date_start, date_end, idx_to_save, f_id):
#     gp = f_dataset['gp']
#     gp_prd = gp[(gp.index >= date_start) & (gp.index <= date_end)]
#     cumrtn_lag_0 = gp_prd['long_short_0'].sum()
#     direction = 1 if cumrtn_lag_0 > 0 else -1
#     return direction
# =============================================================================


def fac_gp_related_metrics(factor_dataset, date_start, date_end, idx_to_save, f_id, 
                           direction_fac, metric_func, gp_lag):
    direction = direction_fac.loc[idx_to_save, f_id]
    gp = factor_dataset[f_id]
    if gp is None:
        return np.nan
    gp_prd = gp[(gp.index >= date_start) & (gp.index <= date_end)]
    metric = metric_func(gp_prd[f'long_short_{gp_lag}']*direction)
    return metric


def fac_icd_related_metrics(factor_dataset, date_start, date_end, idx_to_save, f_id, 
                            direction_fac, metric_func, ic_type):
    direction = direction_fac.loc[idx_to_save, f_id]
    icd = factor_dataset[f_id]
    if icd is None:
        return np.nan
    icd_prd = icd[(icd.index >= date_start) & (icd.index <= date_end)]
    metric = metric_func(icd_prd[ic_type]*direction)
    return metric


def fac_hsr_related_metrics(factor_dataset, date_start, date_end, idx_to_save, f_id):
    hsr = factor_dataset[f_id]
    if hsr is None:
        return np.nan
    hsr_prd = hsr[(hsr.index >= date_start) & (hsr.index <= date_end)]
    hsr_avg = hsr_prd.mean(axis=1).mean(axis=0)
    return hsr_avg


def fac_bins_related_metrics(factor_dataset, date_start, date_end, idx_to_save, f_id, 
                             direction_fac, metric_func):
    direction = direction_fac.loc[idx_to_save, f_id]
    bins = factor_dataset[f_id]
    if bins is None:
        return np.nan
    bins_of_lag_0 = bins[0]
    bins_of_lag_0 = bins_of_lag_0[(bins_of_lag_0.index >= date_start) & (bins_of_lag_0.index <= date_end)]
    bin_long_short = bins_of_lag_0['90% - 100%'] - bins_of_lag_0['0% - 10%']
    metric = metric_func(bin_long_short*direction)
    return metric


def get_one_factor_sharpe_and_gp(df_gp, *, date_start, date_end, date_range, valid_prop_thresh, filter_gp):
    if df_gp is None:
        return 0.0, pd.Series(0, index=date_range)
    
    # time period
    df_gp = df_gp[(df_gp.index >= date_start) & (df_gp.index <= date_end)].reindex(date_range)
    # ic
    ic_series = df_gp[filter_gp].fillna(0)
    # long short metrics
    gps = df_gp[filter_gp].fillna(0)
    if gps.count() < valid_prop_thresh * gps.size:
        return 0.0, pd.Series(0, index=date_range)
    cumrtn_lag_0 = df_gp[filter_gp].sum()
    direction = 1 if cumrtn_lag_0 > 0 else -1
    sharpe = calc_sharpe(df_gp[filter_gp]*direction)
    return sharpe, ic_series


def filter_correlated_factors(factor_dataset, func, date_start, date_end, idx_to_save,
                              valid_prop_thresh, corr_thresh, filter_gp):
    date_range = pd.date_range(start=date_start, end=date_end, freq='D')
    calc_func = partial(get_one_factor_sharpe_and_gp, date_start=date_start, date_end=date_end, date_range=date_range,
                        valid_prop_thresh=valid_prop_thresh, filter_gp=filter_gp)
    len_of_dataset = len(factor_dataset)
    results = [calc_func(factor_dataset[f_id]) for f_id in range(len_of_dataset)]

    ic_list = [result[1] for result in results]
    sharpe_list = [result[0] for result in results]
    del results
    
    ic_matrix = np.array(ic_list)
    corr_matrix = np.corrcoef(ic_matrix)
    
    # 创建一个布尔数组来标记是否保留因子
    keep = np.ones(len_of_dataset, dtype=bool)
    
    # 按照 Sharpe 比率从高到低排序因子索引
    sorted_indices = np.argsort(sharpe_list)[::-1]
    
    # 遍历排序后的因子索引，保留高 Sharpe 比率因子，并标记与其高度相关的因子
    for i in sorted_indices:
        if keep[i]:
            for j in range(len_of_dataset):
                if i != j and abs(corr_matrix[i, j]) > corr_thresh:
                    keep[j] = False
    
    return keep


# %% class
def _process_one_period(factor_dataset, func, date_start, date_end, idx_to_save, corr_filter=None):
    one_row = []
    len_of_dataset = len(factor_dataset)
    
    if corr_filter is None:
        for f_id in range(len_of_dataset):
            result = func(factor_dataset, date_start, date_end, idx_to_save, f_id)
            one_row.append(result)
    else:
        corr_filter_to_use = corr_filter.loc[idx_to_save]
        for f_id in range(len_of_dataset):
            if corr_filter_to_use[f_id]:
                result = func(factor_dataset, date_start, date_end, idx_to_save, f_id)
                one_row.append(result)
            else:
                one_row.append(np.nan)
    return one_row


def _save_parquet(fac_df, fac_name, save_dir):
    fac_df.to_parquet(save_dir / f'{fac_name}.parquet')
    
    
def _generate(func=None, func_one_period=_process_one_period, factor_dataset={}, rolling_choice='fit', rolling_choices={}, 
              window=None, name_to_save='', save_dir=Path(), workers=1):
    fac = pd.DataFrame()
    rolling_target = rolling_choices[rolling_choice]
    tasks = []
    all_idx = []
    len_of_dataset = len(factor_dataset)

    # 准备任务，for p in rolling_target 循环并行
    for p in rolling_target:
        date_start, date_end = p
        idx_to_save = date_end if rolling_choice == 'fit' else date_start
        if window is not None:
            if rolling_choice == 'fit':
                date_start = date_end - parse_relativedelta(window)
            elif rolling_choice == 'predict':
                date_end = date_start + parse_relativedelta(window)
        tasks.append((factor_dataset, func, date_start, date_end, idx_to_save))
        all_idx.append(idx_to_save)
        
    # 预先构建 fac 数据框，行是 rolling_target 的时间点，列是 factor_id
    fac = pd.DataFrame(index=all_idx, columns=list(range(len_of_dataset)))

    # 并行执行 for p in rolling_target 的循环
    if workers == 1:
        for task in tasks:
            idx_to_save = task[-1]
            one_row = func_one_period(*task)
            fac.loc[idx_to_save, :] = one_row
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(func_one_period, *task): task[-1] for task in tasks}

            for future in tqdm(as_completed(futures), total=len(futures), desc=f'{name_to_save} 📊'):
                idx_to_save = futures[future]
                one_row = future.result()
                fac.loc[idx_to_save, :] = one_row

    _save_parquet(fac, name_to_save, save_dir)
    return fac


class GenerateFeaturesOfFactors:
    
    def __init__(self, param_name, data_workers=1, calc_workers=1):
        self.param_name = param_name
        self.data_workers = data_workers
        self.calc_workers = calc_workers
        self.features_of_factors = []
        
        self._load_paths()
        self._load_params()
        self._init_dirs()
        self._init_director_processor()
        self._init_factor_mapping()
        self._init_rolling_periods()
        self._load_all_factor_dataset()
        
    def _load_paths(self):
        file_path = Path(__file__).resolve()
        file_dir = file_path.parents[1]
        path_config_path = file_dir / '.path_config.yaml'
        with path_config_path.open('r') as file:
            path_config = yaml.safe_load(file)

        self.param_dir = Path(path_config['param'])
        self.result_dir = Path(path_config['result'])
        self.processed_data_dir = Path(path_config['processed_data'])
        
    def _load_params(self):
        self.params = toml.load(self.param_dir / 'features_of_factors' / 'generate' / f'{self.param_name}.toml')
        
    def _init_dirs(self):
        self.test_dir = self.result_dir  / 'test'
        pool_params = self.params['pool']

        pool_name = pool_params['pool_name']
        self.save_dir = self.processed_data_dir / 'features_of_factors' / pool_name
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
    def _init_director_processor(self):
        root_dir_dict = self.params['root_dir_dict']
        
        self.dirp = DirectoryProcessor(root_dir_dict)
        
    def _save_parquet(self, fac_df, fac_name):
        fac_df.to_parquet(self.save_dir / f'{fac_name}.parquet')
        
    def _init_factor_mapping(self):
        test_data_params = self.params['test_data']
        test_name = test_data_params['test_name']
        list_of_tuple = self.dirp.list_of_tuple
        
        factor_list = []

        for (root_dir, tag_name, process_name) in list_of_tuple:
             factor_dir = Path(root_dir) / process_name
             factor_name_list = [path.stem for path in factor_dir.glob('*.parquet')]
             for factor_name in factor_name_list:
                 factor_list.append((root_dir, test_name, tag_name, process_name, factor_name))

        factor_mapping = pd.DataFrame(factor_list, columns=[
            'root_dir',  'test_name', 'tag_name', 'process_name', 'factor'])
        _save_parquet(factor_mapping, 'factor_mapping', self.save_dir)
        self.factor_mapping = factor_mapping
        
    def _init_rolling_periods(self):
        rolling_params = translate_rolling_params(self.params['rolling_params'])
        rolling = RollingPeriods(**rolling_params)
        self.rolling_choice = {
            'fit': rolling.fit_periods,
            'predict': rolling.predict_periods,
            }
        
    def _load_all_factor_dataset(self):
        test_data_params = self.params['test_data']
        data_to_load = test_data_params['data_to_load']
        factor_mapping = self.factor_mapping
        test_dir = self.test_dir
        
        factor_dataset = self._load_dataset()
        
        if factor_dataset:
            self.factor_dataset = factor_dataset
            return

        tasks = []
        
        # 准备任务
        for f_id in factor_mapping.index:
            test_name = factor_mapping.loc[f_id, 'test_name']
            tag_name = factor_mapping.loc[f_id, 'tag_name']
            process_name = factor_mapping.loc[f_id, 'process_name']
            factor_name = factor_mapping.loc[f_id, 'factor']
            test_data_dir = test_dir / test_name / tag_name / process_name / 'data'
            tasks.append((f_id, factor_name, test_data_dir, data_to_load))
        
        # 并行执行任务
        with ThreadPoolExecutor(max_workers=self.data_workers) as executor:
            futures = {executor.submit(self._load_factor_data, *task[1:]): task[0] for task in tasks}
            
            for future in tqdm(as_completed(futures), total=len(futures), desc='load factor dataset 🔢'):
                f_id = futures[future]
                res = future.result()
                for data_type in data_to_load:
                    factor_dataset[data_type][f_id] = res[data_type]
                
        self._save_dataset(factor_dataset)
        
        self.factor_dataset = factor_dataset
        
    def _save_dataset(self, factor_dataset):
        with open(self.save_dir / 'dataset.pkl', 'wb') as file:
            pickle.dump(factor_dataset, file)
            
    def _load_dataset(self):
        dataset_path = self.save_dir / 'dataset.pkl'
        if os.path.exists(dataset_path):
            try:
                with open(dataset_path, 'rb') as file:
                    factor_dataset = pickle.load(file)
            except:
                factor_dataset = defaultdict(dict)
        else:
            factor_dataset = defaultdict(dict)
        return factor_dataset
            
    def _load_factor_data(self, factor_name, test_data_dir, data_to_load):
        f_dataset = {}
        for data_type in data_to_load:
            try:
                f_dataset[data_type] = self._load_one_data_type(data_type, factor_name, test_data_dir)
            except Exception:
                f_dataset[data_type] = None
        return f_dataset

        
    def _load_one_data_type(self, data_type, factor_name, test_data_dir):
        if data_type in ['gp', 'icd', 'hsr']:
            return pd.read_parquet(test_data_dir / f'{data_type}_{factor_name}.parquet')
        elif data_type in ['bins']:
            with open(test_data_dir / f'{data_type}_{factor_name}.pkl', 'rb') as file:
                data = pickle.load(file)
            return data
        
    def generate_all(self):
        # generate func
        self.generate_func = partial(_generate, rolling_choices=self.rolling_choice, save_dir=self.save_dir)
        # corr filter
        corr_filter_fac = self._load_or_generate_corr_filter()
        # direction
        direction_fac = self._load_or_generate_direction_fac()
        # features
        self._parallel_generate(corr_filter_fac, direction_fac)
        
    def _load_or_generate_corr_filter(self):
        corr_filter_path = self.save_dir / 'corr_filter.parquet'
        if os.path.exists(corr_filter_path):
            corr_filter_fac = pd.read_parquet(corr_filter_path)
        else:
            filter_params = self.params['corr_filter']
            valid_prop_thresh = filter_params['valid_prop_thresh']
            corr_thresh = filter_params['corr_thresh']
            filter_gp = filter_params['filter_gp']
            func_one_period = partial(filter_correlated_factors, valid_prop_thresh=valid_prop_thresh,
                                      corr_thresh=corr_thresh, filter_gp=filter_gp)
            corr_filter_fac = self.generate_func(func_one_period=func_one_period, 
                                                 factor_dataset=self.factor_dataset['gp'], 
                                                 name_to_save='corr_filter', workers=self.calc_workers)
        return corr_filter_fac
        
    def _load_or_generate_direction_fac(self):
        direction_path = self.save_dir / 'direction.parquet'
        if os.path.exists(direction_path):
            direction_fac = pd.read_parquet(direction_path)
        else:
            direction_fac = self.generate_func(func=fac_direction, factor_dataset=self.factor_dataset['gp'], 
                                               name_to_save='direction', workers=self.calc_workers)
        return direction_fac
        
    def _parallel_generate(self, corr_filter_fac, direction_fac):
        feature_params = self.params['feature']
        lookback_wd_list = feature_params['lookback_window']
        predict_wd_list = feature_params['predict_window']
        gp_params = feature_params['gp']
        icd_params = feature_params['icd']
        bins_params = feature_params['bins']
        predict_params = feature_params['predict'] # !!!: 暂时只用gp相关指标
        
        tasks = []
        func_one_period = partial(_process_one_period, corr_filter=corr_filter_fac)

        # GP Metrics
        gp_metrics = gp_params['metrics']
        gp_lag_list = gp_params['gp_lag']
        gp_metrics = [(metric_name, globals()[metric_func_name]) for metric_name, metric_func_name in gp_metrics]
        for lookback in lookback_wd_list:
            for gp_lag in gp_lag_list:
                for (metric_name, metric_func) in gp_metrics:
                    fac_gp_related_metrics_func = partial(fac_gp_related_metrics, 
                                                          direction_fac=direction_fac, 
                                                          metric_func=metric_func, gp_lag=gp_lag)
                    name_to_save = f'lb{lookback}_gp_{metric_name}_{gp_lag}'
                    tasks.append({
                        'func': fac_gp_related_metrics_func,
                        'func_one_period': func_one_period,
                        'factor_dataset': self.factor_dataset['gp'], 
                        'window': lookback,
                        'name_to_save': name_to_save
                    })
                    self.features_of_factors.append(name_to_save)

        # ICD Metrics
        icd_metrics = icd_params['metrics']
        ic_type_list = icd_params['ic_type']
        icd_metrics = [(metric_name, globals()[metric_func_name]) for metric_name, metric_func_name in icd_metrics]
        for lookback in lookback_wd_list:
            for ic_type in ic_type_list:
                for (metric_name, metric_func) in icd_metrics:
                    fac_icd_related_metrics_func = partial(fac_icd_related_metrics, 
                                                           direction_fac=direction_fac, 
                                                           metric_func=metric_func, ic_type=ic_type)
                    name_to_save = f'lb{lookback}_{ic_type}_{metric_name}'
                    tasks.append({
                        'func': fac_icd_related_metrics_func,
                        'func_one_period': func_one_period,
                        'factor_dataset': self.factor_dataset['icd'], 
                        'window': lookback,
                        'name_to_save': f'lb{lookback}_{ic_type}_{metric_name}'
                    })
                    self.features_of_factors.append(name_to_save)

        # HSR Metrics
        for lookback in lookback_wd_list:
            name_to_save = f'lb{lookback}_hsr'
            tasks.append({
                'func': fac_hsr_related_metrics,
                'func_one_period': func_one_period,
                'factor_dataset': self.factor_dataset['hsr'], 
                'window': lookback,
                'name_to_save': f'lb{lookback}_hsr'
            })
            self.features_of_factors.append(name_to_save)

        # Bins Metrics
        bins_metrics = bins_params['metrics']
        bins_metrics = [(metric_name, globals()[metric_func_name]) for metric_name, metric_func_name in bins_metrics]
        for lookback in lookback_wd_list:
            for (metric_name, metric_func) in bins_metrics:
                fac_bins_related_metrics_func = partial(fac_bins_related_metrics, direction_fac=direction_fac, 
                                                        metric_func=metric_func)
                name_to_save = f'lb{lookback}_bins_{metric_name}'
                tasks.append({
                    'func': fac_bins_related_metrics_func,
                    'func_one_period': func_one_period,
                    'factor_dataset': self.factor_dataset['bins'], 
                    'window': lookback,
                    'name_to_save': name_to_save
                })
                self.features_of_factors.append(name_to_save)

        # Predict Metrics
        predict_metrics = predict_params['metrics']
        predict_lag_list = predict_params['predict_lag']
        predict_metrics = [(metric_name, globals()[metric_func_name]) for metric_name, metric_func_name in predict_metrics]
        for pp in predict_wd_list:
            for predict_lag in predict_lag_list:
                for (metric_name, metric_func) in predict_metrics:
                    fac_gp_related_metrics_func = partial(fac_gp_related_metrics, direction_fac=direction_fac, 
                                                          metric_func=metric_func, gp_lag=predict_lag)
                    tasks.append({
                        'func': fac_gp_related_metrics_func,
                        'func_one_period': func_one_period,
                        'factor_dataset': self.factor_dataset['gp'], 
                        'window': pp,
                        'name_to_save': f'pred{pp}_{metric_name}_{predict_lag}',
                        'rolling_choice': 'predict'
                    })
                    
        # Predict Metrics
        for pp in predict_wd_list:
            name_to_save = f'pred{pp}_hsr'
            tasks.append({
                'func': fac_hsr_related_metrics,
                'func_one_period': func_one_period,
                'factor_dataset': self.factor_dataset['hsr'], 
                'window': pp,
                'name_to_save': f'pred{pp}_hsr',
                'rolling_choice': 'predict'
            })
                    
        # save features of factor
        self._save_features_of_factors()

        # 并行执行任务并处理异常
        with ProcessPoolExecutor(max_workers=self.calc_workers) as executor:  # 你可以调整 max_workers 的数量
            futures = [
                executor.submit(self.generate_func, **task) 
                for task in tasks
            ]

            for future in tqdm(as_completed(futures), total=len(futures), desc="Processing tasks"):
                try:
                    result = future.result()  # 获取任务的结果
                except Exception as e:
                    print(f"任务失败: {e}")  # 捕获并打印异常

    def _save_features_of_factors(self):
        path = self.save_dir / 'features_of_factors.pkl'
        if os.path.exists(path):
            with open(path, 'rb') as f:
                features_of_factors = pickle.load(f)
            features_of_factors.extend(self.features_of_factors)
        else:
            features_of_factors = self.features_of_factors
        with open(path, 'wb') as f:
            pickle.dump(features_of_factors, f)