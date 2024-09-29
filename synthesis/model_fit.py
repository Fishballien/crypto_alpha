# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 10:26:17 2024

@author: Xintang Zheng

"""
# %% imports
import os
import toml
from pathlib import Path
import glob
import json
import pandas as pd
import numpy as np
import numba as nb
from sklearn.impute import SimpleImputer
import joblib
from abc import ABC, abstractmethod
from tqdm import tqdm
import lightgbm as lgb
from sklearn.decomposition import PCA
from sklearn.linear_model import ElasticNet
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import roc_curve, auc, mean_squared_error
import shap
import queue
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import matplotlib.pyplot as plt
from functools import partial
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
import pickle
from sklearn.linear_model import LassoCV
from sklearn.linear_model import RidgeCV
from scipy import stats
import gc
from xgboost import XGBRanker


from utils.dirutils import load_path_config
from utils.timeutils import period_shortcut, timestr_to_minutes
from utils.datautils import get_one_factor, load_all_factors, load_one_group, align_index_with_main, replace_sp
from utils.datautils import qcut_row
from data_processing.feature_engineering import normalization
from test_and_eval.factor_tester_adaptive import FactorTest
from utils.speedutils import gc_collect_after
from synthesis.optuna_optimizer import *
from utils.logutils import FishStyleLogger
from synthesis.average_model import AverageModel


# %% select model
def choose_model(model_name):
    return globals()[model_name]


# %% funcs
def model_predict(model, X, y, y_data, normalization_func, to_mask, to_mask_ravel):
    y_pred_container = np.zeros_like(y_data.values.reshape(-1, order='F'))
    try:
        y_pred_selected = model.predict(X)
    except ValueError:
        breakpoint()
    y_pred_container[~to_mask_ravel] = y_pred_selected
    y_pred_container = y_pred_container.reshape(y_data.shape, order='F')
    y_pred = pd.DataFrame(y_pred_container, index=y_data.index, columns=y_data.columns)
    y_pred = normalization_func(y_pred.mask(to_mask))
    return y_pred

   
# %% model fit
class ModelFit(ABC):

    def __init__(self, model, test_name, preprocess_params={}, fit_params={}, predict_params={},
                 n_workers=1):
        self.model = model
        self.test_name = test_name
        self.preprocess_params = preprocess_params
        self.fit_params = fit_params
        self.predict_params = predict_params
        self.n_workers = n_workers
        
        self.log = FishStyleLogger()
        
        self._load_public_paths()
        self._init_dir()
        self._init_utils()
        
    def _load_public_paths(self):
        file_path = Path(__file__).resolve()
        project_dir = file_path.parents[1]
        self.path_config = load_path_config(project_dir)
        
        self.result_root_dir = Path(self.path_config['result'])
        self.data_dir = Path(self.path_config['processed_data'])
        self.twap_dir = Path(self.path_config['twap_price'])
        self.cluster_dir = Path(self.path_config['cluster'])
        self.dataset_dir = Path(self.path_config['dataset'])
        
    def _init_dir(self):
        self.result_dir = self.result_root_dir / 'model' / self.test_name
        self.result_dir.mkdir(parents=True, exist_ok=True)
        self.model_dir = self.result_dir / 'model'
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.predict_dir = self.result_dir / 'predict'
        self.predict_dir.mkdir(parents=True, exist_ok=True)
        
    def _init_utils(self):
        outlier = self.preprocess_params.get('outlier')
        self.normalization_func = partial(normalization, outlier_n=outlier)
        
    def set_filter_period(self, filter_start_date, filter_end_date):
        self.filter_period_suffix = period_shortcut(filter_start_date, filter_end_date)
        
    def fit_once(self, start_date, end_date, weight_func=None): # TODO: 考虑weight_func如何输入
        self.mode = 'fit'
        model_source_prefix = 'multi_'
        self.model_period_suffix = period_shortcut(start_date, end_date)
        self.log.info(f'Start Fit: {self.model_period_suffix}')
        self.model_suffix = f'{model_source_prefix}{self.model_period_suffix}'

        X, y, y_data = self._load_n_preprocess_x_y(start_date, end_date)

        return self._model_fit_and_save(X, y)
    
    @gc_collect_after
    def _load_n_preprocess_x_y(self, start_date, end_date):
        rtn_1p = self._load_n_preprocess_twap_return(start_date, end_date)
        factor_data = self._load_n_preprocess_factors(start_date, end_date, ref_order_col=rtn_1p.columns,
                                                      ref_index=rtn_1p.index)
        rtn_1p_ravel = rtn_1p.values.reshape(-1, order='F')
        self.to_mask_ravel = self.to_mask.values.reshape(-1, order='F')
        X = pd.DataFrame(factor_data[~self.to_mask_ravel, :], columns=self.features)
        y = rtn_1p_ravel[~self.to_mask_ravel]
        # if self.mode == 'predict':
        # breakpoint()
        return X, y, rtn_1p
    
    @gc_collect_after
    def _load_n_preprocess_twap_return(self, start_date, end_date):
        twap_name = self.preprocess_params.get('twap_name')
        pred_wd = self.preprocess_params.get('pred_wd')
        sp = self.preprocess_params.get('fit_sp') if self.mode == 'fit' else self.preprocess_params.get('predict_sp')
        
        twap_name = replace_sp(twap_name, sp)
        twap_path = self.twap_dir / f'{twap_name}.parquet'
        twap_price = pd.read_parquet(twap_path)
        
        sp_in_min = timestr_to_minutes(sp)
        pp_by_sp = int(pred_wd / sp_in_min)
        rtn_1p = twap_price.pct_change(pp_by_sp, fill_method=None
                                       ).shift(-pp_by_sp).replace([np.inf, -np.inf], np.nan)
        rtn_1p = rtn_1p[(rtn_1p.index >= start_date) & (rtn_1p.index < end_date)]
        self.rtn_1p = rtn_1p
        self.to_mask = rtn_1p.isna()
        rtn_1p = self.normalization_func(rtn_1p)
        rtn_1p = rtn_1p.fillna(0)
        # breakpoint()
        return rtn_1p
    
    @gc_collect_after
    def _load_n_preprocess_factors(self, start_date, end_date, ref_order_col, ref_index):
        cluster = self.preprocess_params.get('cluster')
        fix_changed_root = self.preprocess_params.get('fix_changed_root', False)

        if cluster is None:
            raise NotImplementedError()
        cluster_file_dir = self.cluster_dir / cluster
        cluster_file_path = cluster_file_dir / f'cluster_info_{self.filter_period_suffix}.csv'
        cluster_info = pd.read_csv(cluster_file_path)
        
        sp = self.preprocess_params.get('fit_sp') if self.mode == 'fit' else self.preprocess_params.get('predict_sp')
        get_one_factor_func = partial(get_one_factor, sp=sp, 
                                      date_start=start_date, date_end=end_date,
                                      ref_order_col=ref_order_col, ref_index=ref_index,
                                      fix_changed_root=fix_changed_root)
        
        factor_dict = load_all_factors(cluster_info, get_one_factor_func, self.data_dir, self.n_workers)
        
        load_one_group_func = partial(load_one_group, 
                                      normalization_func=self.normalization_func,
                                      to_mask=self.to_mask)
        
        factor_arr = None
        grouped = cluster_info.groupby('group')
        n_of_groups = len(grouped)
        assert max(list(grouped.groups.keys())) == n_of_groups
        group_names = [f'group_{num}' for num in range(1, n_of_groups+1)]
        
        if self.n_workers is None or self.n_workers == 1:
            for group_num, group_info in tqdm(grouped, desc='load_factors_by_group'):
                factor_dict_involved = {k: factor_dict[k] for k in group_info.index}
                group_num, group_factor = load_one_group_func(group_num, group_info, factor_dict=factor_dict_involved)
                group_idx = group_num - 1
                if group_factor is not None:
                    if factor_arr is None:
                        len_of_one_factor = np.multiply(*group_factor.shape)
                        factor_arr = np.zeros((len_of_one_factor, n_of_groups), dtype=np.float64)
                    factor_arr[:, group_idx] = group_factor.values.reshape(-1, order='F')
                gc.collect()
        else:
            task_params = []
            for group_num, group_info in tqdm(grouped, desc='split factor dict'):
                factor_dict_involved = {k: factor_dict[k] for k in group_info.index}
                task_params.append((group_num, group_info, factor_dict_involved))
            with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
                all_tasks = [executor.submit(load_one_group_func, group_num, group_info, factor_dict=factor_dict_involved)
                             for group_num, group_info, factor_dict_involved in task_params]
                # res_queue = queue.Queue()
                for task in tqdm(as_completed(all_tasks), total=len(all_tasks), desc='load_factors_by_group'):
                    res = task.result()
                    if res is not None:
                        group_num, group_factor = res
                        group_idx = group_num - 1
                        if group_factor is not None:
                            if factor_arr is None:
                                len_of_one_factor = np.multiply(*group_factor.shape)
                                factor_arr = np.zeros((len_of_one_factor, n_of_groups), dtype=np.float64)
                            factor_arr[:, group_idx] = group_factor.values.reshape(-1, order='F')
                        del group_factor
                        gc.collect()
        self.features = group_names

        return factor_arr
    
# =============================================================================
#     @gc_collect_after
#     def _load_n_preprocess_factors(self, start_date, end_date, ref_order_col, ref_index):
#         cluster = self.preprocess_params.get('cluster')
# 
#         if cluster is None:
#             raise NotImplementedError()
#         cluster_file_dir = self.cluster_dir / cluster
#         cluster_file_path = cluster_file_dir / f'cluster_info_{self.filter_period_suffix}.csv'
#         cluster_info = pd.read_csv(cluster_file_path)
#         
#         sp = self.preprocess_params.get('fit_sp') if self.mode == 'fit' else self.preprocess_params.get('predict_sp')
#         get_one_factor_func = partial(get_one_factor, sp=sp, 
#                                       date_start=start_date, date_end=end_date,
#                                       ref_order_col=ref_order_col, ref_index=ref_index)
#         load_one_group_func = partial(load_one_group, get_one_factor_func=get_one_factor_func,
#                                       normalization_func=self.normalization_func,
#                                       factor_data_dir=self.data_dir, to_mask=self.to_mask)
#         
#         factor_arr = None
#         grouped = cluster_info.groupby('group')
#         n_of_groups = len(grouped)
#         assert max(list(grouped.groups.keys())) == n_of_groups
#         group_names = [f'group_{num}' for num in range(1, n_of_groups+1)]
#         
#         if self.n_workers is None or self.n_workers == 1:
#             for group_num, group_info in tqdm(grouped, desc='load_factors_by_group'):
#                 group_num, group_factor = load_one_group_func(group_num, group_info)
#                 group_idx = group_num - 1
#                 if group_factor is not None:
#                     if factor_arr is None:
#                         len_of_one_factor = np.multiply(*group_factor.shape)
#                         factor_arr = np.zeros((len_of_one_factor, n_of_groups), dtype=np.float64)
#                     factor_arr[:, group_idx] = group_factor.values.reshape(-1, order='F')
#                 gc.collect()
#         else:
#             with ThreadPoolExecutor(max_workers=self.n_workers) as executor:
#                 all_tasks = [executor.submit(load_one_group_func, group_num, group_info)
#                              for group_num, group_info in grouped]
#                 # res_queue = queue.Queue()
#                 for task in tqdm(as_completed(all_tasks), total=len(all_tasks), desc='load_factors_by_group'):
#                     res = task.result()
#                     if res is not None:
#                         group_num, group_factor = res
#                         group_idx = group_num - 1
#                         if group_factor is not None:
#                             if factor_arr is None:
#                                 len_of_one_factor = np.multiply(*group_factor.shape)
#                                 factor_arr = np.zeros((len_of_one_factor, n_of_groups), dtype=np.float64)
#                             factor_arr[:, group_idx] = group_factor.values.reshape(-1, order='F')
#                         del group_factor
#                         gc.collect()
#         self.features = group_names
# 
#         return factor_arr
# =============================================================================

    @abstractmethod
    def _model_fit_and_save(self, **kwargs):
        pass
    
    def predict_once(self, model_start_date, model_end_date, 
                     pred_start_date, pred_end_date, source='multi', symbol=None):
        self.mode = 'predict'
        model_source_prefix = f'{source}_'
        self.model_period_suffix = period_shortcut(model_start_date, model_end_date)
        self.model_suffix = f'{model_source_prefix}{self.model_period_suffix}'
        self.predict_suffix = period_shortcut(pred_start_date, pred_end_date)
        self.log.info(f'Start Predict: {self.predict_suffix}')
        
        model = self._model_load()
        X, y, y_data = self._load_n_preprocess_x_y(pred_start_date, pred_end_date)
        self._model_predict(model, X, y, y_data)
    
    @abstractmethod
    def _model_load(self):
        pass
        
    @abstractmethod
    def _model_predict(self, model, X, y, y_data):
        pass 
    
    def _save_predict_and_merge(self, y_pred):
        y_pred.to_parquet(self.predict_dir / f'predict_{self.predict_suffix}_by_{self.model_suffix}.parquet')
        
        pred_all_path = self.predict_dir / f'predict_{self.test_name}.parquet'
        if os.path.exists(pred_all_path):
            pred_all = pd.read_parquet(pred_all_path)
        else:
            pred_all = pd.DataFrame()
        pred_all = pd.concat([pred_all, y_pred])
        pred_all.to_parquet(pred_all_path)
    
    def test_predicted(self):
        process_name = None
        factor_data_dir = self.predict_dir
        result_dir = self.predict_dir
        params = self.predict_params
        
        ft = FactorTest(process_name, None, factor_data_dir, result_dir=result_dir, params=params)
        ft.test_one_factor(f'predict_{self.test_name}')
        
    def compare_model_with_factors(self):
         cluster = self.preprocess_params.get('cluster')
         ftest_dir = self.result_root_dir / 'test'

         if cluster is None:
             raise NotImplementedError()
         cluster_file_dir = self.cluster_dir / cluster
         selected_factor_list = []
         for cluster_file in os.listdir(cluster_file_dir):
             cluster_file_path = cluster_file_dir / cluster_file
             if os.path.isdir(cluster_file_path):
                 continue
             cluster_info = pd.read_csv(cluster_file_path)
             period_selected = [row for row in 
                                cluster_info[['test_name', 'tag_name', 'process_name', 'factor', 'direction']
                                             ].itertuples(index=False, name=None)]
             selected_factor_list.extend(period_selected)
         unq_selected_factors = list(set(selected_factor_list))
         
         # plot
         FONTSIZE_L1 = 20
         FONTSIZE_L2 = 18
         FONTSIZE_L3 = 15
         
         title = f'{self.test_name} vs factors'
         
         fig = plt.figure(figsize=(36, 27), dpi=100, layout="constrained")
         spec = fig.add_gridspec(ncols=47, nrows=23)
         
         ax0 = fig.add_subplot(spec[:, :])
         ax0.set_title(f'{title}', fontsize=FONTSIZE_L1, pad=25)
         test_res_dir = self.predict_dir / 'data'
         df_gp_test = pd.read_parquet(test_res_dir / f'gp_predict_{self.test_name}.parquet')
         ax0.plot(df_gp_test['long_short_0'].cumsum(), color='k', linewidth=5, label=f'{self.test_name}')
         
         for test_name, tag_name, process_name, factor_name, direction in unq_selected_factors:
             data_dir = ftest_dir / test_name / tag_name / process_name / 'data'
             df_gp = pd.read_parquet(data_dir / f'gp_{factor_name}.parquet')
             df_gp = align_index_with_main(df_gp_test.index, df_gp)
             ax0.plot((df_gp['long_short_0'] * direction).cumsum(), alpha=.5)
             

         for ax in [ax0,]:
             ax.grid(linestyle=":")
             ax.legend(loc="upper left", borderaxespad=0.5, fontsize=FONTSIZE_L3)
             ax.tick_params(labelsize=FONTSIZE_L2, pad=15)
         
         name_to_save = f'{self.test_name}_vs_factors'
         plt.savefig(self.predict_dir / f"{name_to_save}.jpg", dpi=100, bbox_inches="tight")
         plt.close()
        
    
# %% Linear Regression
class LinearRegressionFit(ModelFit):
    
    def _model_fit_and_save(self, X, y):
        train_params = self.fit_params['train']
        cv_params = self.fit_params.get('cv', {})
        alpha_cv = cv_params.get('alpha_cv', False)
        
        if alpha_cv:
            best_alpha = self._alpha_cv(X, y)
            train_params['alpha'] = best_alpha
            
        self.log.info('Start training...')
        self.log.info(train_params)
        model = self.model(**train_params)
        # try:
        model.fit(X, y)
        # except:
        #     breakpoint()
        self.log_model_info(model=model)
        score = model.score(X, y)
        self.log.info(f'is score: {score}')
        self._save_model(model)
        self.log.success('Model saved!')
# =============================================================================
#         self.log.info(f'y: {y}')
#         self.log.info(f'y_pred: {y_pred}')
# =============================================================================

    def _alpha_cv(self, X, y):
        pass
    
    def log_model_info(self, model=None, model_start_date=None, model_end_date=None, symbol=None):
        if model_start_date is not None and model_end_date is not None:
            model_source_prefix = 'multi_' if symbol is None or isinstance(symbol, list) else f'{symbol}_'
            self.model_period_suffix = period_shortcut(model_start_date, model_end_date)
            self.model_suffix = f'{model_source_prefix}{self.model_period_suffix}'
        
        if model is None:
            model = self._model_load()
        self._save_model_info(model)
        self.log.info(f'coef: {list(zip(self.features, model.coef_))}')
        self.log.info(f'intercept: {model.intercept_}')
        
    def _save_model(self, model):
        joblib.dump(model, self.model_dir / f'model_{self.model_suffix}.pkl')

    def _model_load(self):
        model = joblib.load(self.model_dir / f'model_{self.model_suffix}.pkl')
        return model
    
    def _save_model_info(self, model):
        # 将系数和特征名称保存到文本文件
        with open(self.model_dir / f'model_info_{self.model_suffix}.txt', 'w') as file:
            file.write("Feature Coefficients:\n")
            for name, coef in zip(self.features,  model.coef_):
                file.write(f"{name}: {coef}\n")
        
    def _model_predict(self, model, X, y, y_data):
        score = model.score(X, y)
        self.log.info(f'oos score: {score}')
        y_pred = model_predict(model, X, y, y_data, self.normalization_func, 
                               self.to_mask, self.to_mask_ravel)
        self._save_predict_and_merge(y_pred)
        
        
# %% ridge
class RidgeFit(LinearRegressionFit):
    
    def __init__(self, test_name, **kwargs):
        model = Ridge
        super().__init__(model, test_name, **kwargs)
        
    def _alpha_cv(self, X, y):
        cv_params = self.fit_params.get('cv', {})
        alphas = cv_params.get('alphas', None)
        
        model = RidgeCV(cv=5, alphas=alphas)
        model.fit(X, y)
        return model.alpha_
        

# %% lasso
class LassoFit(LinearRegressionFit):
    
    def __init__(self, test_name, **kwargs):
        model = Lasso
        super().__init__(model, test_name, **kwargs)
        

# %% lgbm
class LGBMFit(ModelFit):
    
    def __init__(self, test_name, **kwargs):
        model = lgb
        super().__init__(model, test_name, **kwargs)
        self._init_best_iteration_dict()
        
    def _init_best_iteration_dict(self):
        best_iteration = {}
        with open(self.model_dir / 'best_iteration.pkl', 'wb') as f:
            pickle.dump(best_iteration, f)
            
    def _load_best_iteration(self):
        with open(self.model_dir / 'best_iteration.pkl', 'rb') as f:
            best_iteration = pickle.load(f)
        return best_iteration
    
    def _add_to_best_iteration(self, model_suffix, best_iteration_v):
        best_iteration = self._load_best_iteration()
        best_iteration[model_suffix] = best_iteration_v
        with open(self.model_dir / 'best_iteration.pkl', 'wb') as f:
            pickle.dump(best_iteration, f)
            
    def _read_target_best_iteration(self, model_suffix):
        best_iteration = self._load_best_iteration()
        return best_iteration.get(model_suffix)
        
    def fit_once(self, start_date, end_date, weight_func=None):
        self.mode = 'fit'
        model_source_prefix = 'multi_'
        self.model_period_suffix = period_shortcut(start_date, end_date)
        self.log.info(f'Start Fit: {self.model_period_suffix}')
        self.model_suffix = f'{model_source_prefix}{self.model_period_suffix}'
        train_data = self._load_dataset(start_date, end_date, weight_func=weight_func)

        return self._model_fit_and_save(train_data=train_data)
    
    def _load_dataset(self, start_date, end_date, weight_func=None):
        self.train_params = self.fit_params['train']
        max_bin = self.train_params.get('max_bin', 255)
        force_reload = self.preprocess_params.get('force_reload', False)
        pred_wd = self.preprocess_params.get('pred_wd')
        sp = self.preprocess_params.get('fit_sp') if self.mode == 'fit' else self.preprocess_params.get('predict_sp')
        cluster = self.preprocess_params.get('cluster', 'no_cluster')
        
        weight_func_name = f'_{weight_func.__name__}' if weight_func is not None else ''
        dataset_name = f'{self.OBJ}_data_{self.model_suffix}_pred_{int(pred_wd)}_sp_{sp}_mb_{max_bin}_cl_{cluster}{weight_func_name}.bin'
        possible_model_path = self.dataset_dir / dataset_name
        if os.path.exists(possible_model_path) and (not force_reload):
            train_data = lgb.Dataset(possible_model_path)
            self.log.success(f'Load the existing dataset: {dataset_name}')
        else:
            self.log.info('Dataset not found. Loading new...')
            # self.periods = get_period(start_date, end_date)
            X, y, data = self._load_n_preprocess_x_y(start_date, end_date)
            self.log.info(f'y mean: {np.mean(y)}')
            # breakpoint()
            # self.X = X
            # feature_list = list(X.columns)
            weight = weight_func(data) if weight_func is not None else None
            train_data = lgb.Dataset(
                X, label=y, 
                feature_name=list(self.features),
                params={'max_bin': max_bin},
                weight=weight,
                )
            train_data.save_binary(possible_model_path)
        train_data.construct()
        return train_data
    
    def _model_fit_and_save(self, train_data):
        self.train_params = self.fit_params['train']
        self.cv_params = self.fit_params.get('cv')
        if self.cv_params is not None:
            params = self.cv_params['params']
            kwargs = self.cv_params['kwargs']
            cv_results = lgb.cv(
                params=params, 
                train_set=train_data, 
                **kwargs,
                )
            best_iteration = len(cv_results["valid l2-mean"])
            self.log.info(f'Best Iteration: {best_iteration}')
            self._add_to_best_iteration(self.model_suffix, best_iteration)
        self.log.info('Start training...')
        self.log.info(self.train_params)
        results = {}
        # breakpoint()
        model = lgb.train(
            self.train_params, 
            train_data,
            callbacks=[
                lgb.log_evaluation(period=10),
                lgb.record_evaluation(results)
                ],
            )
        model.save_model(self.model_dir / f'model_{self.model_suffix}.txt')
        self.log.success('Model saved!')
        # self._plot_metric(results)
        self._save_split_and_gain(model)
        # self._shap(model, train_data)
        
    def _plot_metric(self, results):
        lgb.plot_metric(results)
        plt.savefig(self.model_dir / 'loss_curve_{self.model_suffix}.jpg', dpi=300) 
    
    def plot_trees(self, model_start_date, model_end_date, source='multi'):
        model_source_prefix = f'{source}_'
        model_source_prefix = 'multi_'
        self.model_period_suffix = period_shortcut(model_start_date, model_end_date)
        self.model_suffix = f'{model_source_prefix}{self.model_period_suffix}'
        model = self._model_load()
        
        tree_dir = self.model_dir / f'trees_{self.model_suffix}'
        tree_dir.mkdir(parents=True, exist_ok=True)
        for tree_index in range(0, model.num_trees(), 1):
            ax = lgb.create_tree_digraph(model, tree_index=tree_index) # leaf中的值为叶子节点中样本平均值
            outpath = tree_dir / rf"tree_{tree_index}.jpg"
            ax.render(outfile=str(outpath), format='jpg', view=False)
            
    def _save_split_and_gain(self, model):
        split = model.feature_importance()
        gain = model.feature_importance('gain')
        split_and_gain = pd.DataFrame({'feature': model.feature_name(), 'split': split, 'gain': gain})
        split_and_gain.to_csv(self.model_dir / rf"split_and_gain_{self.model_suffix}.csv", index=None)
        self.log.success('Split and gain saved!')
        
# =============================================================================
#     def _shap(self, model, train_data):
#         breakpoint()
#         # 创建一个解释器
#         explainer = shap.TreeExplainer(model)
#         
#         # 计算SHAP值
#         shap_values = explainer.shap_values(self.X)
#         
#         shap.plots.heatmap(shap_values[1:100])
#         shap.plots.waterfall(shap_values[0])
#         # 创建摘要性状图
#         shap.summary_plot(shap_values, self.X, plot_type="bar")
#         plt.savefig(self.model_dir / 'summary_plot.jpg')
#         
#         # 创建单个特征的SHAP值摘要图
#         shap.summary_plot(shap_values, self.X)
#         plt.savefig(self.model_dir / 'summary_plot_single_feature.jpg')
#         
#         # # 创建SHAP水平瀑布图
#         # shap.initjs()
#         # force_plot = shap.force_plot(explainer.expected_value, shap_values[0,:], self.X.iloc[0,:])
#         # shap.save_html(self.model_dir / 'force_plot.html', force_plot)
#     
# =============================================================================
    def _model_load(self):
        model = lgb.Booster(model_file=self.model_dir / f'model_{self.model_suffix}.txt')
        return model
        
    def _model_predict(self, model, X, y, y_data):
        cv_params = self.fit_params.get('cv')
        
        best_iteration = self._read_target_best_iteration(self.model_suffix) if cv_params is not None else None
        y_pred_container = np.zeros_like(y_data.values.reshape(-1, order='F'))
        try:
            if best_iteration is not None:
                y_pred_selected = model.predict(X, num_iteration=best_iteration) # , num_iteration=model.best_iteration
            else:
                y_pred_selected = model.predict(X)
        except ValueError:
            breakpoint()
        # score = model.score(X, y)
        # self.log.info(f'oos score: {score}')
        y_pred_container[~self.to_mask_ravel] = y_pred_selected
        y_pred_container = y_pred_container.reshape(y_data.shape, order='F')
        y_pred = pd.DataFrame(y_pred_container, index=y_data.index, columns=y_data.columns)
        y_pred = self.normalization_func(y_pred.mask(self.to_mask)) #.fillna(0)
        # breakpoint()
        y_pred.to_parquet(self.predict_dir / f'predict_{self.predict_suffix}_by_{self.model_suffix}.parquet')
        
        self._save_predict_and_merge(y_pred)
        
                
# %% LGBM regression
class LGBMRegression(LGBMFit):
    
    OBJ = 'regression'
    

# %%
def classify_v0(x):
    if np.isnan(x):
        return x
    if x >= 0.8:
        return 2
    elif x <= 0.2:
        return 0
    else:
        return 1
    
    
class XGBRankerFit(ModelFit):
    
    def __init__(self, test_name, **kwargs):
        model = XGBRanker
        super().__init__(model, test_name, **kwargs)
        
    @gc_collect_after
    def _load_n_preprocess_x_y(self, start_date, end_date):
        cut_groups = self.preprocess_params.get('cut_groups')
        
        rtn_1p_raw = self._load_n_preprocess_twap_return(start_date, end_date)
        # rtn_1p = rtn_1p_raw.mask(self.to_mask).apply(lambda r: qcut_row(r, q=cut_groups), axis=1)
        # print(rtn_1p)
        factor_rank = rtn_1p_raw.mask(self.to_mask).rank(axis=1, pct=True)
        rtn_1p = factor_rank.applymap(classify_v0)
        print(rtn_1p)
        
        factor_data = self._load_n_preprocess_factors(start_date, end_date, ref_order_col=rtn_1p.columns,
                                                      ref_index=rtn_1p.index)
        rtn_1p_ravel = rtn_1p.values.reshape(-1, order='F')
        self.to_mask_ravel = self.to_mask.values.reshape(-1, order='F')
        X = factor_data[~self.to_mask_ravel, :]
        y = rtn_1p_ravel[~self.to_mask_ravel].astype(int)
        self.groups = self._get_groups()
        self.sample_weights = self._get_sample_weight(rtn_1p_raw, self.groups)
        return X, y, rtn_1p
    
    def _get_groups(self):
        groups = (~self.to_mask).sum(axis=1)
        assert np.sum(groups) == np.sum(~self.to_mask_ravel)
        return groups
    
    def _get_sample_weight(self, rtn_1p_raw, groups):
        rtn_std = rtn_1p_raw.std(axis=1)
        assert len(rtn_std) == len(groups)
# =============================================================================
#         sample_weights = []
#         for std, group_len in list(zip(rtn_std, groups)):
#             sample_weights.extend([std]*group_len)
# =============================================================================
        return rtn_std
    
    def _model_fit_and_save(self, X, y):
        train_params = self.fit_params['train']
        if_sample_weight = self.fit_params.get('if_sample_weight', False)
            
        self.log.info('Start training...')
        self.log.info(train_params)
        model = self.model(**train_params)
        
        sample_weight = self.sample_weights if if_sample_weight else None
        model.fit(X, y, group=self.groups, sample_weight=sample_weight)

        self._save_model(model)
        self.log.success('Model saved!')
        
    def _save_model(self, model):
        joblib.dump(model, self.model_dir / f'model_{self.model_suffix}.pkl')

    def _model_load(self):
        model = joblib.load(self.model_dir / f'model_{self.model_suffix}.pkl')
        return model
        
    def _model_predict(self, model, X, y, y_data):
        y_pred_container = np.zeros_like(y_data.values.reshape(-1, order='F'))
        try:
            y_pred_selected = model.predict(X)
        except ValueError:
            breakpoint()

        y_pred_container[~self.to_mask_ravel] = y_pred_selected
        y_pred_container = y_pred_container.reshape(y_data.shape, order='F')
        y_pred = pd.DataFrame(y_pred_container, index=y_data.index, columns=y_data.columns)
        print(y_pred)
        y_pred = self.normalization_func(y_pred.mask(self.to_mask)) #.fillna(0)
        # breakpoint()
        y_pred.to_parquet(self.predict_dir / f'predict_{self.predict_suffix}_by_{self.model_suffix}.parquet')
        
        pred_all_path = self.predict_dir / 'predict.parquet'
        if os.path.exists(pred_all_path):
            pred_all = pd.read_parquet(pred_all_path)
        else:
            pred_all = pd.DataFrame()
        pred_all = pd.concat([pred_all, y_pred])
        print(pred_all)
        pred_all.to_parquet(pred_all_path)


# %% Ensemble model
class EnsembleModelFit(ModelFit):
    
    def __init__(self, test_name, **kwargs):
        model = EnsembleModelOptimizer
        super().__init__(model, test_name, **kwargs)
        
    def fit_once(self, start_date, end_date, weight_func=None): # TODO: 考虑weight_func如何输入
        self.mode = 'fit'
        model_source_prefix = 'multi_'
        self.model_period_suffix = period_shortcut(start_date, end_date)
        self.log.info(f'Start Fit: {self.model_period_suffix}')
        self.model_suffix = f'{model_source_prefix}{self.model_period_suffix}'

        X, y, y_data = self._load_n_preprocess_x_y(start_date, end_date)

        return self._model_fit_and_save(X, y, y_data)
        
    def _model_fit_and_save(self, X, y, y_data):
        train_params = self.fit_params['train']
        processed_train_params = self._init_train_params(y_data)
        processed_train_params['study_name'] = f'{self.test_name}_{self.model_suffix}'
        
        self.log.info('Start training...')
        self.log.info(train_params)
        model = self.model(**processed_train_params)

        model.fit(X, y)

        self.log_model_info(model=model)
        score = model.score(X, y)
        self.log.info(f'is score: {score}')
        self._save_model(model)
        self.log.success('Model saved!')

    def _init_train_params(self, y_data):
        train_params = self.fit_params['train']
        
        processed_train_params = {k: v for k, v in train_params.items()}
        
        # save & load
        model_dir = self.model_dir / self.model_suffix
        model_dir.mkdir(exist_ok=True, parents=True)
        save_model_func = partial(save_model, model_dir=model_dir, model_name_prefix='model')
        load_model_func = partial(load_model, model_dir=model_dir, model_name_prefix='model')
        
        # objective
        objective_func_name = train_params['objective']
        objective_org_func = globals()[objective_func_name]
        model_predict_func = partial(model_predict, y_data=y_data, 
                                     normalization_func=self.normalization_func, 
                                     to_mask=self.to_mask, to_mask_ravel=self.to_mask_ravel)
        objective_func = partial(objective_org_func, rtn=self.rtn_1p, 
                                 model_predict_func=model_predict_func,
                                 save_model_func=save_model_func)
        processed_train_params['objective'] = objective_func
        
        # select
        select_func_info = train_params['select_func']
        select_func_name = select_func_info['name']
        select_func_params = select_func_info.get('params', {})
        select_org_func = globals()[select_func_name]
        select_func = partial(select_org_func, **select_func_params)
        processed_train_params['select_func'] = select_func
        
        # directions
        directions_name = train_params['directions']
        directions = globals()[directions_name]
        processed_train_params['directions'] = directions
        
        # load model
        processed_train_params['load_model_func'] = load_model_func
        
        return processed_train_params
    
    def log_model_info(self, model=None, model_start_date=None, model_end_date=None, symbol=None):
        if model_start_date is not None and model_end_date is not None:
            model_source_prefix = 'multi_' if symbol is None or isinstance(symbol, list) else f'{symbol}_'
            self.model_period_suffix = period_shortcut(model_start_date, model_end_date)
            self.model_suffix = f'{model_source_prefix}{self.model_period_suffix}'
        
        if model is None:
            model = self._model_load()
        self._save_best_params(model)
        
    def _save_model(self, model):
        joblib.dump(model, self.model_dir / f'model_{self.model_suffix}.pkl')

    def _model_load(self):
        model = joblib.load(self.model_dir / f'model_{self.model_suffix}.pkl')
        return model
    
    def _save_best_params(self, model):
        best_params_list = model.best_params_list
        
        save_path = self.model_dir / f'best_params_{self.model_suffix}.json'
        with open(save_path, 'w') as file:
            json.dump(best_params_list, file, indent=4)
        self.log.info(f"Best parameters: {best_params_list}")
        
    def _model_predict(self, model, X, y, y_data):
        score = model.score(X, y)
        self.log.info(f'oos score: {score}')
        y_pred = model_predict(model, X, y, y_data, self.normalization_func, 
                               self.to_mask, self.to_mask_ravel)
        self._save_predict_and_merge(y_pred)
        
        
# %% Ensemble model
class AverageModelFit(ModelFit):
    
    def __init__(self, test_name, **kwargs):
        model = AverageModel
        super().__init__(model, test_name, **kwargs)
        
    def fit_once(self, start_date, end_date, weight_func=None): # TODO: 考虑weight_func如何输入
        self.mode = 'fit'
        model_source_prefix = 'multi_'
        self.model_period_suffix = period_shortcut(start_date, end_date)
        self.log.info(f'Start Fit: {self.model_period_suffix}')
        self.model_suffix = f'{model_source_prefix}{self.model_period_suffix}'

        return self._model_fit_and_save()
        
    def _model_fit_and_save(self):
        self.log.info('Start training...')
        model = self.model()
        self._save_model(model)
        self.log.success('Model saved!')
        
    def _save_model(self, model):
        joblib.dump(model, self.model_dir / f'model_{self.model_suffix}.pkl')

    def _model_load(self):
        model = joblib.load(self.model_dir / f'model_{self.model_suffix}.pkl')
        return model

    def _model_predict(self, model, X, y, y_data):
        score = model.score(X, y)
        self.log.info(f'oos score: {score}')
        y_pred = model_predict(model, X, y, y_data, self.normalization_func, 
                               self.to_mask, self.to_mask_ravel)
        self._save_predict_and_merge(y_pred)
        

        
