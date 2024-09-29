# -*- coding: utf-8 -*-
"""
Created on Fri Jul 12 17:31:06 2024

@author: Xintang Zheng

星星: ★ ☆ ✪ ✩ 🌟 ⭐ ✨ 🌠 💫 ⭐️
勾勾叉叉: ✓ ✔ ✕ ✖ ✅ ❎
报警啦: ⚠ ⓘ ℹ ☣
箭头: ➔ ➜ ➙ ➤ ➥ ↩ ↪
emoji: 🔔 ⏳ ⏰ 🔒 🔓 🛑 🚫 ❗ ❓ ❌ ⭕ 🚀 🔥 💧 💡 🎵 🎶 🧭 📅 🤔 🧮 🔢 📊 📈 📉 🧠 📝

"""
# %% imports
import optuna
import lightgbm as lgb
from sklearn.linear_model import Ridge, ElasticNet
from group_lasso import GroupLasso
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import mean_squared_error
import numpy as np
import warnings
from enum import Enum
from datetime import datetime
import joblib


from test_and_eval.scores import get_general_return_metrics
from test_and_eval.test_functions import calc_rank, calc_gp, calc_hsr, calc_ic
from synthesis.modelutils import group_features_by_correlation


# %% mapping
class ModelType(Enum): # 定义模型映射枚举类
    LGBM = lgb.LGBMRegressor
    RIDGE = Ridge
    ELASTICNET = ElasticNet
    GROUPLASSO = GroupLasso


class SamplerType(Enum): # 定义采样器映射枚举类
    TPE = optuna.samplers.TPESampler
    RANDOM = optuna.samplers.RandomSampler
    CMAES = optuna.samplers.CmaEsSampler
    NSGAII = optuna.samplers.NSGAIISampler


# %% model
class EnsembleModelOptimizer(BaseEstimator, RegressorMixin):
    
    def __init__(self, study_name, objective, select_func, load_model_func, 
                 n_trials=100, sampler_name='RANDOM', directions=None):
        self.study_name = study_name
        self.objective = objective
        self.select_func = select_func
        self.load_model_func = load_model_func
        self.n_trials = n_trials
        self.sampler_name = sampler_name
        self.directions = directions
        
        self.study = None
        self.models = []
        self.best_params_list = []

    def _get_sampler(self):
        try:
            SamplerClass = SamplerType[self.sampler_name].value
            return SamplerClass()
        except KeyError:
            raise ValueError(f"Unknown sampler name: {self.sampler_name}")

    def fit(self, X, y):
        sampler = self._get_sampler()
        if self.directions is None:
            raise ValueError("Directions for optimization must be provided.")
        study_name_unique = f"{self.study_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.study = optuna.create_study(directions=self.directions, sampler=sampler,
                                         storage='sqlite:///ensemble.db',  # 指定 SQLite 数据库
                                         study_name=study_name_unique,      # 指定研究名称
                                         load_if_exists=True,              # 如果已存在则加载
                                         )
        self.study.optimize(lambda trial: self.objective(trial, X, y), 
                            n_trials=self.n_trials)

        self.best_params_list = self.select_func(self.study, directions=self.directions)
        self._load_models()
        return self

# =============================================================================
#     def _fit_models(self, best_params_list, X, y):
#         self.models = []
#         for params in best_params_list:
#             model_name = params.get('model')
#             if not model_name:
#                 warnings.warn("Model name not found in params, skipping this model.")
#                 continue
#             try:
#                 ModelClass = ModelType[model_name].value
#             except KeyError:
#                 raise ValueError(f"Unknown model name: {model_name}")
#             model = ModelClass(**{k: v for k, v in params.items() if k != 'model'})
#             model.fit(X, y)
#             self.models.append(model)
# =============================================================================
            
    def _load_models(self):
        self.models = []
        for params in self.best_params_list:
            model_name = params.get('model')
            trial_number = params.get('trial_number')
            if not model_name or trial_number is None:
                warnings.warn("Model name or trial number not found in params, skipping this model.")
                continue
            try:
                model = self.load_model_func(trial_number)
                self.models.append(model)
            except FileNotFoundError:
                warnings.warn(f"Model for trial {trial_number} not found, skipping this model.")

    def predict(self, X):
        if not self.models:
            raise ValueError("Models have not been fit yet. Call `fit` first.")
        predictions = np.zeros((X.shape[0], len(self.models)))
        for i, model in enumerate(self.models):
            predictions[:, i] = model.predict(X)
        return np.mean(predictions, axis=1)
    
    
# %% save & load model
def save_model(model, trial_number, model_dir, model_name_prefix):
    model_path = model_dir / f"{model_name_prefix}_{trial_number}.joblib"
    joblib.dump(model, model_path)


def load_model(trial_number, model_dir, model_name_prefix):
    model_path = model_dir / f"{model_name_prefix}_{trial_number}.joblib"
    return joblib.load(model_path)


# %% select
def assign_pareto_fronts(trials, directions):
    """
    Assign Pareto fronts to each trial.
    :param trials: List of trials with objective values
    :param directions: List of directions ('minimize' or 'maximize') for each objective
    :return: List of Pareto front numbers for each trial
    """
    num_trials = len(trials)
    
    objectives = np.array([trial.values for trial in trials])
    is_minimize = np.array([1 if direction == 'minimize' else -1 for direction in directions])
    
    pareto_fronts = np.zeros(num_trials, dtype=int)
    
    current_front = 0
    remaining_indices = np.arange(num_trials)
    
    while remaining_indices.size > 0:
        pareto_mask = is_pareto_efficient(objectives[remaining_indices] * is_minimize)
        pareto_indices = remaining_indices[pareto_mask]
        
        pareto_fronts[pareto_indices] = current_front
        current_front += 1
        remaining_indices = remaining_indices[~pareto_mask]
    
    return pareto_fronts


def is_pareto_efficient(costs):
    """
    Find the Pareto-efficient points in the given costs array.
    :param costs: An (n_points, n_costs) array
    :return: A (n_points,) boolean array, indicating whether each point is Pareto efficient
    """
    is_efficient = np.ones(costs.shape[0], dtype=bool)
    for i, c in enumerate(costs):
        if is_efficient[i]:
            # 任何一个目标都必须不劣于其他点的所有目标，且至少有一个目标优于其他点的目标
            is_efficient[is_efficient] = np.any(costs[is_efficient] < c, axis=1) | np.all(costs[is_efficient] == c, axis=1)
            is_efficient[i] = True  # 保持自我
    return is_efficient


def get_pareto_front_params(study, directions=None):
    pareto_fronts = assign_pareto_fronts(study.trials, directions)
    
    pareto_front_params = []
    for trial, front in zip(study.trials, pareto_fronts):
        if front == 0:  # 只选取第一层的 Pareto 前沿
            params = trial.params
            params['trial_number'] = trial.number
            pareto_front_params.append(params)

    return pareto_front_params


def get_top_percent_params(study, directions=None, percent=0.3):
    all_trials = study.trials_dataframe()
    
    # 使用适当的列名
    if len(directions) > 1:
        for i in range(len(directions)):
            try:
                all_trials[f'objective{i+1}'] = all_trials[f'values_{i}'].apply(lambda x: x)
            except KeyError:
                print(all_trials)
                raise
    else:
        try:
            all_trials['objective1'] = all_trials['value'].apply(lambda x: x)
        except KeyError:
            print(all_trials)
            raise
    
    # 给每个 trial 分配 Pareto 层数
    pareto_fronts = assign_pareto_fronts(study.trials, directions)
    all_trials['pareto_front'] = pareto_fronts
    
    # 定义排序列和排序方向
    sort_columns = [f'objective{i+1}' for i in range(len(directions))]
    ascending = [True if direction == 'minimize' else False for direction in directions]
    
    # 按 Pareto 层数和目标进行排序
    sorted_trials = all_trials.sort_values(by=['pareto_front'] + sort_columns, ascending=[True] + ascending)

    top_percent_index = int(len(sorted_trials) * percent)
    top_percent_trials = sorted_trials.head(top_percent_index)
    top_percent_params = []
    for idx in top_percent_trials['number']:
        params = study.trials[int(idx)].params
        params['trial_number'] = int(idx)
        top_percent_params.append(params)
    return top_percent_params


# %% v0
def objective_v0(trial, X, y, *, rtn, model_predict_func, save_model_func):
    model_name = trial.suggest_categorical('model', ['LGBM', 'RIDGE'])
    
    if model_name == 'LGBM':
        max_depth = trial.suggest_int('max_depth', 2, 8)
        param = {
            'boosting_type': trial.suggest_categorical('boosting_type', ['gbdt', 'dart', 'goss']),
            'n_estimators': trial.suggest_int('n_estimators', 50, 700, step=10),
            'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.5, step=0.001), 
            'max_depth': max_depth,
            'num_leaves': trial.suggest_int('num_leaves', max(int(0.3 * 2**(max_depth-1)), 2*max_depth-2), max(int(0.75 * 2**max_depth)+1, 4*max_depth)),
            'min_child_samples': trial.suggest_int('min_child_samples', 1000, 6000, step=100),
            'max_bin': trial.suggest_int('max_bin', 60, 500, step=5),
            'verbosity': -1  # 关闭日志输出
        }
        model = lgb.LGBMRegressor(**param)
    elif model_name == 'RIDGE':
        param = {
            'alpha': trial.suggest_float('alpha', 0.1, 10000.0, log=True), # ???
            'fit_intercept': False
        }
        model = Ridge(**param)
    
    model.fit(X, y)
    save_model_func(model, trial.number)
    
    ## step 1: 获得predict结果
    factor = model_predict_func(model, X, y)
    # 统一mask（其实理论上应该是本来就一样的）
    to_mask = factor.isna() | rtn.isna()
    factor = factor.mask(to_mask)
    rtn = rtn.mask(to_mask)
    ## step 2: 结合rtn计算指标
    fct_n_pct = calc_rank(factor)
    gps, gpd = calc_gp(fct_n_pct, rtn)
    hsr = calc_hsr(fct_n_pct)
    metrics = get_general_return_metrics(gpd)
    sharpe_ratio = metrics['sharpe_ratio']
    margin = metrics['return_annualized'] / np.nanmean(hsr)
    
    return sharpe_ratio, margin


directions_v0 = ['maximize', 'maximize']


# %% v1
def objective_v1(trial, X, y, *, rtn, model_predict_func, save_model_func):
    model_name = trial.suggest_categorical('model', ['LGBM', 'RIDGE'])
    
    if model_name == 'LGBM':
        max_depth = trial.suggest_int('max_depth', 2, 8)
        param = {
            'boosting_type': trial.suggest_categorical('boosting_type', ['gbdt', 'dart', 'goss']),
            'n_estimators': trial.suggest_int('n_estimators', 50, 700, step=10),
            'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.5, step=0.001), 
            'max_depth': max_depth,
            'num_leaves': trial.suggest_int('num_leaves', max(int(0.3 * 2**(max_depth-1)), 2*max_depth-2), 
                                            max(int(0.75 * 2**max_depth)+1, 4*max_depth)),
            'min_child_samples': trial.suggest_int('min_child_samples', 1000, 6000, step=100),
            'max_bin': trial.suggest_int('max_bin', 60, 500, step=5),
            'verbosity': -1  # 关闭日志输出
        } # TODO: 添加随机拿特征
        model = lgb.LGBMRegressor(**param)
    elif model_name == 'RIDGE':
        param = {
            'alpha': trial.suggest_float('alpha', 0.1, 10000.0, log=True), # ???
            'fit_intercept': False
        }
        model = Ridge(**param)
    
    model.fit(X, y)
    save_model_func(model, trial.number)
    
    ## step 1: 获得predict结果
    factor = model_predict_func(model, X, y)
    # 统一mask（其实理论上应该是本来就一样的）
    to_mask = factor.isna() | rtn.isna()
    factor = factor.mask(to_mask)
    rtn = rtn.mask(to_mask)
    ## step 2: 结合rtn计算指标
    fct_n_pct = calc_rank(factor)
    gps, gpd = calc_gp(fct_n_pct, rtn)
    hsr = calc_hsr(fct_n_pct)
    metrics = get_general_return_metrics(gpd)
    sharpe_ratio = metrics['sharpe_ratio']
    avg_hsr = np.nanmean(hsr)
    
    return sharpe_ratio, avg_hsr


directions_v1 = ['maximize', 'minimize']


# %% v2
def objective_v2(trial, X, y, *, rtn, model_predict_func, save_model_func):
    model_name = trial.suggest_categorical('model', ['LGBM', 'RIDGE', 'ELASTICNET', 'GROUPLASSO']) # , 'GROUPLASSO'
    
# =============================================================================
#     if model_name != 'GROUPLASSO':
#         return (0, 0)
# =============================================================================
    
    if model_name == 'LGBM':
        max_depth = trial.suggest_int('max_depth', 2, 8)
        param = {
            'boosting_type': trial.suggest_categorical('boosting_type', ['gbdt', 'dart', 'goss']),
            'n_estimators': trial.suggest_int('n_estimators', 50, 700, step=10),
            'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.5, step=0.001), 
            'max_depth': max_depth,
            'num_leaves': trial.suggest_int('num_leaves', max(int(0.3 * 2**(max_depth-1)), 2*max_depth-2), max(int(0.75 * 2**max_depth)+1, 4*max_depth)),
            'min_child_samples': trial.suggest_int('min_child_samples', 1000, 6000, step=100),
            'max_bin': trial.suggest_int('max_bin', 60, 500, step=5),
            'verbosity': -1  # 关闭日志输出
        }
    elif model_name == 'RIDGE':
        param = {
            'alpha': trial.suggest_float('alpha', 0.1, 10000.0, log=True), # ???
            'fit_intercept': False
        }
    elif model_name == 'ELASTICNET':
        param = {
            'alpha': trial.suggest_float('alpha', 0.1, 10.0, log=True),
            'l1_ratio': trial.suggest_float('l1_ratio', 0.0, 1.0),
            'fit_intercept': False
        }
    elif model_name == 'GROUPLASSO':
        corr_thresh = trial.suggest_float('corr_thresh', 0.2, 0.55)
        linkage_method = trial.suggest_categorical('linkage_method', ['ward', 'average', 'complete'])
        groups = group_features_by_correlation(X, corr_thresh, linkage_method)
        param = {
            'l1_reg': trial.suggest_float('l1_reg', 0.1, 10.0, log=True),
            'group_reg': trial.suggest_float('group_reg', 0.1, 10.0, log=True),
            'fit_intercept': False,
            'groups': groups,
        }
        
    ModelClass = ModelType[model_name].value
    model = ModelClass(**param)
    
    model.fit(X, y)
    save_model_func(model, trial.number)
    
    ## step 1: 获得predict结果
    factor = model_predict_func(model, X, y)
    # 统一mask（其实理论上应该是本来就一样的）
    to_mask = factor.isna() | rtn.isna()
    factor = factor.mask(to_mask)
    rtn = rtn.mask(to_mask)
    ## step 2: 结合rtn计算指标
    fct_n_pct = calc_rank(factor)
    gps, gpd = calc_gp(fct_n_pct, rtn)
    hsr = calc_hsr(fct_n_pct)
    metrics = get_general_return_metrics(gpd)
    sharpe_ratio = metrics['sharpe_ratio']
    avg_hsr = np.nanmean(hsr)
    
    return sharpe_ratio, avg_hsr


directions_v2 = ['maximize', 'minimize']


# %% v3
def objective_v3(trial, X, y, *, rtn, model_predict_func, save_model_func):
    model_name = trial.suggest_categorical('model', ['LGBM', 'RIDGE'])
    
    if model_name == 'LGBM':
        max_depth = trial.suggest_int('max_depth', 2, 8)
        param = {
            'boosting_type': trial.suggest_categorical('boosting_type', ['gbdt', 'dart', 'goss']),
            'n_estimators': trial.suggest_int('n_estimators', 50, 700, step=10),
            'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.5, step=0.001), 
            'max_depth': max_depth,
            'num_leaves': trial.suggest_int('num_leaves', max(int(0.3 * 2**(max_depth-1)), 2*max_depth-2), max(int(0.75 * 2**max_depth)+1, 4*max_depth)),
            'min_child_samples': trial.suggest_int('min_child_samples', 1000, 6000, step=100),
            'max_bin': trial.suggest_int('max_bin', 60, 500, step=5),
            'verbosity': -1  # 关闭日志输出
        }
        model = lgb.LGBMRegressor(**param)
    elif model_name == 'RIDGE':
        param = {
            'alpha': trial.suggest_float('alpha', 0.1, 10000.0, log=True), # ???
            'fit_intercept': False
        }
        model = Ridge(**param)
    
    model.fit(X, y)
    save_model_func(model, trial.number)
    
    ## step 1: 获得predict结果
    factor = model_predict_func(model, X, y)
    # 统一mask（其实理论上应该是本来就一样的）
    to_mask = factor.isna() | rtn.isna()
    factor = factor.mask(to_mask)
    rtn = rtn.mask(to_mask)
    ## step 2: 结合rtn计算指标
    ic = calc_ic(factor, rtn)
    ic_avg = np.nanmean(ic)
    
    return ic_avg


directions_v3 = ['maximize']


