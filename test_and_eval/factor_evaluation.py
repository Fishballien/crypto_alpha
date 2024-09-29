# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 10:37:19 2024

@author: Xintang Zheng

"""
# %% imports
import os
import toml
import pandas as pd
import numpy as np
import pickle
import yaml
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from functools import partial
from tqdm import tqdm
import traceback
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import warnings
warnings.filterwarnings("ignore")


from utils.dirutils import load_path_config
from utils.timeutils import timestr_to_minutes, period_shortcut
from test_and_eval.scores import get_general_return_metrics, calc_sharpe
from utils.speedutils import multiprocess_with_sequenced_result, timeit


# %%
def eval_one_factor_one_period(factor_name, *, date_start, date_end, data_date_start, data_date_end,
                               process_name, test_name, tag_name, data_dir, processed_data_dir, 
                               sp, fee, valid_prop_thresh):
    res_dict = {
        'root_dir': processed_data_dir, 
        'test_name': test_name, 
        'tag_name': tag_name, 
        'process_name': process_name, 
        'factor': factor_name,
        }
    try:
        df_gp = pd.read_parquet(data_dir / f'gp_{factor_name}.parquet')
        df_ic = pd.read_parquet(data_dir / f'icd_{factor_name}.parquet')
        # df_xicor = pd.read_parquet(data_dir / f'xicord_{factor_name}.parquet')
        df_hsr = pd.read_parquet(data_dir / f'hsr_{factor_name}.parquet')
        try:
            df_mmt = pd.read_parquet(data_dir / f'mmt_{factor_name}.parquet')
        except:
            pass
        with open(data_dir / f'bins_{factor_name}.pkl', 'rb') as file:
            bins_of_lag = pickle.load(file)
        try:
            with open(data_dir / f'bins_sw_{factor_name}.pkl', 'rb') as file:
                bins_of_lag_sw = pickle.load(file)
        except:
            bins_of_lag_sw = {}
    except:
        return res_dict
    
    # direction
    if data_date_start is None or data_date_end is None:
        data_date_start, data_date_end = date_start, date_end
    gp_of_data = df_gp[(df_gp.index >= data_date_start) & (df_gp.index <= data_date_end)]
    cumrtn_lag_0 = gp_of_data['long_short_0'].sum()
    direction = 1 if cumrtn_lag_0 > 0 else -1
    res_dict.update({'direction': direction})
    
    df_gp = df_gp[(df_gp.index >= date_start) & (df_gp.index <= date_end)]
    df_ic = df_ic[(df_ic.index >= date_start) & (df_ic.index <= date_end)]
    # df_xicor = df_xicor[(df_xicor.index >= date_start) & (df_xicor.index <= date_end)]
    df_hsr = df_hsr[(df_hsr.index >= date_start) & (df_hsr.index <= date_end)]
    try:
        df_mmt = df_mmt[(df_mmt.index >= date_start) & (df_mmt.index <= date_end)]
    except:
        pass
    
    # long short metrics
    gps = df_gp['long_short_0'].replace([0.0], np.nan)
    if gps.count() < valid_prop_thresh * gps.size:
        return res_dict
    
    cumrtn_lag_0 = df_gp['long_short_0'].sum()
    
    for gp_period in range(4):
        gp_period_name = f'long_short_{int(gp_period)}'
        
        lag_p_metrics = get_general_return_metrics(df_gp[gp_period_name]*direction)
        if gp_period > 0:
            lag_p_metrics = {f'{metric_name}_{int(gp_period)}': v for metric_name, v in lag_p_metrics.items()}
        res_dict.update(lag_p_metrics)
        
    cumrtn_lag_1 = df_gp['long_short_1'].sum()
    diff_with_lag_1 = (cumrtn_lag_0 - cumrtn_lag_1) / cumrtn_lag_0
    res_dict.update({'diff_with_lag_1': diff_with_lag_1})
    
    # ic
    for ic_type in df_ic.columns:
        ic_sharpe = np.mean(df_ic[ic_type]) / np.std(df_ic[ic_type]) * np.sqrt(365)
        ic_sharpe_with_direction = ic_sharpe * direction
        res_dict.update({f'{ic_type}': ic_sharpe, f'{ic_type}_with_direction': ic_sharpe_with_direction})
        
        
    # xicor
# =============================================================================
#     for ic_type in df_xicor.columns:
#         ic_sharpe = np.mean(df_xicor[ic_type]) / np.std(df_xicor[ic_type]) * np.sqrt(365)
#         res_dict.update({f'{ic_type}': ic_sharpe})
# =============================================================================
    
    # hsr
    hsr = df_hsr.mean(axis=1).mean(axis=0)
    res_dict.update({'hsr': hsr})
    
    # bins
    bins_of_lag_0 = bins_of_lag[0]
    bins_of_lag_0 = bins_of_lag_0[(bins_of_lag_0.index >= date_start) & (bins_of_lag_0.index <= date_end)]
    bin_long_short = (bins_of_lag_0['90% - 100%'] - bins_of_lag_0['0% - 10%']) * direction
    bin_long_short_metrics = get_general_return_metrics(bin_long_short)
    res_dict.update({f'bin_{k}': v for k, v in bin_long_short_metrics.items()})
    bins_cumrtn = bins_of_lag_0.sum(axis=0) * direction
    quantile_performance = np.corrcoef(bins_cumrtn, list(range(10)))[0, 1]
    res_dict.update({'quantile_performance': quantile_performance})
    
    # bins sw
    if bins_of_lag_sw:
        try:
            bins_sw_list = [
                ('125', '87.5% - 100.0%', '0.0% - 12.5%'),
                ('150', '85.0% - 100.0%', '0.0% - 15.0%'),
                ('175', '82.5% - 100.0%', '0.0% - 17.5%'),
                ]
            bins_of_lag_0 = bins_of_lag_sw[0]
            bins_of_lag_0 = bins_of_lag_0[(bins_of_lag_0.index >= date_start) & (bins_of_lag_0.index <= date_end)]
            for name, upper, lower in bins_sw_list:
                if upper not in bins_of_lag_0.columns or lower not in bins_of_lag_0.columns:
                    continue
                bin_long_short = (bins_of_lag_0[upper] - bins_of_lag_0[lower]) * direction
                bin_long_short_metrics = get_general_return_metrics(bin_long_short)
                res_dict.update({f'bin_{name}_{k}': v for k, v in bin_long_short_metrics.items()})
        except:
            print(bins_of_lag_0)
            
    # rtn with fee
    sp_in_min = timestr_to_minutes(sp)
    oneday_hs = 24*60 / sp_in_min * hsr
    bin_ls_annrtn_with_fee = res_dict['bin_return_annualized'] - fee * 2 * oneday_hs * 365
    res_dict.update({'bin_ls_annrtn_with_fee': bin_ls_annrtn_with_fee})
    
    # adf
    try:
        adf_result = adfuller(df_mmt['factor_mean'].dropna())
        res_dict.update({'adf_statistic': adf_result[0], 'adf_p_value': adf_result[1]})
    except:
        pass
    return res_dict


def get_one_factor_sharpe_and_gp(factor_name, *, date_start, date_end, date_range, data_dir, valid_prop_thresh,
                                 filter_gp):
    try:
        df_gp = pd.read_parquet(data_dir / f'gp_{factor_name}.parquet')
        # df_icd = pd.read_parquet(data_dir / f'icd_{factor_name}.parquet')
    except:
        traceback.print_exc()
        return 0.0, pd.Series(0, index=date_range)
    
    # time period
    df_gp = df_gp[(df_gp.index >= date_start) & (df_gp.index <= date_end)].reindex(date_range)
    # ic
    ic_series = df_gp[filter_gp].fillna(0)
    # ic_series = df_icd[(df_icd.index >= date_start) & (df_icd.index <= date_end)].reindex(date_range)['ic_240min'].fillna(0)
    # long short metrics
    gps = df_gp[filter_gp].fillna(0)
    if gps.count() < valid_prop_thresh * gps.size:
        return 0.0, pd.Series(0, index=date_range)
    cumrtn_lag_0 = df_gp[filter_gp].sum()
    direction = 1 if cumrtn_lag_0 > 0 else -1
    sharpe = calc_sharpe(df_gp[filter_gp]*direction)
    return sharpe, ic_series


@timeit
def filter_correlated_features(factor_name_list, data_dir, *, date_start, date_end, valid_prop_thresh, corr_thresh,
                               filter_gp):
    date_range = pd.date_range(start=date_start, end=date_end, freq='D')
    calc_func = partial(get_one_factor_sharpe_and_gp, date_start=date_start, date_end=date_end, date_range=date_range,
                        data_dir=data_dir, valid_prop_thresh=valid_prop_thresh, filter_gp=filter_gp)
    results = [calc_func(factor_name) for factor_name in factor_name_list]

    ic_list = [result[1] for result in results]
    sharpe_list = [result[0] for result in results]
    del results
    
    ic_matrix = np.array(ic_list)
    corr_matrix = np.corrcoef(ic_matrix)
    
    # 创建一个布尔数组来标记是否保留因子
    keep = np.ones(len(factor_name_list), dtype=bool)
    
    # 按照 Sharpe 比率从高到低排序因子索引
    sorted_indices = np.argsort(sharpe_list)[::-1]
    
    # 遍历排序后的因子索引，保留高 Sharpe 比率因子，并标记与其高度相关的因子
    for i in sorted_indices:
        if keep[i]:
            for j in range(len(factor_name_list)):
                if i != j and abs(corr_matrix[i, j]) > corr_thresh:
                    keep[j] = False
    
    # 筛选后的因子列表
    filtered_factors = [factor_name_list[i] for i in range(len(factor_name_list)) if keep[i]]
    return filtered_factors


class FactorEvaluation:
    
    FEE = 0.0005
    valid_prop_thresh = 0.5
    
    def __init__(self, eval_name, n_workers=1):
        self.eval_name = eval_name
        self.n_workers = n_workers
        
        self._load_public_paths()
        self._load_test_params()
        self._init_dirs()
        self.filtered_factors = {}
        
    def _load_public_paths(self):
        file_path = Path(__file__).resolve()
        project_dir = file_path.parents[1]
        self.path_config = load_path_config(project_dir)
        
        self.result_dir = Path(self.path_config['result'])
        self.param_dir = Path(self.path_config['param'])
        self.factor_data_dir = self.path_config['factor_data']
        
    def _load_test_params(self):
        self.params = toml.load(self.param_dir / 'feval' / f'{self.eval_name}.toml')
        self.process_name_list = self.params['process_name_list']
        
    def _init_dirs(self):
        self.feval_dir = self.result_dir / 'factor_evaluation'
        self.save_dir = self.feval_dir / self.eval_name
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.test_dir = self.result_dir  / 'test'
  
    def eval_one_period(self, date_start, date_end, data_date_start=None, data_date_end=None):
        test_name = self.params.get('test_name')
        corr_thresh = self.params.get('corr_thresh', None)
        filter_gp = self.params.get('filter_gp', 'long_short_0')
        check_exists = self.params.get('check_exists', True)
        name_to_autosave = self.params.get('name_to_autosave', 'test')
        sp = self.params['sp']
        test_dir = self.test_dir
        FEE = self.FEE
        valid_prop_thresh = self.valid_prop_thresh
        
        res_df_list = []
        period_name = period_shortcut(date_start, date_end)
        filter_func = partial(filter_correlated_features, 
                              date_start=data_date_start, date_end=data_date_end,
                              valid_prop_thresh=valid_prop_thresh, 
                              corr_thresh=corr_thresh, filter_gp=filter_gp)

        for process_info in self.process_name_list:
            if not isinstance(process_info, str):
                processed_data_dir, tag_name, process_name = process_info
                processed_data_dir = processed_data_dir or self.factor_data_dir
            else:
                processed_data_dir = self.factor_data_dir
                tag_name = None
                process_name = process_info
            if process_name.startswith('gp'):
                data_period_name = period_shortcut(data_date_start, data_date_end)
                process_name = f'{process_name}/{data_period_name}'
            
            # 检查是否已经评估过
            process_eval_dir = (self.feval_dir / name_to_autosave / test_name / tag_name / process_name 
                                if tag_name is not None
                                else self.feval_dir / name_to_autosave / test_name / process_name)
            process_eval_dir.mkdir(exist_ok=True, parents=True)
            corr_thresh_suffix = '' if corr_thresh is None else f'_{str(int(corr_thresh * 100)).zfill(3)}'
            process_res_filename = f'factor_eval_{period_name}{corr_thresh_suffix}'
            process_res_path = process_eval_dir / f'{process_res_filename}.csv'
            if check_exists and os.path.exists(process_res_path):
                res_df = pd.read_csv(process_res_path)
                res_df_list.append(res_df)
                continue
            
            # 定位test结果
            process_dir = (test_dir / test_name / tag_name if tag_name is not None
                          else test_dir / test_name)
            data_dir = process_dir / process_name / 'data' 
            
            # 定位factors
            factor_dir = Path(processed_data_dir) / process_name
            factor_name_list = [path.stem for path in factor_dir.glob('*.parquet')]
            
            # 筛选相关性
            if corr_thresh is not None:
                filtered_factor_list = self.filtered_factors.get((data_date_start, data_date_end, process_name))
                if filtered_factor_list is None:
                    filtered_factor_list = filter_func(factor_name_list, data_dir)
                    self.filtered_factors[(data_date_start, data_date_end, process_name)] = filtered_factor_list
                factor_name_list = filtered_factor_list
            
            # evaluate
            eval_func = partial(eval_one_factor_one_period, date_start=date_start, date_end=date_end,
                                data_date_start=data_date_start, data_date_end=data_date_end,
                                process_name=process_name, test_name=test_name, tag_name=tag_name, 
                                data_dir=data_dir, processed_data_dir=processed_data_dir,
                                sp=sp, fee=FEE, valid_prop_thresh=valid_prop_thresh)
            
            res_list = []
            if self.n_workers is None or self.n_workers == 1:
                for factor_name in tqdm(factor_name_list, desc=f'{self.eval_name} - {process_name} - {period_name}'):
                    res_dict = eval_func(factor_name)
                    res_list.append(res_dict)
            else:
                # res_list = multiprocess_with_sequenced_result(eval_func, factor_name_list, self.n_workers,
                #                                               desc=f'{self.eval_name} - {period_name}')
                with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
                    all_tasks = [executor.submit(eval_func, factor_name)
                                 for factor_name in factor_name_list]
                    for task in tqdm(as_completed(all_tasks), total=len(all_tasks), 
                                     desc=f'{self.eval_name} - {process_name} - {period_name}'):
                        res_dict = task.result()
                        if res_dict is not None:
                            res_list.append(res_dict)
            res_df = pd.DataFrame(res_list)
            res_df_list.append(res_df)
            res_df.to_csv(process_res_path, index=None)
                
        res = pd.concat(res_df_list, axis=0, ignore_index=True)
        
        res.to_csv(self.save_dir / f'factor_eval_{period_name}.csv', index=None)
        
        self._plot_sharpe_dist(period_name, res)
        self._plot_adf_and_sharpe(period_name, res)
        if len(self.process_name_list) == 2:
            self._plot_diff(period_name, res)
        
    def _plot_sharpe_dist(self, period_name, res):
        
        FONTSIZE_L1 = 20
        FONTSIZE_L2 = 18
        FONTSIZE_L3 = 15
        
        title = f'{self.eval_name} Sharpe Ratio {period_name}'
        
        fig = plt.figure(figsize=(36, 27), dpi=100, layout="constrained")
        spec = fig.add_gridspec(ncols=1, nrows=1)
        
        ax0 = fig.add_subplot(spec[:, :])
        ax0.set_title(f'{title}', fontsize=FONTSIZE_L1, pad=25)
        for process_name, group_data in res.groupby('process_name'):
            ax0.hist(group_data['sharpe_ratio'], label=process_name, alpha=.5, bins=50)
        
        for ax in [ax0,]:
            ax.grid(linestyle=":")
            ax.legend(loc="upper left", borderaxespad=0.5, fontsize=FONTSIZE_L3)
            ax.tick_params(labelsize=FONTSIZE_L2, pad=15)
        
        plt.savefig(self.save_dir / f"factor_eval_{period_name}.jpg", dpi=100, bbox_inches="tight")
        plt.close()
        
    def _plot_adf_and_sharpe(self, period_name, res):
        # plot adf p & sharpe
        try:
            FONTSIZE_L1 = 20
            FONTSIZE_L2 = 18
            FONTSIZE_L3 = 15
            
            title = f'{self.eval_name} ADF vs Sharpe Ratio {period_name}'
            
            fig = plt.figure(figsize=(36, 27), dpi=100, layout="constrained")
            spec = fig.add_gridspec(ncols=1, nrows=1)
            
            ax0 = fig.add_subplot(spec[:, :])
            ax0.set_title(f'{title}', fontsize=FONTSIZE_L1, pad=25)
            ax0.scatter(res['adf_p_value'], res['sharpe_ratio'])
            ax0.axvline(x=0.05, linestyle='--')
            ax0.set_xlabel('ADF p-value')
            ax0.set_ylabel('Sharpe Ratio')
            
            for ax in [ax0,]:
                ax.grid(linestyle=":")
                ax.legend(loc="upper left", borderaxespad=0.5, fontsize=FONTSIZE_L3)
                ax.tick_params(labelsize=FONTSIZE_L2, pad=15)
            
            plt.savefig(self.save_dir / f"adf_vs_sharpe_{period_name}.jpg", dpi=100, bbox_inches="tight")
            plt.close()
        except:
            pass
        
    def _plot_diff(self, period_name, res):
         p_name_1 = self.process_name_list[0]
         p_name_2 = self.process_name_list[1]
         res_1 = res[res['process_name'] == p_name_1].set_index('factor')
         res_2 = res[res['process_name'] == p_name_2].set_index('factor')
         diff = pd.DataFrame()
         diff[f'sharpe_{p_name_1}'] = res_1['sharpe_ratio']
         diff[f'sharpe_{p_name_2}'] = res_2['sharpe_ratio']
         diff['sharpe_diff'] = res_2['sharpe_ratio'] - res_1['sharpe_ratio']
         diff.to_csv(self.save_dir / f'diff_{period_name}.csv')
                    
        
# %%
if __name__=='__main__':
    pass