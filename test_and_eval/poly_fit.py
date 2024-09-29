# -*- coding: utf-8 -*-
"""
Created on Fri May 10 16:00:29 2024

@author: Xintang Zheng

"""
# %% imports
import os
import pandas as pd
import numpy as np
from functools import partial
from tqdm import tqdm
from abc import ABC, abstractmethod
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge, Lasso


# from dirs import PROCESSED_DATA_DIR, TWAP_PRICE_DIR
from test_and_eval.feature_engineering import normalization
from utils.datautils import extract_sp, get_one_factor, align_index_with_main
from utils.timeutils import timestr_to_minutes, RollingPeriods


# %%
class PolyFit:

    def __init__(self, poly_name, params={}, rolling_params_fitted={}):
        self.model = globals()[params['model']]
        self.poly_name = poly_name
        self.preprocess_params = params['preprocess']
        self.poly_params = params['poly']
        self.fit_params = params['fit']
        self.rolling_params_variable = params['rolling']
        self.rolling_params_fitted = rolling_params_fitted
        self._init_dir()
        self._init_utils()
        self._init_rolling_period()
        self._load_n_preprocess_twap_return()
        
    def _init_dir(self):
        self.data_dir = PROCESSED_DATA_DIR
        self.twap_name = self.preprocess_params.get('twap_name')
        self.twap_dir = TWAP_PRICE_DIR
        
    def _init_utils(self):
        outlier = self.preprocess_params.get('outlier')
        self.normalization_func = partial(normalization, outlier_n=outlier)
        
    def _init_rolling_period(self):
        self.rolling = RollingPeriods(**self.rolling_params_fitted, 
                                      **{'rrule_kwargs': self.rolling_params_variable['rrule'], 
                                         'window_kwargs': self.rolling_params_variable['window'],})
        
    def _load_n_preprocess_twap_return(self):
        pred_wd = self.preprocess_params.get('pred_wd')
        sp = extract_sp(self.twap_name)
        
        twap_path = self.twap_dir / f'{self.twap_name}.parquet'
        twap_price = pd.read_parquet(twap_path)

        # sp_in_min = timestr_to_minutes(sp)
        pp_by_sp = int(pred_wd / sp)
        rtn_1p = twap_price.pct_change(pp_by_sp, fill_method=None
                                       ).shift(-pp_by_sp).replace([np.inf, -np.inf], np.nan)
        rtn_1p = self.normalization_func(rtn_1p)
        self.rtn_1p = rtn_1p
    
    def rolling_fit(self, process_name, factor_name):
        self.process_name = process_name
        self.factor_name = factor_name
        factor_path = self.data_dir / process_name / f'{factor_name}.parquet'
        if not os.path.exists(factor_path):
            return
        process_name_to_save = f'{process_name}_{self.poly_name}'
        factor_poly_dir = self.data_dir  / process_name_to_save
        factor_poly_dir.mkdir(parents=True, exist_ok=True)
        
        self.get_one_factor_func = partial(get_one_factor, process_name=process_name, factor_name=factor_name,
                                          factor_data_dir=self.data_dir,
                                          ref_order_col=self.rtn_1p.columns)
        fit_periods = self.rolling.fit_periods
        predict_periods = self.rolling.predict_periods
        # for i, (fp, pp) in enumerate(tqdm(list(zip(fit_periods, predict_periods)), desc='rolling_poly_fit')):
        for i, (fp, pp) in enumerate(list(zip(fit_periods, predict_periods))):
            if i == 0:
                pp = (fp[0], pp[1])
            model = self._fit_once(*fp)
            self._predict_once(model, *pp)
        
    def _fit_once(self, start_date, end_date, weight_func=None):
        X, y, y_data = self._load_n_preprocess_x_y(start_date, end_date)
        return self._model_fit(X, y)
    
    def _load_n_preprocess_x_y(self, start_date, end_date):
        poly_n = self.poly_params.get('poly_n')
        
        rtn_1p = self._cut_twap_return_by_period(start_date, end_date)
        factor = self._load_n_preprocess_factors(start_date, end_date, ref_index=rtn_1p.index)
        # rtn_1p = align_index_with_main(factor.index, rtn_1p)
        self.to_mask = rtn_1p.isna() | factor.isna()
        rtn_1p_ravel = rtn_1p.values.reshape(-1, order='F')
        factor_ravel = factor.values.reshape((-1, 1), order='F')
        self.to_mask_ravel = self.to_mask.values.reshape(-1, order='F')
        X = factor_ravel[~self.to_mask_ravel]
        y = rtn_1p_ravel[~self.to_mask_ravel]
        poly = PolynomialFeatures(poly_n, include_bias=False)
        X = poly.fit_transform(X)
        return X, y, rtn_1p
    
    def _cut_twap_return_by_period(self, start_date, end_date):
        rtn_1p = self.rtn_1p
        
        rtn_1p_cut = rtn_1p[(rtn_1p.index >= start_date) & (rtn_1p.index < end_date)]
        # self.to_mask = rtn_1p_cut.isna()
        rtn_1p_cut = rtn_1p_cut.fillna(method='ffill')
        return rtn_1p_cut

    def _load_n_preprocess_factors(self, start_date, end_date, ref_index=None):
        factor = self.get_one_factor_func(date_start=start_date, date_end=end_date, ref_index=ref_index)
        factor = self.normalization_func(factor)
        # self.to_mask = self.to_mask | factor.isna()
        return factor

    def _model_fit(self, X, y):
        model = self.model(**self.fit_params)
        model.fit(X, y)
        return model
    
    def _predict_once(self, model, pred_start_date, pred_end_date):
        X, y, y_data = self._load_n_preprocess_x_y(pred_start_date, pred_end_date)
        self._model_predict(model, X, y, y_data)
    
    def _model_predict(self, model, X, y, y_data):
        y_pred_container = np.zeros_like(y_data.values.reshape(-1, order='F'))
        try:
            y_pred_selected = model.predict(X)
        except ValueError:
            breakpoint()
        y_pred_container[~self.to_mask_ravel] = y_pred_selected
        y_pred_container = y_pred_container.reshape(y_data.shape, order='F')
        y_pred = pd.DataFrame(y_pred_container, index=y_data.index, columns=y_data.columns)
        y_pred = self.normalization_func(y_pred.mask(self.to_mask)) #.fillna(0)
        
        pred_all_path = self.data_dir / f'{self.process_name}_{self.poly_name}' / f'{self.factor_name}.parquet'
        if os.path.exists(pred_all_path):
            pred_all = pd.read_parquet(pred_all_path)
        else:
            pred_all = pd.DataFrame()
        pred_all = pd.concat([pred_all, y_pred])
        pred_all.to_parquet(pred_all_path)
    
