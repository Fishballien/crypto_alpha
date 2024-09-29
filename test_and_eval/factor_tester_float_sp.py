# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 10:37:19 2024

@author: Xintang Zheng

"""
# %% imports
import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pickle
import copy
import traceback
import xicorpy
import toml
from datetime import datetime
from scipy.stats import skew
from functools import partial


import warnings
warnings.filterwarnings("ignore")


from timeutils import timestr_to_minutes, get_wd_name
from datautils import align_columns, align_index, replace_sp, yeojohnson_transform
from dirs import PROCESSED_DATA_DIR, TWAP_PRICE_DIR, RESULT_DIR, PARAM_DIR
from feature_engineering import normalization, neutralization, normalization_with_mask, neutralization_multi
from algo import parallel_xi_correlation, xi_correlation, calc_corr_daily
from neu_gradients import neu_0, neu_1, neu_2, neu_3
from poly_fit import PolyFit


# %%
factor_data_dir = PROCESSED_DATA_DIR
twap_data_dir = TWAP_PRICE_DIR
result_dir = RESULT_DIR
param_dir = PARAM_DIR
process_name = 'ma15_sp15'
# process_name = 'gp/v0/210101_220101'
twap_name = 'twd15_sp15'

# factor_name = 'bidask_volume_ratio'
factor_name = 'spread_strength'
# factor_name = 'ask_sigma_pdelta5q'
# factor_name = 'askv_mean'
# factor_name = 'factor_gp_88'


# %% params
sp = '60T'
outlier_n = 30
pp_list = [15, 30, 60, 120, 240, 360, 720]
lag_list = list(range(4))
bin_step = 0.1
# neu = neu_3
# neu_lookback = {'days': 30}
# poly_name = 'poly_1y_4'
# poly_params = toml.load(param_dir / 'poly' / f'{poly_name}.toml')
# rolling_params_fitted = {
#     'fstart': datetime(2021, 1, 1),
#     'pstart': datetime(2021, 2, 1),
#     'puntil': datetime(2024, 3, 29),
#     'end_by': 'time'
#     }
yj_transform = True

params = {
    'sp': sp,
    'outlier_n': outlier_n,
    'pp_list': pp_list,
    'lag_list': lag_list,
    'bin_step': bin_step,
    # 'neu': neu,
    # 'neu_lookback': neu_lookback,
    # 'poly_name': poly_name,
    # 'poly_params': poly_params,
    # 'rolling_fitted': rolling_params_fitted,
    # 'yj_transform': yj_transform,
}


# %%
# =============================================================================
# with open('factors.pickle', 'rb') as f:
#     factor_name_list = pickle.load(f)
#     
# factor_name = factor_name_list[111]
# =============================================================================


# %%
class FactorTest:
    
    def __init__(self, process_name, twap_name, factor_data_dir, twap_data_dir, result_dir, params):
        self.process_name = process_name
        self.twap_name = twap_name
        self.factor_data_dir = factor_data_dir
        self.twap_data_dir = twap_data_dir
        self.result_dir = result_dir
        self.params = params
        
        self._load_dirs()
        self._load_twap_price()
        self._preprocess_twap_return()
        self._init_poly_if_needed()
        
    def _load_dirs(self):
        neu = self.params.get('neu', None)
        neu_lookback = self.params.get('neu_lookback', None)
        poly_name = self.params.get('poly_name', None)
        yj_transform = self.params.get('yj_transform', False)
        
        self.factor_dir = (self.factor_data_dir / self.process_name if self.process_name is not None
                           else self.factor_data_dir)
        yj_suffix = '' if not yj_transform else '_yj'
        if neu is None and poly_name is None:
            process_name_to_save = f'{self.process_name}{yj_suffix}'
            save_dir = (self.result_dir / process_name_to_save if self.process_name is not None
                               else self.result_dir)
        elif neu is not None:
            neu_method_name = '' if neu_lookback is None else f'_{get_wd_name(neu_lookback)}'
            process_name_to_save = f'{self.process_name}_{neu.name}{neu_method_name}{yj_suffix}'
            save_dir = self.result_dir / process_name_to_save
            self.factor_neu_dir = self.factor_data_dir  / process_name_to_save
            self.factor_neu_dir.mkdir(parents=True, exist_ok=True)
        elif poly_name is not None:
            process_name_to_save = f'{self.process_name}_{poly_name}{yj_suffix}'
            save_dir = self.result_dir / process_name_to_save
            self.factor_poly_dir = self.factor_data_dir  / process_name_to_save
            self.factor_poly_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir = save_dir / 'data'
        self.plot_dir = save_dir / 'plot'
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.plot_dir.mkdir(parents=True, exist_ok=True)
        
    def _load_twap_price(self):
        sp = self.params['sp']
        
        twap_path = self.twap_data_dir / f'{self.twap_name}.parquet'
        self.twap_price = pd.read_parquet(twap_path)
        if sp != '15T':
            self.twap_price.index = self.twap_price.index + pd.Timedelta(minutes=1)
            self.twap_price = self.twap_price.resample(sp).first()
            self.twap_price = self.twap_price[self.twap_price.index > datetime(2023, 1, 1)]
            
    def _preprocess_twap_return(self):
        pp_list = self.params['pp_list']
        sp = self.params['sp']
        twap_price = self.twap_price
        
        twap_to_mask = twap_price.isna()
        
        rtn_1p = twap_price.pct_change(1, fill_method=None).shift(-1)
        
        # lag
        rtn_of_lag = {}
        for lag in lag_list:
            rtn_lag = rtn_1p.shift(-lag).replace([np.inf, -np.inf], np.nan)
            twap_to_mask = twap_to_mask | rtn_lag.isna()
            rtn_of_lag[lag] = rtn_lag
        
        # pp
        sp_in_min = timestr_to_minutes(sp)
        pp_list = [pp for pp in pp_list if pp % sp_in_min == 0]
        rtn_of_pp = {}
        for pp in pp_list:
            pp_by_sp = int(pp / sp_in_min)
            rtn_pp = twap_price.pct_change(pp_by_sp, fill_method=None).shift(-pp_by_sp).replace([np.inf, -np.inf], np.nan)
            twap_to_mask = twap_to_mask | rtn_pp.isna()
            rtn_of_pp[pp] = rtn_pp
            
        self.twap_to_mask = twap_to_mask
        self.sp_in_min = sp_in_min
        self.pp_list = pp_list
        self.rtn_1p = rtn_1p
        self.rtn_of_lag = rtn_of_lag
        self.rtn_of_pp = rtn_of_pp

    def _reload_twap_return(self, twap_price):
        self.twap_price = twap_price
        self._preprocess_twap_return()
        
    def _init_poly_if_needed(self):
        poly_name = self.params.get('poly_name', None)
        if poly_name is None:
            return
        poly_params = self.params.get('poly_params', {})
        rolling_params_fitted = self.params.get('rolling_fitted', {})
        self.poly_fit = PolyFit(poly_name, poly_params, rolling_params_fitted)
    
    def _load_factor(self, factor_name):
        sp = self.params['sp']
        neu = self.params.get('neu', None)
        neu_lookback = self.params.get('neu_lookback', None)
        poly_name = self.params.get('poly_name', None)
        yj_transform = self.params.get('yj_transform', False)
        
        org_factor_path = self.factor_dir / f'{factor_name}.parquet'
        if not os.path.exists(org_factor_path):
            return None
        
        if neu is not None:
            target_neu_file_path = self.factor_neu_dir / f'{factor_name}.parquet'
            # if os.path.exists(target_neu_file_path):
            #     factor = pd.read_parquet(target_neu_file_path)
            # else:
            try:
                factor = pd.read_parquet(org_factor_path)
            except:
                return None
            to_mask = factor.isna()
            try:
                factor_normed_masked = normalization_with_mask(factor, to_mask, outlier_n)
            except:
                return None
            to_mask = to_mask | factor_normed_masked.isna()
            
            neu_data_list = []
            for pn, fn in neu.gradients:
                if pn == 'curr_price':
                    curr_px_path = twap_data_dir / f'curr_price_sp{self.sp_in_min}.parquet'
                    curr_price = pd.read_parquet(curr_px_path)
                    fn_data = (curr_price / curr_price.shift(fn) - 1).replace([np.inf, -np.inf], np.nan)
                elif fn == 'bias':
                    ma_path = self.factor_data_dir / pn / 'midprice.parquet'
                    ma_price = pd.read_parquet(ma_path)
                    curr_px_path = twap_data_dir / f'curr_price_sp{self.sp_in_min}.parquet'
                    curr_price = pd.read_parquet(curr_px_path)
                    # align columns
                    ma_price = align_columns(factor.columns, ma_price)
                    curr_price = align_columns(factor.columns, curr_price)
                    # align index
                    ma_price, curr_price = align_index(ma_price, curr_price)
                    # calc bias
                    fn_data = (curr_price - ma_price) / ma_price
                else:
                    fn_path = self.factor_data_dir / pn / f'{fn}.parquet'
                    fn_data = pd.read_parquet(fn_path)
                    # if pn == 'tradv':
                    #     fn_data = np.log(fn_data)
                fn_data = align_columns(factor.columns, fn_data)
                to_mask = to_mask | fn_data.isna()
                neu_data_list.append(fn_data)
                
            for fn_data in neu_data_list:
                fn_data = fn_data.mask(to_mask)
                
            neu_data = pd.concat(neu_data_list, axis=0)
            
            if neu_lookback is None:
                neu_func = partial(neutralization, ogn_fct_df=factor_normed_masked, 
                                   zxh_df=neu_data, if_jy_df=~to_mask)
            else:
                neu_func = partial(neutralization_multi, ogn_fct_df=factor_normed_masked, 
                                   neu_data_list=neu_data_list, if_jy_df=~to_mask, lookback_param=neu_lookback)
            factor_neu = [neu_func(tm) for tm in factor_normed_masked.index]
            factor_neu = pd.DataFrame(factor_neu, index=factor_normed_masked.index, columns=factor_normed_masked.columns)
            factor_neu = factor_neu.replace([np.inf, -np.inf], np.nan).fillna(method='ffill')
            
            # save neu
            factor_neu = factor_neu.mask(to_mask)
            factor_neu.to_parquet(self.factor_neu_dir / f'{factor_name}.parquet')
        
            factor = factor_neu
        elif poly_name is not None:
            target_poly_file_path = self.factor_poly_dir / f'{factor_name}.parquet'
            if not os.path.exists(target_poly_file_path):
                self.poly_fit.rolling_fit(self.process_name, factor_name)
            factor = pd.read_parquet(target_poly_file_path)
        else:
            try:
                factor = pd.read_parquet(org_factor_path)
                # factor = factor.shift(-1)
            except:
                return None
        if yj_transform:
            factor = factor.apply(yeojohnson_transform, axis=1)
        # resample
        if sp != '15T':
            # breakpoint()
            factor.index = factor.index + pd.Timedelta(minutes=1)
            factor = factor.resample(sp).first()
            factor = factor[factor.index > datetime(2023, 1, 1)]
        return factor
    
    def test_one_factor(self, factor_name):
        outlier_n = self.params['outlier_n']
        lag_list = self.params['lag_list']
        bin_step = self.params['bin_step']
        neu = self.params.get('neu', None)
        
        twap_price = self.twap_price
        twap_to_mask = self.twap_to_mask
        sp_in_min = self.sp_in_min
        pp_list = self.pp_list
        rtn_1p = copy.deepcopy(self.rtn_1p)
        rtn_of_lag = copy.deepcopy(self.rtn_of_lag)
        rtn_of_pp = copy.deepcopy(self.rtn_of_pp)

        factor = self._load_factor(factor_name)
        if factor is None:
            print(self.factor_dir / f'{factor_name}.parquet')
            return 0
        # breakpoint()
        
        # align columns
        factor = align_columns(twap_price.columns, factor)
        
        # align index
        org_len_twap = len(twap_price)
        factor, twap_price = align_index(factor, twap_price)
        if len(twap_price) != org_len_twap:
            self._reload_twap_return(twap_price)
            twap_to_mask = self.twap_to_mask
            sp_in_min = self.sp_in_min
            pp_list = self.pp_list
            rtn_1p = copy.deepcopy(self.rtn_1p)
            rtn_of_lag = copy.deepcopy(self.rtn_of_lag)
            rtn_of_pp = copy.deepcopy(self.rtn_of_pp)
        # breakpoint()
        
        # rough mask
        to_mask = factor.isna() | twap_to_mask
        
        # norm
        try:
            if self.process_name.startswith('gp'):
                factor_processed = factor
            else:
                factor_processed = normalization_with_mask(factor, to_mask, outlier_n)
        except:
            breakpoint()
            return 0

        # apply mask
        for lag in lag_list:
            rtn_of_lag[lag] = rtn_of_lag[lag].mask(to_mask) #.fillna(0)
            
        for pp in pp_list:
            rtn_of_pp[pp] = rtn_of_pp[pp].mask(to_mask)
            
        # rank
        factor_rank = factor_processed.rank(axis=1, pct=True
                                            ).sub(0.5 / factor_processed.count(axis=1), axis=0
                                                  ).replace([np.inf, -np.inf], np.nan
                                                            ) #.fillna(0)
        fct_n_pct = 2 * (factor_rank - 0.5)
        # tb_only = ((factor_rank <= 0.1) & (factor_rank > 0)) | (factor_rank >= 0.9)
        # breakpoint()
        
        # long short
        df_gp = pd.DataFrame()
        for lag in lag_list:
            rtn_lag = rtn_of_lag[lag]
            gps = (fct_n_pct * rtn_lag).mean(axis=1).shift(lag).fillna(0)
            gpsd = gps.resample('D').sum(min_count=1).dropna()
            df_gp[f'long_short_{lag}'] = gpsd
        df_gp.to_parquet(self.data_dir / f'gp_{factor_name}.parquet')
        
            
        # ic
        df_ic = pd.DataFrame()
        df_icd = pd.DataFrame()
        for pp in pp_list:
            rtn_pp = rtn_of_pp[pp]
            ics = factor_processed.corrwith(rtn_pp, axis=1, method='pearson').replace([np.inf, -np.inf], np.nan).fillna(0)
            icsm = ics.resample('M').mean()
            df_ic[f'ic_{pp}min'] = icsm
            icsd = ics.resample('D').mean()
            df_icd[f'ic_{pp}min'] = icsd
            
# =============================================================================
#             f_tb = factor_processed.mask(~tb_only)
#             r_tb = rtn_pp.mask(~tb_only)
#             icstb = f_tb.corrwith(r_tb, axis=1, method='spearman').replace([np.inf, -np.inf], np.nan).fillna(0)
#             icstbm = icstb.resample('M').mean()
#             df_ic[f'ic_tb_{pp}min'] = icstbm
#             
#             ics = factor_processed.corrwith(rtn_pp, axis=1, method='pearson').replace([np.inf, -np.inf], np.nan).fillna(0)
#             icsm = ics.resample('M').mean()
#             df_ic[f'ic_ps_{pp}min'] = icsm
#             
#             ics = factor_processed.corrwith(rtn_pp, axis=1, method='kendall').replace([np.inf, -np.inf], np.nan).fillna(0)
#             icsm = ics.resample('M').mean()
#             df_ic[f'ic_kd_{pp}min'] = icsm
# =============================================================================
            
# =============================================================================
#             if pp == 240:
#                 temp_dir = Path('/home/xintang/crypto/multi_factor/factor_test_by_alpha/sample_data/spread_strength')
#                 temp_dir.mkdir(parents=True, exist_ok=True)
#                 for i, idx in enumerate(ics.index):
#                     factor_row = factor_processed.loc[idx, :]
#                     rtn_row = rtn_pp.loc[idx, :]
#                     plt.figure(figsize=(10, 10))
#                     plt.title(f'{idx} {ics[idx]}', fontsize=10, pad=15)
#                     plt.grid(linestyle=':')
#                     plt.scatter(factor_row, rtn_row, color='r' if ics[idx] > 0 else 'g')
#                     plt.axvline(0, linestyle='--')
#                     plt.axvline(factor_row.median(), color='k', linestyle='--')
#                     plt.axhline(0, linestyle='--')
#                     plt.savefig(temp_dir / f"{i}.jpg", dpi=100, bbox_inches="tight")
#                     plt.close()
# =============================================================================
        df_ic.to_parquet(self.data_dir / f'ic_{factor_name}.parquet')
        df_icd.to_parquet(self.data_dir / f'icd_{factor_name}.parquet')
        
        # xicor
# =============================================================================
#         df_xicor = pd.DataFrame()
#         df_xicord = pd.DataFrame()
#         for pp in pp_list:
#             rtn_pp = rtn_of_pp[pp]
#             ics = pd.Series(xi_correlation(factor_processed, rtn_pp), 
#                             index=factor_processed.index).replace([np.inf, -np.inf], np.nan).fillna(0)
#             icsm = ics.resample('M').mean()
#             df_xicor[f'xi_{pp}min'] = icsm
#             icsd = ics.resample('D').mean()
#             df_xicord[f'xi_{pp}min'] = icsd
#         df_xicor.to_parquet(self.data_dir / f'xicor_{factor_name}.parquet')
#         df_xicord.to_parquet(self.data_dir / f'xicord_{factor_name}.parquet')
# =============================================================================
        df_xicor = pd.DataFrame()
        df_xicord = pd.DataFrame()
        for pp in pp_list:
            rtn_pp = rtn_of_pp[pp]
            icsd = calc_corr_daily(factor_processed, rtn_pp, method='xi')
            icsm = icsd.resample('M').mean()
            df_xicor[f'xi_{pp}min'] = icsm
            df_xicord[f'xi_{pp}min'] = icsd
        df_xicor.to_parquet(self.data_dir / f'xicor_{factor_name}.parquet')
        df_xicord.to_parquet(self.data_dir / f'xicord_{factor_name}.parquet')
        
        # skewness
        df_skewd = pd.DataFrame()
        factor_skewness = factor.skew(axis=1)
        df_skewd['skewness'] = factor_skewness.resample('D').mean()
        df_skewd.to_parquet(self.data_dir / f'skewd_{factor_name}.parquet')
            
        # turnover
# =============================================================================
#         df_hsr = pd.DataFrame()
#         for direction in [1, -1]:
#             open_cond = factor_rank > 0.9 if direction > 0 else (factor_rank < 0.1) & (factor_rank > 0)
#             ps = pd.DataFrame(np.where(open_cond, 1, 0), index=factor_rank.index, columns=factor_rank.columns).fillna(0)
#             hsr = ((ps - ps.shift(1)).abs().sum(axis=1) / (2 * ps.abs().sum(axis=1))).replace([np.inf, -np.inf], np.nan)
#             hsrm = hsr.resample('M').mean()
#             df_hsr[f'turnover_direction_{direction}'] = hsrm
#         df_hsr.to_parquet(self.data_dir / f'hsr_{factor_name}.parquet')
# =============================================================================
        
        df_hsr = pd.DataFrame()
        ps = fct_n_pct
        hsr = ((ps - ps.shift(1)).abs().sum(axis=1) / (2 * ps.abs().sum(axis=1))).replace([np.inf, -np.inf], np.nan)
        hsrm = hsr.resample('M').mean()
        df_hsr['turnover'] = hsrm
        df_hsr.to_parquet(self.data_dir / f'hsr_{factor_name}.parquet')
        
        # bins
        steps_start = np.arange(0, 1, bin_step)
        steps_end = np.arange(bin_step, 1+bin_step, bin_step)
        steps_of_lag = {}
        
        for lag in lag_list:
            rtn_lag = rtn_of_lag[lag]
            df_step = pd.DataFrame()
            for step_start, step_end in zip(steps_start, steps_end):
                selected_range = (factor_rank > step_start) & (factor_rank <= step_end)
                ps_step = pd.DataFrame(np.where(selected_range, 1, 0), 
                                       index=factor_rank.index, columns=factor_rank.columns).fillna(method='ffill').fillna(0)
                ps_step = ps_step.fillna(0)
                w_step = ps_step.div(ps_step.sum(axis=1, min_count=1), axis=0).replace([np.inf, -np.inf], np.nan).fillna(0)
                rtn_step = (w_step * rtn_lag).sum(axis=1, min_count=1).shift(lag).fillna(0)
                rtn_step_d = rtn_step.resample('D').sum(min_count=1).dropna()
                df_step[f'{step_start:.0%} - {step_end:.0%}'] = rtn_step_d
            steps_of_lag[lag] = df_step
        with open(self.data_dir / f'bins_{factor_name}.pkl', 'wb') as file:
            pickle.dump(steps_of_lag, file)
        
        # plot
        FONTSIZE_L1 = 20
        FONTSIZE_L2 = 18
        FONTSIZE_L3 = 15
        
        title = f'{factor_name}_{sp_in_min}min'
        
        fig = plt.figure(figsize=(56.4, 27.6), dpi=100, layout="constrained")
        spec = fig.add_gridspec(ncols=47, nrows=23)
        
        ax0 = fig.add_subplot(spec[:7, :23])
        ax0.set_title(f'{title}', fontsize=FONTSIZE_L1, pad=25)
        df_gp.cumsum().plot.line(ax=ax0, linewidth=3)
        
        ax1 = fig.add_subplot(spec[8:15, :23])
        df_ic = df_ic.set_index((df_ic.index.year * 100 + df_ic.index.month) % 10000)
        df_ic.plot.bar(ax=ax1)
        
        ax2 = fig.add_subplot(spec[-7:, :23])
        df_hsr = df_hsr.set_index((df_hsr.index.year * 100 + df_hsr.index.month) % 10000)
        df_hsr.plot.bar(ax=ax2)
        ax2.set_ylim([0, 1])
        
        cmap = cm.get_cmap('jet')
        colors = cmap(np.linspace(0.2, 0.9, 10))
        ax3 = fig.add_subplot(spec[:11, 24:35])
        # ax3.set_title(f'lag_{lag}', fontsize=FONTSIZE_L2, pad=10)
        steps_of_lag[0].cumsum().plot.line(ax=ax3, color=colors)
        
        ax4 = fig.add_subplot(spec[:11, -11:])
        # ax4.set_title('bin_diff', fontsize=FONTSIZE_L2, pad=10)
        bins_of_lag_0 = steps_of_lag[0]
        ax4.plot((bins_of_lag_0['90% - 100%'] - bins_of_lag_0['0% - 10%']).cumsum(),
                 color='k', linewidth=3, label='bin_diff')
        ax4.tick_params(axis='x', rotation=45)
        
        ax5 = fig.add_subplot(spec[-11:, 24:35])
        # ax5.set_title('bin_return', fontsize=FONTSIZE_L2, pad=10)
        rtn_of_bins = bins_of_lag_0.sum(axis=0)
        ax5.bar(list(rtn_of_bins.index)[::-1], list(rtn_of_bins)[::-1], label='bin_return')
        ax5.tick_params(axis='x', rotation=45)
        
        ax6 = fig.add_subplot(spec[-11:, -11:])
        # ax6.set_title('xi_cor', fontsize=FONTSIZE_L2, pad=10)
        df_skewd.plot.line(ax=ax6)
        # ax6.set_ylim([0, 1])

# =============================================================================
#         bin_ax_list = []
#         cmap = cm.get_cmap('jet')
#         colors = cmap(np.linspace(0.2, 0.9, 10))
#         for i_p, (lag, spec_area) in enumerate(
#                 zip(lag_list, [spec[:11, 24:35], spec[:11, -11:], spec[-11:, 24:35], spec[-11:, -11:]])
#                 ):
#             ax = fig.add_subplot(spec_area)
#             ax.set_title(f'lag_{lag}', fontsize=FONTSIZE_L2, pad=10)
#             steps_of_lag[lag].cumsum().plot.line(ax=ax, color=colors)
#             bin_ax_list.append(ax)
# =============================================================================
        
        for ax in [ax0, ax1, ax2, ax3, ax4, ax5, ax6]:
            ax.grid(linestyle=":")
            ax.legend(loc="upper left", borderaxespad=0.5, fontsize=FONTSIZE_L3)
            ax.tick_params(labelsize=FONTSIZE_L2, pad=15)
        
        plt.savefig(self.plot_dir / f"{title}.jpg", dpi=100, bbox_inches="tight")
        plt.close()
        # breakpoint()
        return 1
        
        
# %% main
if __name__=='__main__':
    tester = FactorTest(process_name, twap_name, factor_data_dir, twap_data_dir, result_dir, params)
    tester.test_one_factor(factor_name)

