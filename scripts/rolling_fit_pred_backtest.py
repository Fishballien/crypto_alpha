# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 10:26:17 2024

@author: Xintang Zheng

"""
# %% import public
import sys
import argparse
from pathlib import Path
from datetime import datetime
import toml
import warnings
warnings.filterwarnings("ignore")


# %% add sys path
file_path = Path(__file__).resolve()
file_dir = file_path.parents[0]
project_dir = file_path.parents[1]
sys.path.append(str(project_dir))


# %% import self-defined
from synthesis.model_fit import choose_model
from utils.dirutils import load_path_config
from utils.timeutils import RollingPeriods

    
# %% main
def main(test_name=None, puntil=None, fstart='20210101', pstart='20220101', mode='rolling', n_workers=1):
    if test_name is None:
        '''read args'''
        parser = argparse.ArgumentParser()
        parser.add_argument('-t', '--test_name', type=str, help='test_name')
        parser.add_argument('-fst', '--fstart', type=str, default='20210101', help='fstart')
        parser.add_argument('-pst', '--pstart', type=str, default='20220101', help='pstart')
        parser.add_argument('-pu', '--puntil', type=str, help='puntil')
        parser.add_argument('-m', '--mode', type=str, default='rolling', help='mode')
        parser.add_argument('-wkr', '--n_workers', type=int, default=1, help='n_workers')
        args = parser.parse_args()
        args_dict = vars(args)
        
        test_name = args.test_name
        mode = args.mode
        n_workers = args.n_workers
    else:
        args_dict = {
            'puntil': puntil,
            'fstart': fstart,
            'pstart': pstart,
            }
    
    # trans date
    rolling_dates = {date_name: datetime.strptime(args_dict[date_name], '%Y%m%d')
                     for date_name in ['fstart', 'pstart', 'puntil']}
    rolling_dates.update({'end_by': 'time'})
    
    # init dir
    path_config = load_path_config(project_dir)
    param_dir = Path(path_config['param']) / 'model'

    # model config
    model_config = toml.load(param_dir / f'{test_name}.toml')
    preprocess_params = model_config['preprocess_params']
    rolling_params_variable = model_config['rolling_params']
    fit_params = model_config['fit_params']
    predict_params = model_config['predict_params']
    model_fit = choose_model(fit_params['model'])
    
    # init model
    mf = model_fit(test_name, preprocess_params=preprocess_params, fit_params=fit_params, predict_params=predict_params,
                   n_workers=n_workers,
                   )
    
    # prepare rolling
    filter_rolling = RollingPeriods(**rolling_dates, 
                                    **{'rrule_kwargs': rolling_params_variable['rrule'], 
                                       'window_kwargs': rolling_params_variable['filter_window'],})
    fit_rolling = RollingPeriods(**rolling_dates, 
                                 **{'rrule_kwargs': rolling_params_variable['rrule'], 
                                    'window_kwargs': rolling_params_variable['fit_window'],})
    
    filter_periods = filter_rolling.fit_periods
    fit_periods = fit_rolling.fit_periods
    predict_periods = fit_rolling.predict_periods
    
    # run fit & predict
    if mode == 'rolling':
        for fltp, fp, pp in list(zip(filter_periods, fit_periods, predict_periods)):
            mf.set_filter_period(*fltp)
            mf.fit_once(*fp)
            # mf.log_model_info(model_start_date=fp[0], model_end_date=fp[1])
            # mf.plot_trees(model_start_date, model_end_date)
            mf.predict_once(*fp, *pp)
            mf.test_predicted()
        mf.compare_model_with_factors()
    elif mode == 'update':
        fltp = filter_periods[-1]
        fp = fit_periods[-1]
        pp = predict_periods[-1]
        mf.set_filter_period(*fltp)
        mf.fit_once(*fp)
        # mf.log_model_info(model_start_date=fp[0], model_end_date=fp[1])
        # mf.plot_trees(model_start_date, model_end_date)
        mf.predict_once(*fp, *pp)
        mf.test_predicted()
        mf.compare_model_with_factors()
    elif mode == 'update_predict':
        pp = predict_periods[-1]
        index = -2 if pp[0] == pp[1] else -1
        pp = predict_periods[index]
        fltp = filter_periods[index]
        fp = fit_periods[index]
        mf.set_filter_period(*fltp)
        mf.predict_once(*fp, *pp)
        mf.test_predicted()
        mf.compare_model_with_factors()


# %% main
if __name__=='__main__':
    main()
    
