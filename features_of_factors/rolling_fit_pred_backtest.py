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
from model_fit import choose_model
from utils.dirutils import load_path_config
from utils.timeutils import RollingPeriods

    
# %% main
def main():
    '''read args'''
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--test_name', type=str, help='test_name')
    parser.add_argument('-fst', '--fstart', type=str, default='20210101', help='fstart')
    parser.add_argument('-pst', '--pstart', type=str, default='20220101', help='pstart')
    parser.add_argument('-pu', '--puntil', type=str, help='puntil')
    parser.add_argument('-wkr', '--n_workers', type=int, default=1, help='n_workers')
    args = parser.parse_args()
    args_dict = vars(args)
    
    test_name = args.test_name
    n_workers = args.n_workers

    # trans date
    rolling_dates = {date_name: datetime.strptime(args_dict[date_name], '%Y%m%d')
                     for date_name in ['fstart', 'pstart', 'puntil']}
    rolling_dates.update({'end_by': 'time'})
    
    # init dir
    path_config = load_path_config(project_dir)
    param_dir = Path(path_config['param']) / 'features_of_factors' / 'model'

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
    for fltp, fp, pp in list(zip(filter_periods, fit_periods, predict_periods)):
        mf.set_filter_period(*fltp)
        mf.fit_once(*fp)
        # mf.log_model_info(model_start_date=fp[0], model_end_date=fp[1])
        # mf.plot_trees(model_start_date, model_end_date)
        mf.predict_once(*fp, *pp)
        mf.test_predicted()


# %% main
if __name__=='__main__':
    main()
    
