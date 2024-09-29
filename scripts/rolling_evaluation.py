# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 10:37:19 2024

@author: Xintang Zheng

"""
# %% imports
import argparse
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")


from timeutils import RollingPeriods
from factor_evaluation import FactorEvaluation


# %% add sys path
file_path = Path(__file__).resolve()
file_dir = file_path.parents[0]


# %% default setting
lb_list = [3, 24]


rolling_params = {
    'rrule_kwargs': {"freq": "M", "interval": 1, "bymonthday": 1},
    'end_by': 'time', 
    }


data_rolling_params = {
    'rrule_kwargs': {"freq": "M", "interval": 1, "bymonthday": 1},
    'window_kwargs': {'months': 24},
    'end_by': 'time', 
    }

                
# %% main
def main(rolling_params, data_rolling_params, lb_list):
    '''read args'''
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--eval_name', type=str, help='eval_name')
    parser.add_argument('-fst', '--fstart', type=str, default='20210101', help='fstart')
    parser.add_argument('-pst', '--pstart', type=str, default='20220101', help='pstart')
    parser.add_argument('-pu', '--puntil', type=str, help='puntil')
    parser.add_argument('-wkr', '--n_workers', type=int, default=1, help='n_workers')
    args = parser.parse_args()
    args_dict = vars(args)
    
    eval_name = args.eval_name
    n_workers = args.n_workers
    
    # trans date
    rolling_dates = {date_name: datetime.strptime(args_dict[date_name], '%Y%m%d')
                      for date_name in ['fstart', 'pstart', 'puntil']}
    
    # fill rolling params
    rolling_params = rolling_params.update(rolling_dates)
    data_rolling_params = data_rolling_params.update(rolling_dates)
    lb_rolling_pr_list = [{**rolling_params, **{'window_kwargs': {'months': lb}}}
                          for lb in lb_list]
    data_rolling = RollingPeriods(**data_rolling_params)
    data_fit_periods = data_rolling.fit_periods
    
    fe = FactorEvaluation(eval_name, n_workers=n_workers)
    for lb_rolling_pr in lb_rolling_pr_list:
        rolling = RollingPeriods(**lb_rolling_pr)
        fit_periods = rolling.fit_periods
        for fp, dfp in list(zip(fit_periods, data_fit_periods)):
            fe.eval_one_period(*fp, *dfp)
        
        
# %% main
if __name__=='__main__':
    main()
        
        
        
    

