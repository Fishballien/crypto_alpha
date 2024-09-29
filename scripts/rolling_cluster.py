# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 17:09:24 2024

@author: Xintang Zheng

"""
# %% import public
import sys
from pathlib import Path
import signal
from datetime import datetime
import toml
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import warnings
warnings.filterwarnings("ignore")


# %% add sys path
file_path = Path(__file__).resolve()
file_dir = file_path.parents[0]
project_dir = file_path.parents[1]
sys.path.append(str(project_dir))


# %% import self-defined
from utils.timeutils import RollingPeriods
from synthesis.factor_cluster import Cluster


# %%
# cluster_name = 'v4_ma15_only_1'
# cluster_name = 'v5_ma15_only_sharpe_3'
# cluster_name = 'v4_fix_rtn_fill_0'
# cluster_name = 'v6'
# cluster_name = 'v6_poly_only'
# cluster_name = 'v6_poly_neu_1'
# cluster_name = 'v9_filter_2'
# cluster_name = 'v10'
# cluster_name = 'gp_v0_0'
cluster_name = 'agg_240712_1'

cl_rolling_params = {
    'fstart': datetime(2021, 1, 1),
    'pstart': datetime(2022, 1, 1),
    'puntil': datetime(2024, 4, 30),
    'rrule_kwargs': {"freq": "M", "interval": 1, "bymonthday": [1, 16]},
    'window_kwargs': {'months': 24},
    'end_by': 'time', 
    }

rec_rolling_params = {
    'fstart': datetime(2021, 1, 1),
    'pstart': datetime(2022, 1, 1),
    'puntil': datetime(2024, 4, 30),
    'rrule_kwargs': {"freq": "M", "interval": 1, "bymonthday": [1, 16]},
    'window_kwargs': {'months': 3},
    'end_by': 'time', 
    }

super_rec_rolling_params = {
    'fstart': datetime(2021, 1, 1),
    'pstart': datetime(2022, 1, 1),
    'puntil': datetime(2024, 4, 30),
    'rrule_kwargs': {"freq": "M", "interval": 1, "bymonthday": [1, 16]},
    'window_kwargs': {'months': 1},
    'end_by': 'time', 
    }


# %% rolling
def rolling_eval(cluster_name, cl_rolling_params, rec_rolling_params, super_rec_rolling_params,
                 t_executor=None, p_executor=None):
    ''' preparation '''
    cl_rolling = RollingPeriods(**cl_rolling_params)
    rec_rolling = RollingPeriods(**rec_rolling_params)
    super_rec_rolling = RollingPeriods(**super_rec_rolling_params)
    fit_periods = cl_rolling.fit_periods
    rec_periods = rec_rolling.fit_periods
    super_rec_periods = super_rec_rolling.fit_periods
    cl = Cluster(cluster_name, t_executor, p_executor)
    for fp, rp, srp in tqdm(list(zip(fit_periods, rec_periods, super_rec_periods)), desc='clustering'):
    # for fp in fit_periods:
        cl.cluster_one_period(*fp, *rp, *srp)
        
        
# %% signal
def signal_handler(sig, frame):
    if t_executor:
        t_executor.shutdown(wait=False)
    if p_executor:
        p_executor.shutdown(wait=False)
    sys.exit(0)
        
        
# %% main
if __name__=='__main__':
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGILL, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)
    
    t_executor = ThreadPoolExecutor(max_workers=50)
    p_executor = ProcessPoolExecutor(max_workers=50)
    rolling_eval(cluster_name, cl_rolling_params, rec_rolling_params, super_rec_rolling_params,
                 t_executor=t_executor,
                 p_executor=p_executor,
                 )
    t_executor.shutdown()
    p_executor.shutdown()
        