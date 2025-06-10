# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 10:37:19 2024

@author: Xintang Zheng

"""
# %% imports
from pathlib import Path
from datetime import datetime
import toml
import warnings
warnings.filterwarnings("ignore")


from utils.dirutils import load_path_config
from utils.timeutils import RollingPeriods
from test_and_eval.factor_evaluation import FactorEvaluation
from utils.logutils import FishStyleLogger

                
# %%
class RollingEval:
    
    def __init__(self, eval_name, eval_rolling_name, pstart='20230701', puntil=None, eval_type='rolling', n_workers=1):
        self.eval_name = eval_name
        self.eval_rolling_name = eval_rolling_name
        self.pstart = pstart
        self.puntil = puntil or datetime.utcnow().date().strftime('%Y%m%d')
        self.eval_type = eval_type
        self.n_workers = n_workers
        
        self._load_path_config()
        self._init_dir()
        self._load_params()
        self._init_log()
        
    def _load_path_config(self):
        file_path = Path(__file__).resolve()
        project_dir = file_path.parents[1]
        self.path_config = load_path_config(project_dir)
        
    def _init_dir(self):
        self.param_dir = Path(self.path_config['param']) / 'eval_rolling'
        
    def _load_params(self):
        self.params = toml.load(self.param_dir / f'{self.eval_rolling_name}.toml')
        
    def _init_log(self):
        self.log = FishStyleLogger()
        
    def run(self):
        rolling_dates = self._get_rolling_dates()
        lb_fit_periods_list = self._get_lb_fit_periods(rolling_dates)
        data_fit_periods = self._get_data_fit_periods(rolling_dates)
        
        fe = FactorEvaluation(self.eval_name, n_workers=self.n_workers)
        for fit_periods in lb_fit_periods_list:
            if self.eval_type == 'rolling':
                for fp, dfp in list(zip(fit_periods, data_fit_periods)):
                    fe.eval_one_period(*fp, *dfp)
            elif self.eval_type == 'update':
                fp = fit_periods[-1]
                dfp = data_fit_periods[-1]
                fe.eval_one_period(*fp, *dfp)
        
    def _get_rolling_dates(self):
        fstart = self.params['fstart']
        
        dates = {
            'fstart': fstart,
            'pstart': self.pstart,
            'puntil': self.puntil,
            }
        
        dates.update({'puntil': self.puntil})
        rolling_dates = {date_name: datetime.strptime(dates[date_name], '%Y%m%d')
                         for date_name in dates}
        
        return rolling_dates
        
    def _get_lb_fit_periods(self, rolling_dates):
        rolling_params = self.params['rolling_params']
        lb_list = self.params['lb_list']
        
        rolling_params.update(rolling_dates)
        lb_rolling_pr_list = [{**rolling_params, **{'window_kwargs': {'months': lb}}}
                              for lb in lb_list]
        lb_rolling_list = [RollingPeriods(**lb_rolling_pr) for lb_rolling_pr in lb_rolling_pr_list]
        lb_fit_periods_list = [rolling.fit_periods for rolling in lb_rolling_list]
        return lb_fit_periods_list
        
    def _get_data_fit_periods(self, rolling_dates):
        data_rolling_params = self.params['data_rolling_params']
        data_rolling_params.update(rolling_dates)
        data_rolling = RollingPeriods(**data_rolling_params)
        data_fit_periods = data_rolling.fit_periods
        return data_fit_periods

        
        
    

