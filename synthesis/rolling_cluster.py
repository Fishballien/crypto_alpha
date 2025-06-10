# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 10:37:19 2024

@author: Xintang Zheng

"""
# %% imports
from pathlib import Path
from datetime import datetime
import toml
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")


from utils.dirutils import load_path_config
from utils.timeutils import RollingPeriods
from synthesis.factor_cluster import Cluster
from utils.logutils import FishStyleLogger

                
# %%
class RollingCluster:
    
    def __init__(self, cluster_name, pstart='20230701', 
                 puntil=None, cluster_type='rolling', t_workers=1, p_workers=1):
        self.cluster_name = cluster_name
        self.pstart = pstart
        self.puntil = puntil or datetime.utcnow().date().strftime('%Y%m%d')
        self.cluster_type = cluster_type
        self.t_workers = t_workers
        self.p_workers = p_workers
        
        self._load_path_config()
        self._init_dir()
        self._load_params()
        self._init_log()
        
    def _load_path_config(self):
        file_path = Path(__file__).resolve()
        project_dir = file_path.parents[1]
        self.path_config = load_path_config(project_dir)
        
    def _init_dir(self):
        self.param_dir = Path(self.path_config['param']) / 'cluster'
        
    def _load_params(self):
        self.params = toml.load(self.param_dir / f'{self.cluster_name}.toml')
        
    def _init_log(self):
        self.log = FishStyleLogger()
        
    def run(self):
        rolling_dates = self._get_rolling_dates()
        long_periods, rec_periods, super_rec_periods = self._get_all_fit_periods(rolling_dates)
        
        cl = Cluster(self.cluster_name, self.t_workers, self.p_workers)

        if self.cluster_type == 'rolling':
            # 只对不是 None 的 periods 进行 zip
            periods_zip = zip(
                long_periods,
                rec_periods if rec_periods is not None else [None] * len(long_periods),
                super_rec_periods if super_rec_periods is not None else [None] * len(long_periods),
            )
            for fp, rp, srp in tqdm(list(periods_zip), desc='clustering'):
                if rp is None and srp is not None:
                    raise ValueError("If rec_periods is None, super_rec_periods must also be None.")
                cl.cluster_one_period(*fp, *rp if rp else (None, None), *srp if srp else (None, None))
        
        elif self.cluster_type == 'update':
            # update 时，对 None 的输入两个 None
            rp = rec_periods[-1] if rec_periods is not None else (None, None)
            srp = super_rec_periods[-1] if super_rec_periods is not None else (None, None)
            if rp == (None, None) and srp != (None, None):
                raise ValueError("If rec_periods is None, super_rec_periods must also be None.")
            cl.cluster_one_period(*long_periods[-1], *rp, *srp)
        
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
    
    def _get_all_fit_periods(self, rolling_dates):
        long_rolling_params = self.params['long_rolling_params']
        rec_rolling_params = self.params.get('rec_rolling_params')
        super_rec_rolling_params = self.params.get('super_rec_rolling_params')
        
        long_periods = self._get_fit_periods(rolling_dates, long_rolling_params)
        rec_periods = self._get_fit_periods(rolling_dates, rec_rolling_params)
        super_rec_periods = self._get_fit_periods(rolling_dates, super_rec_rolling_params)
        
        return long_periods, rec_periods, super_rec_periods
        
    def _get_fit_periods(self, rolling_dates, rolling_params):
        if rolling_params is None:
            return None
        rolling_params.update(rolling_dates)
        rolling = RollingPeriods(**rolling_params)
        fit_periods = rolling.fit_periods
        return fit_periods

        
