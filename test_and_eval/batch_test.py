# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 15:36:19 2024

@author: Xintang Zheng

星星: ★ ☆ ✪ ✩ 🌟 ⭐ ✨ 🌠 💫 ⭐️
勾勾叉叉: ✓ ✔ ✕ ✖ ✅ ❎
报警啦: ⚠ ⓘ ℹ ☣
箭头: ➔ ➜ ➙ ➤ ➥ ↩ ↪
emoji: 🔔 ⏳ ⏰ 🔒 🔓 🛑 🚫 ❗ ❓ ❌ ⭕ 🚀 🔥 💧 💡 🎵 🎶 🧭 📅 🤔 🧮 🔢 📊 📈 📉 🧠 📝

"""
# %% imports
import os
from pathlib import Path
import toml
from datetime import datetime, timedelta
import json
import traceback


from utils.dirutils import load_path_config, DirectoryProcessor
from utils.logutils import FishStyleLogger
from test_and_eval.factor_tester_adaptive import FactorTest


# %%
class BatchTest:
    
    dt_format = '%Y-%m-%d'
    
    def __init__(self, batch_test_name):
        self.batch_test_name = batch_test_name
        
        self._load_path_config()
        self._init_dir()
        self._load_params()
        self._init_log()
        
    def _load_path_config(self):
        file_path = Path(__file__).resolve()
        project_dir = file_path.parents[1]
        self.path_config = load_path_config(project_dir)
        
    def _init_dir(self):
        self.param_dir = Path(self.path_config['param']) / 'batch_test'
        self.flag_dir = Path(self.path_config['flag']) / self.batch_test_name
        self.flag_dir.mkdir(parents=True, exist_ok=True)
        
    def _load_params(self):
        self.params = toml.load(self.param_dir / f'{self.batch_test_name}.toml')
        
    def _init_log(self):
        self.log = FishStyleLogger()
        
    def run(self, date_today=None):
        date_today = datetime.utcnow().date() if date_today is None else datetime.strptime(date_today, '%Y%m%d').date()
        self._get_start_date(date_today)
        self._get_end_date(date_today)
        self.log.info(f'Test {self.start_date} ~ {self.end_date}')
        self.msgs = []
        self._load_flags(date_today)
        if 'test_targets' in self.params:
            self._test_by_targets()
        elif 'root_dir_dict' in self.params:
            self._test_by_directory()
        else:
            self.log.error('Need at least one of {test_targets, root_dir_dict} to test!')
        self._save_flags(date_today)
        self._check_update_status(date_today)
        
    def _get_start_date(self, date_today):
        start_date = self.params.get('start_date')
        lookback = self.params.get('lookback')
        if start_date is None and lookback is not None:
            lookback = timedelta(**lookback)
            start_date = (date_today - lookback).strftime(self.dt_format)
        self.start_date = start_date
        
    def _get_end_date(self, date_today):
        end_date = self.params.get('end_date')
        delay = self.params.get('delay')
        if end_date is None and delay is not None:
            delay = timedelta(**self.params['delay'])
            end_date = (date_today - delay).strftime(self.dt_format)
        self.end_date = end_date
        
    def _load_flags(self, date_today):
        if 'test_targets' in self.params:
            keys = [test_info['process_name'] for test_info in self.params['test_targets']]
        elif 'root_dir_dict' in self.params:
            root_dir_dict = self.params['root_dir_dict']
            dirp = DirectoryProcessor(root_dir_dict)
            mapping = dirp.mapping
            keys = ['f{root_dir}_{process_name}' for root_dir, root_info in mapping.items() for process_name in root_info['leaf_dirs']]

        date_in_fmt = date_today.strftime(self.dt_format)
        flag_file_name = f'{date_in_fmt}.json'
        self.flag_path = self.flag_dir / flag_file_name
        if os.path.exists(self.flag_path):
            with open(self.flag_path, 'r') as f:
                self.flags = json.load(f)
                if len(self.flags) == 0:
                    self.flags = {k: False for k in keys}
        else:
            self.flags = {k: False for k in keys}
        
    def _test_by_targets(self):
        test_targets = self.params['test_targets']
        skip_plot = self.params['skip_plot']
        n_workers = self.params['n_workers']
        
        for test_info in test_targets:
            if self.flags[test_info['process_name']]:
                continue
            test_names = test_info['test_name']
            test_names = [test_names] if isinstance(test_names, str) else test_names
            status_list = []
            for test_name in test_names:
                param = {pr: test_info[pr] if pr != 'test_name' else test_name for pr in test_info}
                param.update({'skip_plot': skip_plot, 'n_workers': n_workers})
                status = self._test_one_process(param)
                status_list.append(status)
            if sum(status_list) == 0:
                self.flags[test_info['process_name']] = True
                
    def _test_by_directory(self):
        root_dir_dict = self.params['root_dir_dict']
        skip_plot = self.params['skip_plot']
        n_workers = self.params['n_workers']
        test_name = self.params['test_name']
        
        dirp = DirectoryProcessor(root_dir_dict)
        mapping = dirp.mapping

        for root_dir in mapping:
            root_info = mapping[root_dir]
            tag_name = root_info['tag_name']
            process_name_list = root_info['leaf_dirs']
            kwargs = {
                'tag_name': tag_name,
                'factor_data_dir': Path(root_dir), 
                'test_name': test_name, 
                'skip_plot': skip_plot,
                'n_workers': n_workers,
            }
    
            for process_name in process_name_list:
                kwargs_p = kwargs.copy()
                kwargs_p.update({'process_name': process_name})
                status = self._test_one_process(kwargs_p)
                if status == 0:
                    self.flags['f{root_dir}_{process_name}'] = True
            self.log.success(f'Root Finished: {root_dir}')
                
    def _test_one_process(self, kwargs):
        self.log.info(f"Start testing {kwargs['process_name']} by {kwargs['test_name']}")
        try:
            tester = FactorTest(**kwargs, date_start=self.start_date, date_end=self.end_date)
            tester.test_multi_factors()
            self.log.success(f"Finished testing {kwargs['process_name']} by {kwargs['test_name']}")
            return 0
        except:
            e_format = traceback.format_exc()
            self.log.exception(f"Error testing {kwargs['process_name']} by {kwargs['test_name']}")
            self.msgs.append({
                    'type': 'msg',
                    'content': {
                        'level': 'error',
                        'title': f"test {kwargs['process_name']} by {kwargs['test_name']} error",
                        'msg': e_format,
                        }
                    })
        return 1
        
    def _save_flags(self, date_today):
        with open(self.flag_path, 'w') as f:
            json.dump(self.flags, f)
            
    def _check_update_status(self, date_today):
        if all(self.flags.values()):
            self.msgs.append({
                'type': 'update',
                'content': {
                    'obj': f'test_factors_{self.batch_test_name}',
                    'data_ts': date_today,
                    }
                })
            
