# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 14:59:39 2024

@author: Xintang Zheng

星星: ★ ☆ ✪ ✩ 🌟 ⭐ ✨ 🌠 💫 ⭐️
勾勾叉叉: ✓ ✔ ✕ ✖ ✅ ❎
报警啦: ⚠ ⓘ ℹ ☣
箭头: ➔ ➜ ➙ ➤ ➥ ↩ ↪
emoji: 🔔 ⏳ ⏰ 🔒 🔓 🛑 🚫 ❗ ❓ ❌ ⭕ 🚀 🔥 💧 💡 🎵 🎶 🧭 📅 🤔 🧮 🔢 📊 📈 📉 🧠 📝

"""
# %%
'''
process_name + factor_name & root_dir 找到因子路径
                           & tag_name 找到测试路径
'''
# %% imports
import sys
from pathlib import Path
import yaml
import toml
from loguru import logger
from datetime import datetime
from tqdm import tqdm
import argparse
import warnings
warnings.filterwarnings("ignore")


from utils.dirutils import DirectoryProcessor, compare_directories
from utils.timeutils import RollingPeriods, translate_rolling_params
from test_and_eval.factor_tester_fix_sp import FactorTest
from test_and_eval.factor_evaluation import FactorEvaluation
from utils.logutils import FishStyleLogger


# %% class
class AggTestEval:
    
    def __init__(self, agg_eval_name, test_wkr, eval_wkr):
        self.agg_eval_name = agg_eval_name
        self.test_wkr = test_wkr
        self.eval_wkr = eval_wkr

        self.log = FishStyleLogger()
        self._load_paths()
        self._load_agg_params()

    def _load_paths(self):
        file_path = Path(__file__).resolve()
        file_dir = file_path.parents[1]
        path_config_path = file_dir / '.path_config.yaml'
        with path_config_path.open('r') as file:
            path_config = yaml.safe_load(file)

        self.param_dir = Path(path_config['param'])
        
    def _load_agg_params(self):
        self.agg_params = toml.load(self.param_dir / 'agg' / f'{self.agg_eval_name}.toml')

    def _init_director_processor(self):
        root_dir_dict = self.agg_params['root_dir_dict']
        
        self.dirp = DirectoryProcessor(root_dir_dict)

    def multi_test(self):
        mapping = self.dirp.mapping
        test_pr = self.agg_params['test']
        test_name = test_pr['test_name']
        skip_plot = test_pr['skip_plot']
        retest_tolerance = test_pr['retest_tolerance']
        
        self.test_params = toml.load(self.param_dir / 'test' / f'{test_name}.toml')
        
        for root_dir in mapping:
            root_info = mapping[root_dir]
            tag_name = root_info['tag_name']
            process_name_list = root_info['leaf_dirs']
            kwargs = {
                'tag_name': tag_name,
                'factor_data_dir': Path(root_dir), 
                'test_name': test_name, 
                'skip_plot': skip_plot,
                'n_workers': self.test_wkr,
            }
    
            for process_name in process_name_list:
                self.log.info(f'Test Started: {process_name}')
                tester = FactorTest(process_name, **kwargs)
                factor_dir = tester.factor_dir
                data_dir = tester.data_dir
                if_pass, count1, count2, ratio = compare_directories(
                    factor_dir, '*.parquet', data_dir, 'gp*.parquet', retest_tolerance)
                if if_pass:
                    self.log.info(f'Skip Testing: factors {count1}, tested {count2}, ratio {ratio}')
                    continue
                self.log.info(f'Start Testing: factors {count1}, tested {count2}, ratio {ratio}')
                tester.test_multi_factors()
                self.log.success(f'Test Finished: {process_name}')
            self.log.success(f'Root Finished: {root_dir}')

    def evaluate(self):
        list_of_tuple = self.dirp.list_of_tuple
        test_pr = self.agg_params['test']
        test_name = test_pr['test_name']
        eval_pr = self.agg_params['eval']
        lb_list = eval_pr['lb_list']
        data_lb = eval_pr['data_lb']
        eval_param = eval_pr['param']
        rolling_params = translate_rolling_params(eval_pr['rolling_params'])
        
        eva_params = {
            'sp': self.test_params['sp'], 
            'test_name': test_name,
            'process_name_list': list_of_tuple, 
            **eval_param
            }
        with open(self.param_dir / 'feval' / f'{self.agg_eval_name}.toml', 'w') as toml_file:
            toml.dump(eva_params, toml_file)
        
        data_rolling_pr = {**rolling_params, **{'window_kwargs': {'months': data_lb}}}
        lb_rolling_pr_list = [{**rolling_params, **{'window_kwargs': {'months': lb}}}
                              for lb in lb_list]
        data_rolling = RollingPeriods(**data_rolling_pr)
        data_fit_periods = data_rolling.fit_periods
        
        fe = FactorEvaluation(self.agg_eval_name, n_workers=self.eval_wkr)
        for lb_rolling_pr in lb_rolling_pr_list:
            self.log.info(f"Evaluation Started: lb {lb_rolling_pr['window_kwargs']}")
            rolling = RollingPeriods(**lb_rolling_pr)
            fit_periods = rolling.fit_periods
            for fp, dfp in list(zip(fit_periods, data_fit_periods)):
                fe.eval_one_period(*fp, *dfp)
            self.log.success(f"Evaluation Finished: lb {lb_rolling_pr['window_kwargs']}")
            
    def run(self):
        self._init_director_processor()
        self.multi_test()
        self.evaluate()


# %% main
def main():
    '''read args'''
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', '--agg_eval_name', type=str, help='agg_eval_name')
    parser.add_argument('-twkr', '--test_wkr', type=int, help='test_wkr')
    parser.add_argument('-ewkr', '--eval_wkr', type=int, help='eval_wkr')

    args = parser.parse_args()
    agg_eval_name, test_wkr, eval_wkr = args.agg_eval_name, args.test_wkr, args.eval_wkr
    
    agg_test_eval = AggTestEval(agg_eval_name, test_wkr, eval_wkr)
    agg_test_eval.run()
    

# %% main
if __name__ == "__main__":
    main()
