# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 10:37:19 2024

@author: Xintang Zheng

"""
# %% 
'''
neu改为输入，不靠参数传递
'''
# %% import public
import sys
import yaml
from pathlib import Path
import argparse
import warnings
warnings.filterwarnings("ignore")


# %% add sys path
file_path = Path(__file__).resolve()
file_dir = file_path.parents[0]
project_dir = file_path.parents[1]
sys.path.append(str(project_dir))


# %% import self-defined
from utils.dirutils import load_path_config
from test_and_eval.factor_tester_adaptive import FactorTest

        
# %% main
def main():
    # init dir
    path_config = load_path_config(project_dir)
    factor_data_dir = path_config['factor_data']
    
    '''read args'''
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--process_name', type=str, help='process_name')
    parser.add_argument('-tag', '--tag_name', type=str, default=None, help='tag_name')
    parser.add_argument('-fdir', '--factor_data_dir', type=str, default=factor_data_dir, help='factor_data_dir')
    parser.add_argument('-t', '--test_name', type=str, default='regular_twd30', help='test_name')
    parser.add_argument('-skip', '--skip_plot', type=bool, default=False, help='skip_plot')
    parser.add_argument('-wkr', '--n_workers', type=int, default=1, help='n_workers')
    parser.add_argument('-se', '--skip_exists', action='store_true', help='skip_exists')
    args = parser.parse_args()
    args_dict = {k: v for k, v in vars(args).items() if k != 'skip_exists'}

    # main
    tester = FactorTest(**args_dict)
    tester.test_multi_factors(skip_exists=args.skip_exists)
        

# %% main
if __name__=='__main__':
    main()
