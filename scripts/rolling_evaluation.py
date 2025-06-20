# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 10:37:19 2024

@author: Xintang Zheng

"""
# %% imports
import sys
import argparse
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")


# %% add sys path
file_path = Path(__file__).resolve()
file_dir = file_path.parents[0]
project_dir = file_path.parents[1]
sys.path.append(str(project_dir))


# %% import self-defined
from test_and_eval.rolling_eval import RollingEval

          
# %% main
def main():
    '''read args'''
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--eval_name', type=str, help='eval_name')
    parser.add_argument('-er', '--eval_rolling_name', type=str, help='eval_rolling_name')
    parser.add_argument('-pst', '--pstart', type=str, default='20220101', help='pstart')
    parser.add_argument('-pu', '--puntil', type=str, help='puntil')
    parser.add_argument('-et', '--eval_type', type=str, help='eval_type')
    parser.add_argument('-wkr', '--n_workers', type=int, default=1, help='n_workers')
    args = parser.parse_args()
    args_dict = vars(args)
    
    e = RollingEval(**args_dict)
    e.run()
        
        
# %% main
if __name__=='__main__':
    main()
        
        
        
    

