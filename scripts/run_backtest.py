# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 10:37:19 2024

@author: Xintang Zheng

"""
# %% import public
import sys
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
from backtest.backtester import backtest

        
# %% main
def main():
    '''read args'''
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model_name', type=str, help='model_name')
    parser.add_argument('-bt', '--backtest_name', type=str, default=None, help='backtest_name')
    args = parser.parse_args()
    
    backtest(args.model_name, args.backtest_name)


# %% main
if __name__=='__main__':
    main()
