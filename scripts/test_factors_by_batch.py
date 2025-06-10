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
from test_and_eval.batch_test import BatchTest

        
# %% main
def main():
    '''read args'''
    parser = argparse.ArgumentParser()
    parser.add_argument('-btn', '--batch_test_name', type=str, help='batch_test_name')
    parser.add_argument('-dt', "--date_today", type=str, help="Date for processing tasks in YYYYMMDD format")

    args = parser.parse_args()
    batch_test_name = args.batch_test_name
    date_today = args.date_today

    # main
    tester = BatchTest(batch_test_name)
    tester.run(date_today)
        

# %% main
if __name__=='__main__':
    main()
