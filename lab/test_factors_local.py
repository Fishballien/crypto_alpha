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


# %%
process_name = 'funding'
tag_name = 'zxt_test'
factor_data_dir = r'D:\mnt\Data\Crypto\ProcessedData\15m_cross_sectional'
test_name = 'regular_twd30'

        
# %% main
tester = FactorTest(process_name=process_name, tag_name=tag_name, factor_data_dir=factor_data_dir, 
                    test_name=test_name)
tester.test_one_factor('funding_filled_mmt12h_ma')
        

