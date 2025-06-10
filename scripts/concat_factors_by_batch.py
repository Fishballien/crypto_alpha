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
from test_and_eval.batch_concat import BatchConcat

        
# %% main
def main():
    '''read args'''
    parser = argparse.ArgumentParser()
    parser.add_argument('-bcn', '--batch_concat_name', type=str, help='batch_concat_name')

    args = parser.parse_args()
    batch_concat_name = args.batch_concat_name

    # main
    tester = BatchConcat(batch_concat_name)
    tester.run()
        

# %% main
if __name__=='__main__':
    main()
