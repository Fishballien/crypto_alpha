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
from synthesis.rolling_cluster import RollingCluster

          
# %% main
def main():
    '''read args'''
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--cluster_name', type=str, help='cluster_name')
    parser.add_argument('-pst', '--pstart', type=str, default='20220101', help='pstart')
    parser.add_argument('-pu', '--puntil', type=str, help='puntil')
    parser.add_argument('-ct', '--cluster_type', type=str, help='cluster_type')
    parser.add_argument('-twkr', '--t_workers', type=int, default=1, help='t_workers')
    parser.add_argument('-pwkr', '--p_workers', type=int, default=1, help='p_workers')
    args = parser.parse_args()
    args_dict = vars(args)
    
    c = RollingCluster(**args_dict)
    c.run()
        
        
# %% main
if __name__=='__main__':
    main()
        
        
        
    

