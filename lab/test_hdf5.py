# -*- coding: utf-8 -*-
"""
Created on Sat May 11 15:44:20 2024

@author: Xintang Zheng

"""
import h5py
import numpy as np

with h5py.File('./test_hdf5.h5', 'a') as hf:
    print(np.array(hf['a']))
    # if 'a' in hf:
    #     del hf['a']
    # hf.create_dataset('a', data=np.array([4, 2, 3]))