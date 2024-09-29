# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 16:27:27 2024

@author: Xintang Zheng

"""
# %% imports
from collections import namedtuple


# %%
NeuGradients = namedtuple('NeuGradients', ['name', 'gradients'])


# %%
neu_0 = NeuGradients(name='neu_0', gradients=[('fma15_sp15', 'askv_sum'),])
neu_1 = NeuGradients(name='neu_1', gradients=[('curr_price', 11),
                                              ('curr_price', 44),])
neu_2 = NeuGradients(name='neu_2', gradients=[('ma1440_sp240', 'tradv'),])
neu_3 = NeuGradients(name='neu_3', gradients=[('ma1440_sp240', 'bias'),])

neu_tf = NeuGradients(name='neu_tf', gradients=[('factor_style', 'rev44h'),
                                                ('factor_style', 'rev176h'),
                                                ('factor_style', 'volatility120_ret4h'),])

neu_tradv = NeuGradients(name='neu_tradv', gradients=[('factor_style', 'rev44h'),
                                                      ('factor_style', 'rev176h'),
                                                      ('factor_style', 'volatility120_ret4h'),
                                                      ('factor_style', 'lnturnover60'),
                                                      ])