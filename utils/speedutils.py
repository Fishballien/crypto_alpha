# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 11:14:27 2024

@author: Xintang Zheng

"""
# %% imports
import time
import gc
from functools import wraps, partial
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
from itertools import chain


# %%
def timeit(func):
    """装饰器函数，用于测量函数执行时间"""
    def wrapper(*args, **kwargs):
        start_time = time.time()  # 记录函数开始时间
        result = func(*args, **kwargs)  # 调用函数
        end_time = time.time()  # 记录函数结束时间
        print(f"{func.__name__} ran in {end_time - start_time:.4f} seconds")
        return result
    return wrapper


# %%
def gc_collect_after(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)  # 调用原始函数
        gc.collect()  # 在函数执行后调用垃圾回收
        return result
    return wrapper


# %% multiprocessing
def split_list(lst, n):
    """将列表分割成 n 组."""
    avg = len(lst) / float(n)
    out = []
    last = 0.0

    while last < len(lst):
        out.append(lst[int(last):int(last + avg)])
        last += avg

    return out


def join_lists(lists):
    """将列表中的子列表连接成一个列表."""
    return list(chain.from_iterable(lists))


def process_one_group(paras, func):
    try:
        return [func(*p) if type(p) is tuple else func(p) for p in paras]
    except:
        print(type(paras[0]))
        raise


# @timeit
def multiprocess_with_sequenced_result(func, paras, n_process, desc=''):
    if n_process == 1:
        return [func(*p) if type(p) is tuple else func(p) for p in paras]
    splited_para_list = split_list(paras, n_process)
    process_one_group_func = partial(process_one_group, func=func)
    with ProcessPoolExecutor(max_workers=n_process) as executor:
        results = list(tqdm(executor.map(process_one_group_func, splited_para_list), 
                            total=len(splited_para_list), desc=desc))
    return join_lists(results)