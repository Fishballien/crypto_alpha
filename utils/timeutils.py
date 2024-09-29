# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 14:12:13 2024

@author: Xintang Zheng

"""
# %%
import numpy as np
import pandas as pd
from dateutil import rrule, relativedelta
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import re


# %%
DAY_SEC = 24*60*60
HOUR_SEC = 60*60
MIN_SEC = 60
DATA_FREQ = 3


# %%
def timestr_to_seconds(time_str):
    td = pd.to_timedelta(time_str)
    return td.total_seconds()


def timestr_to_minutes(time_str):
    td = pd.to_timedelta(time_str)
    return int(td.total_seconds() / 60)


def timedelta_to_seconds(time_delta_info):
    time_delta = timedelta(**time_delta_info)
    return time_delta / timedelta(seconds=1)


def datetime_to_shortcut(dt):
    return dt.strftime('%y%m%d')


# %%
def get_period(start_date, end_date):
    return [dt.strftime("%Y-%m-%d") for dt in rrule.rrule(rrule.DAILY, dtstart=start_date, until=end_date)]


# %%
@dataclass
class RollingPeriods(object):
    
    fstart: datetime
    pstart: datetime
    puntil: datetime
    window_kwargs: dict = field(default_factory=dict)
    rrule_kwargs: dict = field(default_factory=dict)  # HINT💡 like: {'freq': 'M', 'bymonthday': -1}
    end_by: str = field(default_factory='')
    
    def __post_init__(self):
        FREQNAMES = rrule.FREQNAMES[:5]
        # FREQNAMES = ['YEARLY', 'MONTHLY', 'WEEKLY', 'DAILY', 'HOURLY']
        freq_mapping = {FREQNAMES[i][0]: i for i in range(len(FREQNAMES))}
        # {'Y': 0, 'M': 1, 'W': 2, 'D': 3, 'H': 4}
        if (freq := freq_mapping.get(self.rrule_kwargs['freq'], None)) is not None:
            self.rrule_kwargs['freq'] = freq
        cut_points = list(rrule.rrule(
            **self.rrule_kwargs, 
            dtstart=self.pstart, until=self.puntil))
        # breakpoint()
        if self.pstart == cut_points[0]:
            cut_points = cut_points[1:] if cut_points else []
        if cut_points and self.puntil == cut_points[-1]:
            cut_points = cut_points[:-1] if cut_points else []
        pred_period_start = [self.pstart] + cut_points
        pred_period_end = [cut_point - timedelta(days=1) if self.end_by == 'date' else cut_point
                           for cut_point in cut_points] + [self.puntil if self.end_by == 'date' 
                                                           else self.puntil + timedelta(days=1)]
        windows = relativedelta.relativedelta(**self.window_kwargs)
        fit_period_start = [max(dt - windows, self.fstart) for dt in pred_period_start]
        assert self.end_by in ['date', 'time']
        fit_period_end = [dt - relativedelta.relativedelta(days=1) if self.end_by == 'date' else dt 
                          for dt in pred_period_start]
        self.predict_periods = list(zip(pred_period_start, pred_period_end))
        self.fit_periods = list(zip(fit_period_start, fit_period_end))
        
        
def generate_timeline_params(start_date, end_date):
    return {
        'start_date': start_date,
        'end_date': end_date,
        'data_start_date': start_date,
        'data_end_date': end_date + timedelta(days=1),
        }


def period_shortcut(start_date, end_date):
    return f'{datetime_to_shortcut(start_date)}_{datetime_to_shortcut(end_date)}'


def translate_rolling_params(rolling_params):
    for pr in ['fstart', 'pstart', 'puntil']:
        rolling_params[pr] = datetime.strptime(rolling_params[pr], '%Y%m%d')
    return rolling_params


# %%
def get_eq_spaced_intraday_time_series(date: datetime, params, mode='r'):
    if mode == 'r':
        start_time = date + timedelta(**params)
        end_time = date + timedelta(days=1) + timedelta(microseconds=1)
    elif mode == 'l':
        start_time = date
        end_time = date + timedelta(days=1)
    interval = timedelta(**params)
    time_series = np.arange(start_time, end_time, interval).astype('i8') // 1e3
    # time_series[-1] -= 100 # 500ms
    return time_series


# %%
def get_wd_name(wd_pr):
    key, value = next(iter(wd_pr.items()))
    return f"{key}_{value}"


# %% relativedelta
def parse_relativedelta(time_string):
    """
    解析时间间隔字符串并返回 relativedelta 对象。

    参数:
    time_string : str - 表示时间间隔的字符串，比如 "2 years, 3 months"。

    返回值:
    relativedelta - 对应时间间隔的 relativedelta 对象。
    """
    time_string = time_string.lower()
    units = {
        'years': 'years',
        'year': 'years',
        'months': 'months',
        'month': 'months',
        'weeks': 'weeks',
        'week': 'weeks',
        'days': 'days',
        'day': 'days',
        'hours': 'hours',
        'hour': 'hours',
        'minutes': 'minutes',
        'minute': 'minutes',
        'seconds': 'seconds',
        'second': 'seconds',
    }
    
    # 使用正则表达式提取所有的时间单位和数值
    matches = re.findall(r"(\d+)\s*(years?|months?|weeks?|days?|hours?|minutes?|seconds?)", time_string)
    
    # 构建关键字参数字典
    kwargs = {}
    for value, unit in matches:
        if unit in units:
            kwargs[units[unit]] = int(value)
    
    return relativedelta.relativedelta(**kwargs)