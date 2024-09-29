# -*- coding: utf-8 -*-
"""
Created on Sun May 14 11:42:43 2023
@author: fishballien

Spyder Mark&Emoji: TODO☕  FIXME🎨  HINT💡  TIP🚩  HACK🔧  BUG🐞  OPTIMIZE🚀  ???👻  !!!⚠
Common Emoji: ⭐ 🌕 🌙 🌞 🌥️ 🌩️ ⚡ 🌈 🔥 💧 ⏳ 💥 🌱 ☘️ 🍀 🌳 🙏 📈🤑 📉😱 ⛔ ✅ ❎
+------------------------------------------------------------------------------
|1.
+------------------------------------------------------------------------------
"""
# %% import
import numpy as np


# %% config


# %% global
CRYPTO_ONEYEAR_TRADING_DAYS = 365


# %% return metrics
def calc_return(return_arr):
    return np.nansum(return_arr)


def calc_return_annualized(return_arr):
    return np.nansum(return_arr) * CRYPTO_ONEYEAR_TRADING_DAYS / len(return_arr)


def get_max_drawdown_in_ts(return_arr):
    cum_ret_arr = np.cumsum(return_arr)
    return cum_ret_arr - np.maximum.accumulate(cum_ret_arr) 


def calc_max_drawdown(return_arr):
    net_arr = np.cumsum(return_arr)
    max_net_arr = np.maximum.accumulate(net_arr)
    drawdown_arr = max_net_arr - net_arr
    return np.max(drawdown_arr)


def calc_sharpe(return_arr):
    if return_std := np.std(return_arr):
        return np.mean(return_arr) / return_std * np.sqrt(CRYPTO_ONEYEAR_TRADING_DAYS)
    else:
        return np.nan
    
    
def calc_calmar_ratio(return_arr):
    ret_annualized = np.nansum(return_arr) * CRYPTO_ONEYEAR_TRADING_DAYS / len(return_arr)
    max_dd = calc_max_drawdown(return_arr)
    return ret_annualized / max_dd if max_dd > 0 else np.nan


# def convert_str_to_timedelta(time_str):
#     split_time = time_str.split(" ")
#     days = int(split_time[0])
#     hms = split_time[2].split(":")
#     hours = int(hms[0])
#     minutes = int(hms[1])
#     seconds = float(hms[2])
#     duration = timedelta(days=days, hours=hours, minutes=minutes, seconds=seconds)
#     return duration

# def calc_long_ratio(long_duration, short_duration):
#     long_dura_total = np.sum(np.vectorize(convert_str_to_timedelta)(long_duration))
#     short_dura_total = np.sum(np.vectorize(convert_str_to_timedelta)(short_duration))
#     long_ratio = long_dura_total / (long_dura_total + short_dura_total)
#     return long_ratio


def calc_long_ratio(long_duration_arr, short_duration_arr):
    long_duration = np.sum(long_duration_arr)
    short_duration = np.sum(short_duration_arr)
    if total_duration := long_duration + short_duration:
        long_ratio = long_duration / total_duration
    else:
        long_ratio = np.nan
    return long_ratio


# %% additional - by chatgpt
def get_drawdowns(returns):
    """
    Calculate drawdowns from a series of returns.
    
    :param returns: List or numpy array of daily returns.
    :return: List of drawdowns.
    """
    prices = np.cumprod(1 + returns)
    peak_prices = np.maximum.accumulate(prices)
    drawdowns = (prices - peak_prices) / peak_prices
    return drawdowns


def calculate_sortino_ratio(returns, risk_free_rate=0):
    """
    Calculate the Sortino ratio for a series of daily returns.

    :param returns: List or numpy array of daily returns.
    :param risk_free_rate: Daily risk-free rate, default is 0.
    :return: Sortino ratio.
    """
    # Convert the annual risk-free rate to daily if necessary
    rf_daily = (1 + risk_free_rate) ** (1 / CRYPTO_ONEYEAR_TRADING_DAYS) - 1
    excess_returns = returns - rf_daily
    
    # Calculate the negative returns only
    negative_returns = excess_returns[excess_returns < 0]
    
    # Calculate the mean of excess returns and downside standard deviation
    mean_excess_return = np.mean(excess_returns)
    downside_std = np.sqrt(np.mean(negative_returns ** 2))
    
    # Calculate the Sortino ratio
    if downside_std == 0:
        return np.inf  # To handle division by zero if downside_std is zero
    return mean_excess_return / downside_std * np.sqrt(CRYPTO_ONEYEAR_TRADING_DAYS)


def calculate_sterling_ratio(returns, annual_target=0):
    """
    Calculate the Sterling ratio.
    
    :param returns: List or numpy array of daily returns.
    :param annual_target: The annual target return.
    :return: Sterling ratio.
    """
    cumulative_return = np.cumprod(1 + returns)[-1] - 1
    annual_return = (1 + cumulative_return)**(CRYPTO_ONEYEAR_TRADING_DAYS / len(returns)) - 1
    average_drawdown = np.mean([drawdown for drawdown in get_drawdowns(returns) if drawdown < 0])
    return (annual_return - annual_target) / -average_drawdown


def calculate_burke_ratio(returns, annual_target=0):
    """
    Calculate the Burke ratio.
    
    :param returns: List or numpy array of daily returns.
    :param annual_target: The annual target return.
    :return: Burke ratio.
    """
    try:
        cumulative_return = np.cumprod(1 + returns)[-1] - 1
        annual_return = (1 + cumulative_return)**(CRYPTO_ONEYEAR_TRADING_DAYS / len(returns)) - 1
        drawdowns = np.sqrt(np.mean([drawdown**2 for drawdown in get_drawdowns(returns) if drawdown < 0]))
    except OverflowError:
        print('return', returns)
        print('drawdown', get_drawdowns(returns))
        return np.nan
    return (annual_return - annual_target) / drawdowns


def calculate_ulcer_index(returns): # 溃疡指数
    """
    Calculate the Ulcer Index.
    
    :param returns: List or numpy array of daily returns.
    :return: Ulcer Index.
    """
    prices = np.cumprod(1 + returns)  # Convert returns to price series assuming a base of 1.
    peak_prices = np.maximum.accumulate(prices)
    drawdowns = (prices - peak_prices) / peak_prices
    return np.sqrt(np.mean(drawdowns**2)) * 100


def calculate_drawdown_recovery_ratio(returns):
    """
    Calculate the drawdown and recovery ratio for a series of daily returns.

    :param returns: List or numpy array of daily returns.
    :return: Drawdown recovery ratio.
    """
    # Calculate cumulative returns and their peaks
    cumulative_returns = np.cumprod(1 + returns)
    peaks = np.maximum.accumulate(cumulative_returns)
    drawdowns = (cumulative_returns - peaks) / peaks
    
    # Track drawdown and recovery periods
    drawdown_start = None
    recovery_ratios = []
    
    for i in range(len(drawdowns)):
        if drawdowns[i] < 0 and drawdown_start is None:
            drawdown_start = i  # Start of new drawdown
        elif drawdowns[i] == 0 and drawdown_start is not None:
            # End of drawdown
            drawdown_end = i
            drawdown_depth = np.min(drawdowns[drawdown_start:drawdown_end])
            recovery_time = drawdown_end - drawdown_start
            if drawdown_depth != 0 and recovery_time > 0:
                recovery_ratio = recovery_time / abs(drawdown_depth)
                recovery_ratios.append(recovery_ratio)
            drawdown_start = None  # Reset for next drawdown
            
    # Return the average recovery ratio if any recoveries occurred
    if recovery_ratios:
        return np.mean(recovery_ratios)
    else:
        return np.nan  # No recoveries to calculate


# %%
def get_general_return_metrics(return_arr):
    ret = np.sum(return_arr)
    ret_annualized = np.sum(return_arr) * CRYPTO_ONEYEAR_TRADING_DAYS / len(return_arr)
    max_dd = calc_max_drawdown(return_arr)
    sharpe_ratio = calc_sharpe(return_arr)
    calmar_ratio = ret_annualized / max_dd if max_dd > 0 else np.nan
    sortino_ratio = calculate_sortino_ratio(return_arr)
    sterling_ratio = calculate_sterling_ratio(return_arr)
    burke_ratio = calculate_burke_ratio(return_arr)
    ulcer_index = calculate_ulcer_index(return_arr)
    drawdown_recovery_ratio = calculate_drawdown_recovery_ratio(return_arr)
    return {
        "return": ret,
        "return_annualized": ret_annualized,
        "max_dd": max_dd,
        "sharpe_ratio": sharpe_ratio,
        "calmar_ratio": calmar_ratio,
        "sortino_ratio": sortino_ratio,
        "sterling_ratio": sterling_ratio,
        "burke_ratio": burke_ratio,
        "ulcer_index": ulcer_index,
        "drawdown_recovery_ratio": drawdown_recovery_ratio,
    }


def get_all_metrics(pnl_arr):
    summary = get_general_return_metrics(pnl_arr["return"])
    summary["long_ratio"] = calc_long_ratio(pnl_arr["long_duration"], pnl_arr["short_duration"])
    summary["turnover_ratio"] = np.mean(pnl_arr["long_open"] + pnl_arr["short_open"])
    return summary


# %% __main__
if __name__ == "__main__":
    pass
