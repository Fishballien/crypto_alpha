# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 17:01:44 2024

@author: Xintang Zheng

"""
# %% imports
import cvxpy as cp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# %% from gpt sansei
# =============================================================================
# def future_optimal_weight_lp_cvxpy(if_jy, alpha, w0, hs_ctl_double, can_trade, if_ctl_list, array_list, updown_list, max_multi=1):
#     # 过滤和调整输入数据
#     alpha = alpha[if_jy == 1]
#     w0 = w0[if_jy == 1] - np.mean(w0[if_jy == 1])
#     w0 /= np.sum(np.abs(w0)) if np.sum(np.abs(w0)) != 0 else w0
#     can_trade = can_trade[if_jy == 1]
#     
#     # 定义变量
#     n = alpha.size
#     w = cp.Variable(n)
#     
#     # 目标函数
#     objective = cp.Maximize(cp.sum(cp.multiply(alpha, w)))
#     
#     # 约束列表
#     constraints = []
#     
#     # 总权重为0
#     constraints.append(cp.sum(w) == 0)
#     
#     # 权重绝对值和为1
#     constraints.append(cp.norm(w, 1) <= 1)
#     
#     # 换手率控制
#     constraints.append(cp.norm(w - w0, 1) <= hs_ctl_double)
#     
#     # 单个资产权重控制
#     alpha_r = (alpha * can_trade).rank(pct=True) - 0.5
#     max_wgt = (alpha_r / np.abs(alpha_r).sum()).max() * max_multi
#     constraints += [w <= max_wgt, w >= -max_wgt]
#     
#     # 不交易资产权重不变
#     for i in range(n):
#         if not can_trade[i]:
#             constraints.append(w[i] == w0[i])
# 
#     # 风险控制约束（如果应用）
#     for idx, ctl in enumerate(if_ctl_list):
#         if ctl:
#             array = array_list[idx]
#             updown = updown_list[idx]
#             constraints.append(cp.sum(cp.multiply(array, w)) <= updown[0])
#             constraints.append(cp.sum(cp.multiply(array, w)) >= -updown[1])
#     
#     # 定义和求解问题
#     problem = cp.Problem(objective, constraints)
#     problem.solve(solver=cp.GLPK, verbose=True)
#     
#     return w.value, problem.status
# =============================================================================


# %% simple
def adjustable_cubic_function(x, n=3, y_start=0.5, y_end=1):
    y = 1 - (1 - x)**n
    # 线性变换，将范围变换到 [y_start, y_end]
    y = y_start + (y_end - y_start) * y
    return y


def future_optimal_weight_lp_cvxpy(alpha, w0, if_to_trade, tradv_t, mm_t, his_pft_t, to_rate_thresh, 
                                   max_multi=1, max_wgt=None, tradv_thresh=None, momentum_limits={}, pf_limits={},
                                   _lambda=1, steepness=None, min_wgt=0.5):
    # init
    final_w1 = np.zeros(len(w0))
    # 过滤和调整输入数据
    alpha = alpha[if_to_trade == 1]
    w0 = w0[if_to_trade == 1]
    tradv_t = tradv_t[if_to_trade == 1]
    mm_t = {mm_wd: mm[if_to_trade == 1] for mm_wd, mm in mm_t.items()}
    pf_t = {pf_wd: pf[if_to_trade == 1] for pf_wd, pf in his_pft_t.items()}
    w0 = w0 - np.mean(w0) # ???
    w0 = w0 / np.sum(np.abs(w0)) if np.sum(np.abs(w0)) != 0 else w0
    
    # 定义变量
    n = alpha.size
    w = cp.Variable(n)
    
    # 目标函数
    objective = cp.Maximize(cp.sum(cp.multiply(alpha, w)))
# =============================================================================
#     target_w = alpha / np.sum(np.abs(alpha))
#     objective = cp.Maximize(- cp.norm(w - target_w, 1))
# =============================================================================
    # objective = cp.Maximize(cp.sum(cp.multiply(alpha, w)) - _lambda * cp.norm(w, 2))
    
    # 约束列表
    constraints = []
    
    # 总权重为0
    constraints.append(cp.sum(w) == 0)
    
    # 权重绝对值和为1
    constraints.append(cp.norm(w, 1) <= 1)
    
    # 换手率控制
    constraints.append(cp.norm(w - w0, 1) <= to_rate_thresh * 2) # ???
    
    # 单个资产权重控制
    if max_wgt is None:
        if steepness is None:
            alpha_r = pd.Series(alpha)
            alpha_r = alpha_r.rank(pct=True).sub(0.5 / alpha_r.count()).replace([np.inf, -np.inf], np.nan) - 0.5 # ???: 重复？
            max_wgt = (alpha_r / np.abs(alpha_r).sum()).max() * max_multi # org
            # max_wgt = np.abs(alpha) * max_multi
            constraints += [w <= max_wgt, w >= -max_wgt]
        else:
            alpha_r = pd.Series(alpha)
            alpha_r = (alpha_r.rank(pct=True).sub(0.5 / alpha_r.count()).replace([np.inf, -np.inf], np.nan) - 0.5) * 2
            # sigmoid = 1 / (1 + np.exp(-steepness * (alpha_r.abs() - 0.5)))
            adj_alpha = adjustable_cubic_function(alpha_r.abs(), steepness, y_start=min_wgt)
            alpha_sign = np.sign(alpha)
            alpha_sign = np.where(alpha_sign==0, 1, alpha_sign)
            adj_alpha *= alpha_sign
            max_wgt = adj_alpha / adj_alpha.abs().sum() * max_multi
            # breakpoint()
            constraints += [
               w >= np.minimum(-np.min(max_wgt.abs()), max_wgt.values),
               w <= np.maximum(np.max(max_wgt.abs()), max_wgt.values)
               ]
    else:
        constraints += [w <= max_wgt, w >= -max_wgt]
    
    # 流动性约束
    if tradv_thresh > 0:
        constraints.append(w[tradv_t < tradv_thresh] == 0)
    
    # 动量约束
    for mm_wd in momentum_limits:
        mm_sum = cp.sum(mm_t[mm_wd] @ w)
        mm_thres = momentum_limits[mm_wd]
        constraints += [
            mm_sum >= - mm_thres,
            mm_sum <= mm_thres
        ]
        
    # 盈亏约束
    for pf_wd in pf_limits:
        pf_sum = cp.sum(pf_t[pf_wd] @ w)
        pf_thres = pf_limits[pf_wd]
        constraints += [
            pf_sum >= - pf_thres,
            pf_sum <= pf_thres
        ]
        
    problem = cp.Problem(objective, constraints)
    try:
        problem.solve(solver=cp.ECOS, verbose=False)
    except:
        return w0, 'error', None
    w1 = w.value
    
    # if np.max(np.abs(w1)) > 0.05:
    #     breakpoint()

    # 定义和求解问题
    try:
        w1 = w1 / pd.Series(w1).abs().sum()
    except:
        w1 = w0

    final_w1[if_to_trade == 1] = w1
    return final_w1, problem.status, max_wgt


def clean_matrix(matrix):
    """处理矩阵中的 NaN 和 inf 值"""
    matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)
    return matrix


def ensure_positive_definite(matrix, epsilon=1e-6, max_iter=1000):
    """确保矩阵是正定的，通过逐步增加对角线元素的方式"""
    matrix = clean_matrix(matrix)  # 清理矩阵中的 NaN 和 inf 值
    k = 0
    while k < max_iter:
        try:
            # 尝试进行Cholesky分解，如果成功则说明矩阵是正定的
            np.linalg.cholesky(matrix)
            return matrix
        except np.linalg.LinAlgError:
            # 如果分解失败，说明矩阵不是正定的，增加对角线元素
            matrix += np.eye(matrix.shape[0]) * epsilon
            k += 1
    raise np.linalg.LinAlgError("无法将矩阵转换为正定矩阵")


def future_optimal_weight_lp_cvxpy_with_var(alpha, w0, if_to_trade, tradv_t, mm_t, to_rate_thresh, 
                                            max_multi=1, max_wgt=None, tradv_thresh=None, momentum_limits={},
                                            cov_matrix=None, _lambda=1, epsilon=1e-6, max_iter=1000):
    # init
    final_w1 = np.zeros(len(w0))
    # 过滤和调整输入数据
    alpha = alpha[if_to_trade == 1]
    w0 = w0[if_to_trade == 1]
    tradv_t = tradv_t[if_to_trade == 1]
    mm_t = {mm_wd: mm[if_to_trade == 1] for mm_wd, mm in mm_t.items()}
    w0 = w0 - np.mean(w0) # 调整初始权重
    w0 = w0 / np.sum(np.abs(w0)) if np.sum(np.abs(w0)) != 0 else w0
    
    # 定义变量
    n = alpha.size
    w = cp.Variable(n)
    
    # 目标函数
    if cov_matrix is not None:
        cov_matrix = ensure_positive_definite(cov_matrix, epsilon, max_iter)
        risk_term = cp.quad_form(w, cov_matrix)  # 使用显式的矩阵运算替代
        objective = cp.Maximize(cp.sum(cp.multiply(alpha, w)) - _lambda * risk_term)
    else:
        objective = cp.Maximize(cp.sum(cp.multiply(alpha, w)))  # 如果没有协方差矩阵，则最大化alpha
    
    # 约束列表
    constraints = []
    
    # 总权重为0
    constraints.append(cp.sum(w) == 0)
    
    # 权重绝对值和为1
    constraints.append(cp.norm(w, 1) <= 1)
    
    # 换手率控制
    constraints.append(cp.norm(w - w0, 1) <= to_rate_thresh * 2) # 控制换手率
    
    # 单个资产权重控制
    if max_wgt is None:
        alpha_r = pd.Series(alpha)
        alpha_r = alpha_r.rank(pct=True).sub(0.5 / alpha_r.count()).replace([np.inf, -np.inf], np.nan) - 0.5
        max_wgt = (alpha_r / np.abs(alpha_r).sum()).max() * max_multi
    constraints += [w <= max_wgt, w >= -max_wgt]
    
    # 流动性约束
    if tradv_thresh is not None and tradv_thresh > 0:
        constraints.append(w[tradv_t < tradv_thresh] == 0)
    
    # 动量约束
    for mm_wd in momentum_limits:
        mm_sum = cp.sum(mm_t[mm_wd] @ w)
        mm_thres = momentum_limits[mm_wd]
        constraints += [
            mm_sum >= - mm_thres,
            mm_sum <= mm_thres
        ]

    # 定义和求解问题
    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.ECOS, verbose=False)
    w1 = w.value
    
    try:
        w1 = w1 / pd.Series(w1).abs().sum()
    except:
        w1 = w0

    final_w1[if_to_trade == 1] = w1
    return final_w1, problem.status, max_wgt


# %% long short
def optimize_with_level(direction, alpha, w0, if_to_trade, tradv_t, mm_t, to_rate_thresh_l0, to_rate_thresh_l1, opt_func=None):
    w1, status, max_wgt = opt_func(direction, alpha, w0, if_to_trade, tradv_t, mm_t, to_rate_thresh_l0)
    if status != "optimal":
        print(f'round 1 is not optimal... {status}')
        w1, status, max_wgt = opt_func(direction, alpha, w0, if_to_trade, tradv_t, mm_t, to_rate_thresh_l1)
        if status != "optimal":
            w1 = w0.copy()
    return w1
    

def optimize_long_short(alpha, w0, if_to_trade, tradv_t, mm_t, to_rate_thresh_l0, optimize_with_level=None):
    w0_long = np.where(w0 > 0, w0*2, 0)
    w1_long = optimize_with_level(1, alpha, w0_long, if_to_trade, tradv_t, mm_t, to_rate_thresh_l0)
    w0_short = np.where(w0 < 0, w0*2, 0)
    w1_short = optimize_with_level(-1, alpha, w0_short, if_to_trade, tradv_t, mm_t, to_rate_thresh_l0)
    w1 = (w1_long + w1_short) / 2
    return w1
    
    
def future_optimal_weight_lp_cvxpy_long_short(direction, alpha, w0, if_to_trade, tradv_t, mm_t, to_rate_thresh, 
                                              max_multi=1, max_wgt=None, tradv_thresh=None, momentum_limits={},
                                              _lambda=1):
    # init
    final_w1 = np.zeros(len(w0))
    # 过滤和调整输入数据
    alpha = alpha[if_to_trade == 1]
    w0 = w0[if_to_trade == 1]
    tradv_t = tradv_t[if_to_trade == 1]
    mm_t = {mm_wd: mm[if_to_trade == 1] for mm_wd, mm in mm_t.items()}
    # w0 = w0 - np.mean(w0) # ???
    w0 = w0 / np.sum(np.abs(w0)) if np.sum(np.abs(w0)) != 0 else w0
    
    # 定义变量
    n = alpha.size
    w = cp.Variable(n)
    
    # 目标函数
    objective = cp.Maximize(cp.sum(cp.multiply(alpha, w)))
# =============================================================================
#     target_w = alpha / np.sum(np.abs(alpha))
#     objective = cp.Maximize(- cp.norm(w - target_w, 1))
# =============================================================================
    # objective = cp.Maximize(cp.sum(cp.multiply(alpha, w)) - _lambda * cp.norm(w, 2))
    
    # 约束列表
    constraints = []
    
    # 总权重为0
    constraints.append(cp.sum(w) == direction)
    
    # 换手率控制
    constraints.append(cp.norm(w - w0, 1) <= to_rate_thresh * 2) # ???
    
    # 单个资产权重控制
    if max_wgt is None:
        alpha_r = pd.Series(alpha)
        alpha_r = alpha_r.rank(pct=True).sub(0.5 / alpha_r.count()).replace([np.inf, -np.inf], np.nan) - 0.5 # ???: 重复？
        max_wgt = (alpha_r / np.abs(alpha_r).sum()).max() * max_multi # org
        # max_wgt = np.abs(alpha) * max_multi
        # breakpoint()
    constraints += [w <= max_wgt, w >= 0] if direction == 1 else [w <= 0, w >= -max_wgt]
        
    
# =============================================================================
#     # 流动性约束
#     if tradv_thresh > 0:
#         constraints.append(w[tradv_t < tradv_thresh] == 0)
#     
#     # 动量约束
#     for mm_wd in momentum_limits:
#         mm_sum = cp.sum(mm_t[mm_wd] @ w)
#         mm_thres = momentum_limits[mm_wd]
#         constraints += [
#             mm_sum >= - mm_thres,
#             mm_sum <= mm_thres
#         ]
# =============================================================================

    # 定义和求解问题
    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.ECOS, verbose=False)
    w1 = w.value
    
    try:
        w1 = w1 / pd.Series(w1).abs().sum()
    except:
        w1 = w0

    final_w1[if_to_trade == 1] = w1
    return final_w1, problem.status, max_wgt


# %% add constraints only
def optimize_with_constraints_momentum(alpha, if_to_trade, mm_t, momentum_limits={}, max_wgt=None, max_multi=None, 
                                       epsilon=None):
    # init
    final_w1 = np.zeros(len(alpha))
    # 过滤和调整输入数据
    alpha = alpha[if_to_trade == 1]
    mm_t = {mm_wd: mm[if_to_trade == 1] for mm_wd, mm in mm_t.items()}

    # 定义变量
    n = alpha.size
    w = cp.Variable(n)
    
    # 目标函数
    objective = cp.Maximize(cp.sum(cp.multiply(alpha, w)))
    
    # 约束列表
    constraints = []
    
    # # 确保每个权重都不相同
    # for i in range(n):
    #     for j in range(i+1, n):
    #         constraints.append(cp.abs(w[i] - w[j]) >= epsilon)
    
    # 总权重为0
    constraints.append(cp.sum(w) == 0)
    
    # 权重绝对值和为1
    constraints.append(cp.norm(w, 1) <= 1)
    
    # 单个资产权重控制
    if max_wgt is None:
        alpha_r = pd.Series(alpha)
        alpha_r = alpha_r.rank(pct=True).sub(0.5 / alpha_r.count()).replace([np.inf, -np.inf], np.nan) - 0.5 # ???: 重复？
        max_wgt = (alpha_r / np.abs(alpha_r).sum()).max() * max_multi
    constraints += [w <= max_wgt, w >= -max_wgt]
    
    # 动量约束
    for mm_wd in momentum_limits:
        mm_sum = cp.sum(mm_t[mm_wd] @ w)
        mm_thres = momentum_limits[mm_wd]
        constraints += [
            mm_sum >= - mm_thres,
            mm_sum <= mm_thres
        ]

    # 定义和求解问题
    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.GLPK, verbose=False)
    w1 = w.value

    final_w1[if_to_trade == 1] = w1
    return final_w1, problem.status


# %% optimize hsr constraint
def future_optimal_weight_with_to_rate_opt(t, alpha, w0, if_to_trade, tradv_t, mm_t, to_rate_params, 
                                       max_multi=1, max_wgt=None, tradv_thresh=None, momentum_limits={},
                                       _lambda=1, backtest_dir=None):
    # init
    final_w1 = np.zeros(len(w0))
    # 过滤和调整输入数据
    alpha = alpha[if_to_trade == 1]
    w0 = w0[if_to_trade == 1]
    tradv_t = tradv_t[if_to_trade == 1]
    mm_t = {mm_wd: mm[if_to_trade == 1] for mm_wd, mm in mm_t.items()}
    w0 = w0 - np.mean(w0) # ???
    w0 = w0 / np.sum(np.abs(w0)) if np.sum(np.abs(w0)) != 0 else w0
    
    
    to_rate_arr = np.arange(*to_rate_params)
    w_list = []
    alpha_rtn_arr = np.zeros_like(to_rate_arr)
    # fee_arr = np.zeros_like(to_rate_arr)
    
    for i, to_rate_thresh in enumerate(to_rate_arr):
        opt_failed = False
        try:
            # 定义变量
            n = alpha.size
            w = cp.Variable(n)
            
            # 目标函数
            objective = cp.Maximize(cp.sum(cp.multiply(alpha, w)))
        # =============================================================================
        #     target_w = alpha / np.sum(np.abs(alpha))
        #     objective = cp.Maximize(- cp.norm(w - target_w, 1))
        # =============================================================================
            # objective = cp.Maximize(cp.sum(cp.multiply(alpha, w)) - _lambda * cp.norm(w, 2))
            
            # 约束列表
            constraints = []
            
            # 总权重为0
            constraints.append(cp.sum(w) == 0)
            
            # 权重绝对值和为1
            constraints.append(cp.norm(w, 1) <= 1)
            
            # 换手率控制
            constraints.append(cp.norm(w - w0, 1) <= to_rate_thresh * 2) # ???
            
            # 单个资产权重控制
            if max_wgt is None:
                alpha_r = pd.Series(alpha)
                alpha_r = alpha_r.rank(pct=True).sub(0.5 / alpha_r.count()).replace([np.inf, -np.inf], np.nan) - 0.5 # ???: 重复？
                max_wgt = (alpha_r / np.abs(alpha_r).sum()).max() * max_multi
                # breakpoint()
            constraints += [w <= max_wgt, w >= -max_wgt]
            
            # 流动性约束
            if tradv_thresh > 0:
                constraints.append(w[tradv_t < tradv_thresh] == 0)
            
            # 动量约束
            for mm_wd in momentum_limits:
                mm_sum = cp.sum(mm_t[mm_wd] @ w)
                mm_thres = momentum_limits[mm_wd]
                constraints += [
                    mm_sum >= - mm_thres,
                    mm_sum <= mm_thres
                ]
        
            # 定义和求解问题
            problem = cp.Problem(objective, constraints)
            problem.solve(solver=cp.ECOS, verbose=False)
            w1 = w.value
        except:
            opt_failed = True
        
        try:
            w1 = w1 / pd.Series(w1).abs().sum()
        except:
            w1 = w0
        
        w_list.append(w1)
        if opt_failed or problem.status == 'optimal':
            alpha_rtn_arr[i] = np.sum(alpha * w1)
            # fee_arr[i] = np.sum(np.abs(w1 - w0))
        # else:
            # fee_arr[i] = 1000
    max_alpha_rtn = np.sum(alpha * alpha)
    min_alpha_rtn = np.sum(w0 * alpha)
    max_fee = np.sum(np.abs(alpha - w0))
    
    FONTSIZE_L1 = 20
    FONTSIZE_L2 = 18
    FONTSIZE_L3 = 15

    fig = plt.figure(figsize=(10, 20), dpi=100, layout="constrained")
    spec = fig.add_gridspec(ncols=1, nrows=3)
    
    x = to_rate_arr
    y = alpha_rtn_arr-min_alpha_rtn

    ax0 = fig.add_subplot(spec[0, :])
    ax0.set_title(f'{str(t)}', fontsize=FONTSIZE_L1, pad=25)
    ax0.scatter(x, y, label='alpha_rtn')
    ax0.set_xlim([0, max(x)])
    ax0.set_ylim([0, max(y)])

    ax1 = fig.add_subplot(spec[1, :])
    ax1.scatter(x[1:], np.diff(y), label='diff')
    ax1.set_xlim([0, max(x)])
    ax1.set_ylim([0, 0.0004])
    
    ax2 = fig.add_subplot(spec[2, :])
    ax2.scatter(x[2:], np.diff(np.diff(y)), label='diff')
    ax2.set_xlim([0, max(x)])

    for ax in [ax0, ax1, ax2]:
        ax.grid(linestyle=":")
        ax.legend(loc="upper left", borderaxespad=0.5, fontsize=FONTSIZE_L3)
        ax.tick_params(labelsize=FONTSIZE_L2, pad=15)

    plt.savefig(backtest_dir / f'''{str(t).replace(':', '-').replace(' ', '_')}.jpg''', dpi=100, bbox_inches="tight")
    plt.close()
    
# =============================================================================
#     plt.figure(figsize=(10, 10))
#     plt.title(str(t), pad=15, fontsize=12)
#     plt.grid(linestyle=':')
#     plt.scatter(alpha_rtn_arr-min_alpha_rtn, fee_arr/2)
#     plt.xlim([0, max(alpha_rtn_arr-min_alpha_rtn)])
#     plt.ylim([0, max(fee_arr/2)])
#     plt.savefig(backtest_dir / f'''{str(t).replace(':', '-').replace(' ', '_')}.jpg''', dpi=300, bbox_inches='tight')
#     plt.close()
# =============================================================================
    best_i = 0
    for i, rtn_diff in enumerate(np.diff(y)):
        if 0 < rtn_diff < 0.00015:
            best_i = i
            break

    # rate = (alpha_rtn_arr / max_alpha_rtn) / (fee_arr / max_fee)
    # best_i = np.argmax(rate)
    w1 = w_list[best_i]
    final_w1[if_to_trade == 1] = w1
    return final_w1, problem.status, max_wgt

