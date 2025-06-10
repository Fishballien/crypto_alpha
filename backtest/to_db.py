# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 14:01:40 2025

@author: Xintang Zheng

星星: ★ ☆ ✪ ✩ 🌟 ⭐ ✨ 🌠 💫 ⭐️
勾勾叉叉: ✓ ✔ ✕ ✖ ✅ ❎
报警啦: ⚠ ⓘ ℹ ☣
箭头: ➔ ➜ ➙ ➤ ➥ ↩ ↪
emoji: 🔔 ⏳ ⏰ 🔒 🔓 🛑 🚫 ❗ ❓ ❌ ⭕ 🚀 🔥 💧 💡 🎵 🎶 🧭 📅 🤔 🧮 🔢 📊 📈 📉 🧠 📝

"""
# %%
from pathlib import Path
import pandas as pd
import pymysql
from datetime import datetime, timedelta
from tqdm import tqdm
import numpy as np


# %%
mysqldb_conn = pymysql.connect(host='115.159.152.138', port=3306, user='pyprogram', passwd='K@ra0Key', db='btcfuture',charset='utf8mb4')
cursor = mysqldb_conn.cursor()


# %%
def ps_to_st_report_python_temp(dbconn, cursor, dfall, acname, fromtime, ifdelete=False):
    if ifdelete:
        sql = "DELETE FROM btcfuture.st_report_python_temp WHERE acname = '{}'".format(acname)
        cursor.execute(sql)
        dbconn.commit()

    df = dfall[dfall.index >= fromtime]
    symbollist = df.columns
    
    for symbol in tqdm(symbollist, desc=f'writing to db [{acname}]'):
        # print(symbol)
        df_symbol = df[[symbol]].copy()
        df_symbol.columns = ['ps']
        df_symbol = df_symbol[df_symbol['ps'] != df_symbol['ps'].shift()].dropna()
        df_symbol = df_symbol.reset_index()
        
        values = [
            (acname, symbol, item[1], f"{acname}_{symbol}", item[0]) 
            for item in df_symbol.values
        ]
        
        sql = """
        INSERT INTO btcfuture.st_report_python_temp (acname, symbol, P, st, stockdate)
        VALUES (%s, %s, %s, %s, %s)
        """
        cursor.executemany(sql, values)
        
        dbconn.commit()
        
# =============================================================================
#         for item in df_symbol.values:
#             sql = """
#             INSERT INTO btcfuture.st_report_python_temp (acname, symbol, P, st, stockdate)
#             VALUES ('{}', '{}', {}, '{}', '{}')
#             """.format(acname, symbol, item[1], f"{acname}_{symbol}", item[0])
#             cursor.execute(sql)
#         
#         dbconn.commit()
# =============================================================================
        
        
# %%
param = {
    'agg_241114_to_00125_v0': {
        'model_name': 'merge_agg_241227_cgy_zxt_double3m_15d_73',
        'backtest_name': 'to_00125_maxmulti_2_mm_03_pf_001',
        },
    # 'alpha_241227_sp60': {
    #     'model_name': 'merge_agg_241227_cgy_zxt_double3m_15d_73',
    #     'backtest_name': 'to_0025_maxmulti_2_mm_03_pf_001_sp60',
    #     },
    }

model_dir = Path(r'D:\crypto\multi_factor\factor_test_by_alpha\results\model')
# model_dir = Path(r'/mnt/Data/xintang/crypto/multi_factor/factor_test_by_alpha/results/model')
twap_data_dir = Path(r'/mnt/Data/Crypto/ProcessedData/updated_twap')
fromtime = datetime(2023, 1, 1)

curr_px_path = twap_data_dir / f'curr_price_sp30.parquet' # !!!
curr_price = pd.read_parquet(curr_px_path).loc['20230101':]
rtn_1p = np.log(curr_price.shift(-1) / curr_price)
ma_price = curr_price.rolling('2d').mean()

for acname, pr_info in param.items():
    model_name = pr_info['model_name']
    backtest_name = pr_info['backtest_name']
    backtest_dir = model_dir / model_name /'backtest' / backtest_name
    pos_filename = f'pos_{model_name}__{backtest_name}'
    pos_path = backtest_dir / f'{pos_filename}.parquet'
    pos = pd.read_parquet(pos_path)
    pos *= 10000
    pos = pos.round(6)
    # pnl = pos * rtn_1p
    # pnl = (pos * rtn_1p).sum(axis=1)
    pos_in_coin = pos.div(ma_price, fill_value=np.nan)
    pos_in_coin.index = pos_in_coin.index + timedelta(hours=8)
    # # pos = pos.where(pos.abs() >= 1e-4, 0)
    ps_to_st_report_python_temp(mysqldb_conn, cursor, pos_in_coin, acname, fromtime, ifdelete=True)