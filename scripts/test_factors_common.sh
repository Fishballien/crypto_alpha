#!/bin/bash

# 定义参数变量
process_name="LOB_2024-11-10_R2"
tag="cgy"
factor_data_dir="/mnt/30.89_guangyao_limitorderbook/LimitOrderBook"
#factor_data_dir="/mnt/Data/xintang/crypto/multi_factor/factor_test_by_alpha/results/model"
#factor_data_dir="/mnt/133.tianfang_crypto_fac/factors/neu"
test_name="regular_twd30_sp30" # regular_twd30
wkr=150

# 定义日志文件名
log_file="./.logs/${process_name}.log"

# 打印日志文件名
echo "Log file: $log_file"

# 运行指定的 Python 脚本并传递参数
cmd="nohup python3 test_factors.py -p $process_name -tag $tag -fdir $factor_data_dir  -t $test_name -wkr $wkr > $log_file 2>&1 &" # -fdir $factor_data_dir 
echo $cmd
eval $cmd

# 检查上一个命令的退出状态码
#if [ $? -eq 0 ]; then
#    echo "Test ran successfully."
#else
#    echo "Test encountered an error."
#fi
