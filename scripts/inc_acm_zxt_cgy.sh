#!/bin/bash

# 设置每个任务的开关
RUN_TEST=false
RUN_CONCAT=false
RUN_EVAL=false
RUN_CLUSTER=false
RUN_MODELRIDGE=true
RUN_MODELESMB=true
RUN_MERGE=true

start_date="20241201"
end_date="20241217"
mode="rolling"

batch_test_name="batch_inc_test_agg_241214_zxt_cgy"
batch_concat_name="batch_concat_test_agg_241214_zxt_cgy"
eval1_name="agg_241214_cgy_only"
eval2_name="agg_241214_cgy_zxt"
cluster1_name="agg_241214_cgy_double3m"
cluster2_name="agg_241214_cgy_zxt_double3m"
model_ridge1_name="ridge_v13_agg_241214_cgy_double3m_15d"
model_esmb1_name="esmb_objv1_agg_241214_cgy_double3m_15d"
model_ridge2_name="ridge_v13_agg_241214_cgy_zxt_double3m_15d"
model_esmb2_name="esmb_objv1_agg_241214_cgy_zxt_double3m_15d"
merge1_name="merge_agg_241214_cgy_double3m_15d_73"
merge2_name="merge_agg_241214_cgy_zxt_double3m_15d_73"


# 任务命令
test="python3 test_factors_by_batch.py -btn ${batch_test_name} -dt $end_date > .logs/${batch_test_name}.log 2>&1"
echo $test
concat="python3 concat_factors_by_batch.py -bcn ${batch_concat_name} > .logs/${batch_concat_name}.log 2>&1"
eval1="python3 rolling_evaluation.py -e ${eval1_name} -er double_3m -pst $start_date -pu $end_date -et $mode -wkr 100 > .logs/${eval1_name}.log 2>&1"
eval2="python3 rolling_evaluation.py -e ${eval2_name} -er double_3m -pst $start_date -pu $end_date -et $mode -wkr 100 > .logs/${eval2_name}.log 2>&1"
cluster1="python3 run_rolling_cluster.py -c $cluster1_name -pst $start_date -pu $end_date -ct $mode > .logs/${cluster1_name}.log 2>&1"
cluster2="python3 run_rolling_cluster.py -c $cluster2_name -pst $start_date -pu $end_date -ct $mode > .logs/${cluster2_name}.log 2>&1"
model_ridge1="python3 rolling_fit_pred_backtest.py -t ${model_ridge1_name} -pst $start_date -pu $end_date -m $mode -wkr 100 > .logs/${model_ridge1_name}.log 2>&1"
model_esmb1="python3 rolling_fit_pred_backtest.py -t ${model_esmb1_name} -pst $start_date -pu $end_date -m $mode -wkr 100 > .logs/${model_esmb1_name}.log 2>&1"
model_ridge2="python3 rolling_fit_pred_backtest.py -t ${model_ridge2_name} -pst $start_date -pu $end_date -m $mode -wkr 100 > .logs/${model_ridge2_name}.log 2>&1"
model_esmb2="python3 rolling_fit_pred_backtest.py -t ${model_esmb2_name} -pst $start_date -pu $end_date -m $mode -wkr 100 > .logs/${model_esmb2_name}.log 2>&1"
merge1="python3 run_merge_models.py -m ${merge1_name} > .logs/${merge1_name}.log 2>&1"
merge2="python3 run_merge_models.py -m ${merge2_name} > .logs/${merge2_name}.log 2>&1"


# 执行任务的函数
run_task() {
    local task_cmd=$1
    local task_name=$2
    local task_switch=$3

    if [ "$task_switch" = true ]; then
        echo "Running $task_name..."
        if eval "$task_cmd"; then
            echo "$task_name succeeded." 
        else
            echo "$task_name failed." 
            exit 1
        fi
    else
        echo "$task_name is skipped."
    fi
}

# 顺序执行任务
run_task "$test" "TEST" "$RUN_TEST" &&
run_task "$concat" "CONCAT" "$RUN_CONCAT" &&
run_task "$eval1" "EVAL1" "$RUN_EVAL" &&
run_task "$eval2" "EVAL2" "$RUN_EVAL" &&
run_task "$cluster1" "CLUSTER1" "$RUN_CLUSTER" &&
run_task "$cluster2" "CLUSTER2" "$RUN_CLUSTER" &&
run_task "$model_ridge1" "MODELRIDGE1" "$RUN_MODELRIDGE" &&
run_task "$model_esmb1" "MODELESMB1" "$RUN_MODELESMB" &&
run_task "$model_ridge2" "MODELRIDGE2" "$RUN_MODELRIDGE" &&
run_task "$model_esmb2" "MODELESMB2" "$RUN_MODELESMB" &&
run_task "$merge1" "MERGE2" "$RUN_MERGE"
run_task "$merge2" "MERGE2" "$RUN_MERGE"

echo "All tasks completed."