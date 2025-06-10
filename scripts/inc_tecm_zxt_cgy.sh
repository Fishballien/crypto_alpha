#!/bin/bash

# 设置每个任务的开关
RUN_TEST=true
RUN_CONCAT=true
RUN_EVAL=true
RUN_CLUSTER=true
RUN_MODELRIDGE=true
RUN_MODELESMB=true
RUN_MERGE=true

start_date="20241101"
end_date='20241130"
mode="rolling"


# 任务命令
test="python3 test_factors_by_batch.py -btn batch_inc_test_agg_241114_zxt_cgy -dt $end_date > .logs/batch_inc_test_agg_241114_zxt_cgy.log 2>&1"
concat="python3 concat_factors_by_batch.py -bcn batch_concat_test_agg_241114_zxt_cgy > .logs/batch_concat_test_agg_241114_zxt_cgy.log 2>&1"
eval1="python3 rolling_evaluation.py -e agg_241113_cgy_only -er double_3m -pst $start_date -pu $end_date -et $mode -wkr 100 > .logs/agg_241113_cgy_only.log 2>&1"
eval2="python3 rolling_evaluation.py -e agg_241114_zxt_cgy -er double_3m -pst $start_date -pu $end_date -et $mode -wkr 100 > .logs/agg_241114_zxt_cgy.log 2>&1"
cluster1="python3 run_rolling_cluster.py -c agg_241113_double3m -pst $start_date -pu $end_date -ct $mode > .logs/agg_241113_double3m.log 2>&1"
cluster2="python3 run_rolling_cluster.py -c agg_241114_zxt_cgy_double3m -pst $start_date -pu $end_date -ct $mode > .logs/agg_241114_zxt_cgy_double3m.log 2>&1"
model_ridge1="python3 rolling_fit_pred_backtest.py -t ridge_v13_agg_241113_double3m_15d -pst $start_date -pu $end_date -m $mode -wkr 100 > .logs/ridge_v13_agg_241113_double3m_15d.log 2>&1"
model_esmb1="python3 rolling_fit_pred_backtest.py -t esmb_objv1_agg_241113_double3m_15d -pst $start_date -pu $end_date -m $mode -wkr 100 > .logs/esmb_objv1_agg_241113_double3m_15d.log 2>&1"
model_ridge2="python3 rolling_fit_pred_backtest.py -t ridge_v13_agg_241114_zxt_cgy_double3m_15d -pst $start_date -pu $end_date -m $mode -wkr 100 > .logs/ridge_v13_agg_241114_zxt_cgy_double3m_15d.log 2>&1"
model_esmb2="python3 rolling_fit_pred_backtest.py -t esmb_objv1_agg_241114_zxt_cgy_double3m_15d -pst $start_date -pu $end_date -m $mode -wkr 100 > .logs/esmb_objv1_agg_241114_zxt_cgy_double3m_15d.log 2>&1"
merge1="python3 run_merge_models.py -m merge_agg_241113_double3m_15d_73 > .logs/merge_agg_241113_double3m_15d_73.log 2>&1"
merge21="python3 run_merge_models.py -m merge_agg_241114_zxt_cgy_double3m_15d_73 > .logs/merge_agg_241114_zxt_cgy_double3m_15d_73.log 2>&1"


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
run_task "$merge1" "MERGE2" "$RUN_MERGE"

echo "All tasks completed."