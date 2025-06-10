#!/bin/bash

# 设置每个任务的开关
RUN_TEST=true
RUN_CONCAT=true
RUN_EVAL=true
RUN_CLUSTER=true
RUN_MODELRIDGE=true
RUN_MODELESMB=true
RUN_MERGE=true


# 任务命令
test="python3 test_factors_by_batch.py -btn batch_inc_test_agg_241029 -dt 20241031 > .logs/batch_inc_test_agg_241029.log 2>&1"
concat="python3 concat_factors_by_batch.py -bcn batch_concat_test_agg_241029 > .logs/batch_concat_test_agg_241029.log 2>&1"
eval="python3 rolling_evaluation.py -e agg_241029_updated -er double_3m -pst 20240701 -pu 20241101 -et rolling -wkr 100 > .logs/agg_241029_updated.log 2>&1"
cluster="python3 run_rolling_cluster.py -c agg_241029_updated_double3m -pst 20240701 -pu 20241101 -ct rolling > .logs/agg_241029_updated_double3m.log 2>&1"
model_ridge="python3 rolling_fit_pred_backtest.py -t ridge_v13_agg_241029_double3m_15d -pst 20240701 -pu 20241101 -m rolling -wkr 100 > .logs/ridge_v13_agg_241029_double3m_15d.log 2>&1"
model_esmb="python3 rolling_fit_pred_backtest.py -t esmb_objv1_agg_241029_double3m_15d -pst 20240701 -pu 20241101 -m rolling -wkr 100 > .logs/esmb_objv1_agg_241029_double3m_15d.log 2>&1"
merge="python3 run_merge_models.py -m merge_agg_241029_double3m_15d_73 > .logs/merge_agg_241029_double3m_15d_73.log 2>&1"


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
run_task "$eval" "EVAL" "$RUN_EVAL" &&
run_task "$cluster" "CLUSTER" "$RUN_CLUSTER" &&
run_task "$model_ridge" "MODELRIDGE" "$RUN_MODELRIDGE" &&
run_task "$model_esmb" "MODELESMB" "$RUN_MODELESMB" &&
run_task "$merge" "MERGE" "$RUN_MERGE"

echo "All tasks completed."