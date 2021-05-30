#!/usr/bin/env bash

docker run \
        --mount "type=bind,source=$(pwd)/data_files,target=/feature_engineering_aide/data_files/" \
        --mount "type=bind,source=$(pwd)/experiments,target=/feature_engineering_aide/experiments/" \
        --mount "type=volume,source=results,target=/feature_engineering_aide/results/" \
        --env FEA_N_JOBS=8 \
        --env FEA_TOTAL_MEMORY_LIMIT=10000 \
        --env FEA_TASK_TIME=86400 \
        feature-engineering-aide:latest \
        experiment_config/run_automl_experiments.py "$1"
