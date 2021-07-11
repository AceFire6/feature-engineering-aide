#!/usr/bin/env bash

docker run \
        --mount "type=bind,source=$(pwd)/data_files,target=/feature_engineering_aide/data_files/" \
        --mount "type=bind,source=$(pwd)/experiments,target=/feature_engineering_aide/experiments/" \
        --mount "type=volume,source=results,target=/feature_engineering_aide/experiments/results/" \
        --env FEA_N_JOBS=4 \
        --env FEA_TOTAL_MEMORY_LIMIT=8000 \
        --env FEA_TASK_TIME=7200 \
        feature-engineering-aide:latest \
        feature_engineer/run_feature_quality_experiment.py "$1"
