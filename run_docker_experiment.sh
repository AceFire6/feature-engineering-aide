#!/usr/bin/env bash

docker run \
        --mount type=bind,source="$(pwd)/data_files",target=/feature_engineering_aide/data_files/ \
        --mount type=bind,source="$(pwd)/experiments",target=/feature_engineering_aide/experiments/ \
        feature-engineering-aide:latest \
        experiment_config/run_automl_experiments.py "$1"
