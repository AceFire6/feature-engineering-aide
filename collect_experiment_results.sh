#!/usr/bin/env bash

docker run \
        --mount "type=volume,source=results,target=/feature_engineering_aide/experiments/results/" \
        --mount "type=bind,source=$(pwd)/collected_results,target=/feature_engineering_aide/collected_results/" \
        --entrypoint=/bin/cp \
        feature-engineering-aide:latest \
        -R /feature_engineering_aide/experiments/results/ /feature_engineering_aide/collected_results/
