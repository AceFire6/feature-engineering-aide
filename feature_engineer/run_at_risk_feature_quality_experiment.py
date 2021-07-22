from feature_engineer.run_feature_quality_experiment import AutoMLFeatureQualityExperiment


if __name__ == '__main__':
    at_risk_experiment_runner = AutoMLFeatureQualityExperiment(
        './experiments/at_risk.toml',
        run_experiments_n_times=10,
        use_random_seeds=True,
    )
    experiment_results = at_risk_experiment_runner.run_experiments()
    print(f'Experiments finished - {experiment_results}')
