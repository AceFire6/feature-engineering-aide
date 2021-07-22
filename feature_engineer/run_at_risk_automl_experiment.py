from feature_engineer.run_automl_experiment import AutoMLPreprocessorExperiment


if __name__ == '__main__':
    at_risk_experiment_runner = AutoMLPreprocessorExperiment(
        './experiments/at_risk.toml',
        run_experiments_n_times=10,
        use_random_seeds=True,
    )
    experiment_results = at_risk_experiment_runner.run_experiments()
    print(f'Experiments finished - {experiment_results}')
