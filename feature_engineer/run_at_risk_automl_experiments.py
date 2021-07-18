from feature_engineer.run_automl_experiments import AutoMLPreprocessorExperiment

if __name__ == '__main__':
    experiment_results = AutoMLPreprocessorExperiment(
        '../experiments/at_risk.toml',
        run_experiments_n_times=10,
        use_random_seeds=True,
    )
    print(f'Experiments finished - {experiment_results}')
