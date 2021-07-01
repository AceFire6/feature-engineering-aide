from datetime import datetime
from io import StringIO
import sys

from autosklearn.estimators import AutoSklearnClassifier
from autosklearn.metrics import make_scorer
from sklearn.metrics import matthews_corrcoef
from sklearn.model_selection import LeaveOneGroupOut

from experiment_config.experiment import Experiment, parse_experiment_paths
from experiment_config.settings import MEMORY_LIMIT, N_JOBS, TASK_TIME, TIME_PER_RUN, TOTAL_MEMORY_LIMIT
from experiment_config.utils import print_metric_results_five_number_summary, write_results


def run_experiment(experiment_counter: str, experiment: Experiment) -> None:
    metrics_as_scorers = [make_scorer(name, score_func) for name, score_func in experiment.metrics.items()]

    preprocessor_count = len(experiment.feature_preprocessors)
    for p_index, (preprocessor_name, preprocessor_class) in enumerate(experiment.feature_preprocessors.items()):
        preprocessor_start = datetime.now()
        preprocessor_counter = f'[{p_index + 1}/{preprocessor_count}]'
        print(
            f'{experiment_counter} Running preprocessor {preprocessor_name} '
            f'{preprocessor_counter} - experiment {experiment.name} - {preprocessor_start:%Y-%m-%d_%H:%M:%S}',
        )
        classifier = AutoSklearnClassifier(
            include_estimators=['decision_tree', 'random_forest', 'gaussian_nb'],
            time_left_for_this_task=TASK_TIME,
            per_run_time_limit=TIME_PER_RUN,
            n_jobs=N_JOBS,
            # Each one of the N_JOBS jobs is allocated MEMORY_LIMIT
            memory_limit=MEMORY_LIMIT,
            resampling_strategy=LeaveOneGroupOut,
            resampling_strategy_arguments={'groups': experiment.groups},
            ensemble_size=1,
            # metric=metric_scorer,
            scoring_functions=metrics_as_scorers,
            seed=experiment.seed,
        )

        features_selected = experiment.prediction_data_columns
        if preprocessor_class is not None:
            preprocessor = preprocessor_class().fit(experiment.X, experiment.y)
            features_selected_mask = preprocessor.get_support()
            features_selected = [column for column in experiment.X.columns[features_selected_mask]]

        training_data = experiment.X[features_selected].copy()
        targets = experiment.y.copy()
        classifier.fit(
            X=training_data,
            y=targets,
            X_test=experiment.holdout_x[features_selected],
            y_test=experiment.holdout_y,
            dataset_name=f'{experiment.name} - {preprocessor_name}',
        )

        leave_one_out = LeaveOneGroupOut()
        test_train_splitter = leave_one_out.split(training_data, targets, experiment.groups)
        for train_index, test_index in test_train_splitter:
            x_train, x_test = experiment.get_split(training_data, train_index, test_index)
            y_train, y_test = experiment.get_split(targets, train_index, test_index)

            # Value of the group used for splitting
            split_value = experiment.groups.iloc[test_index].unique()[0]

            classifier.refit(x_train, y_train)
            y_hat = classifier.predict(x_test)

            for metric, metric_function in experiment.metrics.items():
                metric_result = metric_function(y_test, y_hat)
                experiment.add_result(metric, metric_result, label=split_value)

        classifier.refit(training_data, targets)
        holdout_prediction = classifier.predict(experiment.holdout_x[features_selected])
        holdout_mcc_result = matthews_corrcoef(experiment.holdout_y, holdout_prediction)

        results_io = StringIO()
        print_metric_results_five_number_summary(experiment.metric_results, results_io)

        preprocessor_time_taken = datetime.now() - preprocessor_start
        write_results(
            experiment,
            f'{preprocessor_name}\n',
            f'\ttime_taken = {preprocessor_time_taken}\n',
            f'\tfeatures_used = {features_selected}\n',
            f'\tholdout_mcc_result = {holdout_mcc_result}\n',
            f'\t{classifier.sprint_statistics()}\n\n',
            *results_io.readlines(),
        )


def run_experiments(experiments: list[Experiment]) -> None:
    total_experiments = len(experiments)
    print(
        f'Running {total_experiments} experiment(s)!',
        'Config:',
        f'\tn_jobs = {N_JOBS}',
        f'\ttotal_time = {TASK_TIME}',
        f'\ttime_per_run = {TIME_PER_RUN}',
        f'\ttotal_memory_limit = {TOTAL_MEMORY_LIMIT}',
        f'\tmemory_limit_per_run = {MEMORY_LIMIT}',
        sep='\n',
    )

    for index, experiment in enumerate(experiments):
        start_time = f'{datetime.now():%Y-%m-%d_%H:%M:%S}'
        experiment.start_time = start_time
        experiment_counter = f'[{index + 1}/{total_experiments}]'
        print(f'{experiment_counter} Starting experiment {experiment.name} at {start_time}')

        write_results(
            experiment,
            'Config:\n',
            f'\tn = {experiment.training_set_sample_size()}\n',
            f'\tn_jobs = {N_JOBS}\n',
            f'\ttotal_time = {TASK_TIME}\n',
            f'\ttime_per_run = {TIME_PER_RUN}\n',
            f'\ttotal_memory_limit = {TOTAL_MEMORY_LIMIT}\n',
            f'\tmemory_limit_per_run = {MEMORY_LIMIT}\n\n',
        )

        run_experiment(experiment_counter, experiment)


if __name__ == '__main__':
    experiment_input_paths = sys.argv[1:]
    if not experiment_input_paths:
        print('Please pass in experiment files as arguments to this script')

    experiment_list = parse_experiment_paths(experiment_input_paths)
    run_experiments(experiment_list)
    print('Experiments finished!')
