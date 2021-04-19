from datetime import datetime
from functools import partial
import sys
from typing import List

from autosklearn.classification import AutoSklearnClassifier
from autosklearn.metrics import make_scorer
from numpy.ma import MaskedArray
import orjson
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import RFE, SelectKBest, SelectPercentile

from experiment_config.experiment import Experiment, parse_experiment_paths
from experiment_config.settings import (
    MEMORY_LIMIT,
    N_JOBS,
    SUPPORTED_METRICS,
    TASK_TIME,
    TIME_PER_RUN,
)
from experiment_config.utils import print_metric_results_five_number_summary


def serialize_numpy(obj):
    if isinstance(obj, MaskedArray):
        return obj.tolist()

    raise TypeError


def run_experiments(experiments):
    total_experiments = len(experiments)
    print(f'Running {total_experiments} experiment(s)!')

    for index, experiment in enumerate(experiments):
        now = f'{datetime.now():%Y-%m-%d_%H:%M:%S}'
        experiment_counter = f'[{index + 1}/{total_experiments}]'
        print(f'{experiment_counter} Starting experiment {experiment.name} at {now}')

        decision_tree_rfe = partial(RFE, estimator=DecisionTreeClassifier())
        smol_k_best = partial(SelectKBest, k=len(experiment.prediction_data_columns) // 2)

        metrics_as_scorers = [
            make_scorer(name, score_func)
            for name, score_func in experiment.metrics.items()
        ]

        preprocessor_map = {
            'no_preprocessor': None,
            'SelectKBest': smol_k_best,
            'SelectPercentile': SelectPercentile,
            'DecisionTreeRFE': decision_tree_rfe,
        }
        preprocessor_count = len(preprocessor_map)

        for p_index, (preprocessor_name, preprocessor_class) in enumerate(preprocessor_map.items()):
            preprocessor_start = f'{datetime.now():%Y-%m-%d_%H:%M:%S}'
            preprocessor_counter = f'[{p_index + 1}/{preprocessor_count}]'
            print(
                f'{experiment_counter} Running preprocessor {preprocessor_name} '
                f'{preprocessor_counter} - experiment {experiment.name} - {preprocessor_start}',
            )
            # metric_scorer = make_scorer(
            #     name='MCC Score',
            #     score_func=SUPPORTED_METRICS["Matthew's Correlation Coefficient"],
            #     optimum=1.0,
            #     worst_possible_result=-1.0,
            # )
            classifier = AutoSklearnClassifier(
                include_preprocessors=['no_preprocessing'],
                time_left_for_this_task=TASK_TIME,
                per_run_time_limit=TIME_PER_RUN,
                n_jobs=N_JOBS,
                # Each one of the N_JOBS jobs is allocated MEMORY_LIMIT
                memory_limit=MEMORY_LIMIT,
                resampling_strategy=LeaveOneGroupOut,
                resampling_strategy_arguments={'groups': experiment.groups},
                # metric=metric_scorer,
                scoring_functions=metrics_as_scorers,
                seed=experiment.seed,
            )

            # TODO: Ensure this works
            features_selected = experiment.X.axes
            if preprocessor_class is not None:
                preprocessor = preprocessor_class()
                preprocessor = preprocessor.fit(experiment.X, experiment.y)
                preprocessor.transform(experiment.X)
                print(experiment.X.head(1))
                features_selected = experiment.X.axes

            classifier.fit(experiment.X.copy(), experiment.y.copy())

            leave_one_out = LeaveOneGroupOut()
            test_train_splitter = leave_one_out.split(experiment.X, experiment.y, experiment.groups)
            for train_index, test_index in test_train_splitter:
                x_train, x_test = experiment.get_x_train_test_split(train_index, test_index)
                y_train, y_test = experiment.get_y_train_test_split(train_index, test_index)

                classifier.refit(x_train, y_train)
                y_hat = classifier.predict(x_test)

                for metric, metric_function in experiment.metrics.items():
                    metric_result = metric_function(y_test, y_hat)
                    split_value = experiment.groups.iloc[test_index].unique()[0]
                    experiment.add_result(metric, metric_result, label=split_value)

            results_file_name = (
                f'{experiment.name}-{preprocessor_name}_automl_mcc_results_{now}.txt'
            )
            results_file_path = experiment.file_path.parent / 'results' / experiment.name
            results_file_path.mkdir(parents=True, exist_ok=True)

            results_file_path_with_name = results_file_path / results_file_name

            with results_file_path_with_name.open('w') as results_file:
                result_output = [
                    f'n = {experiment.training_set_sample_size()}\n',
                    f'n_jobs = {N_JOBS}\n',
                    f'total_time = {TASK_TIME}\n',
                    f'time_per_run = {TIME_PER_RUN}\n',
                    f'memory_limit = {MEMORY_LIMIT}\n',
                    f'features_used = {features_selected}\n',
                    f'{classifier.sprint_statistics()}\n',
                ]
                results_file.writelines(result_output)

                print_metric_results_five_number_summary(experiment.metric_results, results_file)
                print(
                    orjson.dumps(
                        classifier.cv_results_,
                        option=orjson.OPT_SERIALIZE_NUMPY,
                        default=serialize_numpy
                    ),
                    file=results_file,
                )


if __name__ == '__main__':
    experiment_input_paths = sys.argv[1:]
    if not experiment_input_paths:
        print('Please pass in experiment files as arguments to this script')

    experiment_list: List[Experiment] = parse_experiment_paths(experiment_input_paths)
    run_experiments(experiment_list)
    print('Experiments finished!')
