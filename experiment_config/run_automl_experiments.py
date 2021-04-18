from datetime import datetime
from functools import partial
import sys
from typing import List

from autosklearn.classification import AutoSklearnClassifier
from autosklearn.metrics import make_scorer
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import RFE, SelectKBest, SelectPercentile

from experiment_config.experiment import Experiment, parse_experiment_paths
from experiment_config.settings import N_JOBS, SUPPORTED_METRICS, TASK_TIME, TIME_PER_RUN
from experiment_config.utils import print_metric_results_five_number_summary


def run_experiments(experiments):
    for experiment in experiments:
        decision_tree_rfe = partial(RFE, estimator=DecisionTreeClassifier())
        smol_k_best = partial(SelectKBest, k=len(experiment.prediction_data_columns) // 2)

        for preprocessor_class in [None, smol_k_best, SelectPercentile, decision_tree_rfe]:
            metric_scorer = make_scorer(
                name='MCC Score',
                score_func=SUPPORTED_METRICS["Matthew's Correlation Coefficient"],
                optimum=1.0,
                worst_possible_result=-1.0,
            )
            classifier = AutoSklearnClassifier(
                include_preprocessors=['no_preprocessing'],
                time_left_for_this_task=TASK_TIME,
                per_run_time_limit=TIME_PER_RUN,
                n_jobs=N_JOBS,
                resampling_strategy=LeaveOneGroupOut,
                resampling_strategy_arguments={'groups': experiment.groups},
            )

            # TODO: Ensure this works
            features_selected = experiment.X.axes
            if preprocessor_class is not None:
                preprocessor = preprocessor_class()
                preprocessor = preprocessor.fit(experiment.X, experiment.y)
                preprocessor.transform(experiment.X)
                print(experiment.X.head(1))
                features_selected = experiment.X.axes

            classifier.fit(experiment.X.copy(), experiment.y.copy(), metric=metric_scorer)

            leave_one_out = LeaveOneGroupOut()
            test_train_splitter = leave_one_out.split(experiment.X, experiment.y, experiment.groups)
            for train_index, test_index in test_train_splitter:
                x_train, x_test = experiment.get_x_train_test_split(train_index, test_index)
                y_train, y_test = experiment.get_y_train_test_split(train_index, test_index)

                classifier.refit(x_train, y_train)
                y_hat = classifier.predict(x_test)

                for metric, metric_function in experiment.metrics.items():
                    metric_result = metric_function(y_test, y_hat)
                    experiment.add_result(metric, metric_result)

            now = f'{datetime.now():%Y-%m-%d_%H:%M:%S}'

            if preprocessor_class is None:
                preprocessor_name = 'no_preprocessor'
            elif hasattr(preprocessor_class, 'func'):
                preprocessor_name = preprocessor_class.func.__name__
            else:
                preprocessor_name = preprocessor_class.__name__

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
                    f'features_used = {features_selected}\n',
                    f'{classifier.sprint_statistics()}\n',
                ]
                results_file.writelines(result_output)

                print_metric_results_five_number_summary(experiment.metric_results, results_file)


if __name__ == '__main__':
    experiment_input_paths = sys.argv[1:]
    if not experiment_input_paths:
        print('Please pass in experiment files as arguments to this script')

    experiment_list: List[Experiment] = parse_experiment_paths(experiment_input_paths)
    run_experiments(experiment_list)
