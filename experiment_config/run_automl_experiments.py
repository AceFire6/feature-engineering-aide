from datetime import datetime
from functools import partial
import sys
from typing import List

from autosklearn.classification import AutoSklearnClassifier
from autosklearn.metrics import make_scorer
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import RFE, SelectKBest, SelectPercentile

from experiment_config.experiment import Experiment
from experiment_config.settings import SUPPORTED_METRICS
from experiment_config.utils import (
    parse_experiment_paths,
    print_metric_results_five_number_summary,
)

experiment_input_paths = sys.argv[1:]
if not experiment_input_paths:
    print('Please pass in experiment files as arguments to this script')

N_JOBS = 3
TASK_TIME = 1800
TIME_PER_RUN = TASK_TIME // 10

experiments: List[Experiment] = parse_experiment_paths(experiment_input_paths)

decision_tree_rfe = partial(RFE, estimator=DecisionTreeClassifier())

for experiment in experiments:
    smol_k_best = partial(SelectKBest, k=len(experiment.prediction_data_columns) // 2)

    for preprocessor_class in [None, smol_k_best, SelectPercentile, decision_tree_rfe]:
        classifier = AutoSklearnClassifier(
            include_preprocessors=['no_preprocessing'],
            time_left_for_this_task=TASK_TIME,
            per_run_time_limit=TIME_PER_RUN,
            n_jobs=N_JOBS,
            resampling_strategy=LeaveOneGroupOut,
            resampling_strategy_arguments={'groups': experiment.groups},
        )

        metric_scorer = make_scorer('MCC Score', SUPPORTED_METRICS["Matthew's Correlation Coefficient"])

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
            X_train, X_test = experiment.get_x_train_test_split(train_index, test_index)
            y_train, y_test = experiment.get_y_train_test_split(train_index, test_index)

            classifier.refit(X_train, y_train)
            y_hat = classifier.predict(X_test)

            for metric, metric_function in experiment.metrics.items():
                metric_result = metric_function(y_test, y_hat)
                experiment.add_result(metric, metric_result)

        now = f'{datetime.utcnow():%Y-%m-%d_%H:%M:%S}'

        if preprocessor_class is None:
            preprocessor_name = 'no_preprocessor'
        elif hasattr(preprocessor_class, 'func'):
            preprocessor_name = preprocessor_class.func.__name__
        else:
            preprocessor_name = preprocessor_class.__name__

        results_file_name = f'{experiment.name}-{preprocessor_name}_automl_mcc_results_{now}.txt'

        with open(results_file_name, 'w') as results_file:
            print(f'n = {experiment.training_set_sample_size()}', file=results_file)
            print(f'n_jobs = {N_JOBS}', file=results_file)
            print(f'total_time = {TASK_TIME}', file=results_file)
            print(f'time_per_run = {TIME_PER_RUN}', file=results_file)
            print(f'features_used = {features_selected}', file=results_file)
            print(classifier.sprint_statistics(), file=results_file)

            print_metric_results_five_number_summary(experiment.metric_results, results_file)
