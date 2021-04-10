from datetime import datetime
import sys

from autosklearn.classification import AutoSklearnClassifier
from autosklearn.metrics import make_scorer
from sklearn.model_selection import LeaveOneGroupOut

from experiment_config.settings import N_JOBS, SUPPORTED_METRICS, TASK_TIME, TIME_PER_RUN
from experiment_config.utils import (
    parse_experiment_paths,
    print_metric_results_five_number_summary,
)

experiment_input_paths = sys.argv[1:]
if not experiment_input_paths:
    print('Please pass in experiment files as arguments to this script')

experiments = parse_experiment_paths(experiment_input_paths)

for experiment in experiments:
    classifier = AutoSklearnClassifier(
        time_left_for_this_task=TASK_TIME,
        per_run_time_limit=TIME_PER_RUN,
        n_jobs=N_JOBS,
        resampling_strategy=LeaveOneGroupOut,
        resampling_strategy_arguments={'groups': experiment.groups},
    )

    metric_scorer = make_scorer('MCC Score', SUPPORTED_METRICS["Matthew's Correlation Coefficient"])

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
    results_file_name = f'{experiment.name}_automl_mcc_results_{now}.txt'

    with open(results_file_name, 'w') as results_file:
        print(f'n = {experiment.training_set_sample_size()}', file=results_file)
        print(f'n_jobs = {N_JOBS}', file=results_file)
        print(f'total_time = {TASK_TIME}', file=results_file)
        print(f'time_per_run = {TIME_PER_RUN}', file=results_file)
        print(classifier.sprint_statistics(), file=results_file)

        print_metric_results_five_number_summary(experiment.metric_results, results_file)
