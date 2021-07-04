from datetime import datetime
import sys

from sklearn.model_selection import LeaveOneGroupOut
from tqdm import tqdm

from .experiment_config.experiment import parse_experiment_paths
from .experiment_runner.utils import print_metric_results_five_number_summary


def run_experiments(experiments):
    for experiment in experiments:
        classifier_result_metrics = {
            classifier: {metric: [] for metric in experiment.metrics}
            for classifier in experiment.classifiers
        }

        leave_one_out = LeaveOneGroupOut()

        for classifier_name, classifier_class in experiment.classifiers.items():
            print(f'Running {classifier_name}')
            test_train_splitter = tqdm(
                leave_one_out.split(experiment.X, experiment.y, experiment.groups),
            )

            for train_index, test_index in test_train_splitter:
                x_train, x_test = experiment.get_x_train_test_split(train_index, test_index)
                y_train, y_test = experiment.get_y_train_test_split(train_index, test_index)

                classifier = classifier_class()
                classifier.fit(x_train, y_train)

                y_hat = classifier.predict(x_test)

                for metric, metric_function in experiment.metrics.items():
                    metric_result = metric_function(y_test, y_hat)
                    classifier_result_metrics[classifier_name][metric].append(metric_result)

        now = f'{datetime.now():%Y-%m-%d_%H:%M:%S}'
        results_file_name = f'{experiment.name}_automl_mcc_results_{now}.txt'

        with open(results_file_name, 'w') as results_file:
            print(f'n = {experiment.training_set_sample_size()}', file=results_file)

            for classifier_name, result_metrics in classifier_result_metrics.items():
                print(classifier_name, file=results_file)

                print_metric_results_five_number_summary(result_metrics, results_file)


if __name__ == '__main__':
    experiment_input_paths = sys.argv[1:]
    if not experiment_input_paths:
        print('Please pass in experiment files as arguments to this script')

    experiments_list = parse_experiment_paths(experiment_input_paths)
    run_experiments(experiments_list)
